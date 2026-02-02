# -*- coding: utf-8 -*-
"""
RAG Pipeline - 메인 파이프라인 클래스
"""
import re
import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator

from .config import RAGConfig
from .retriever import hybrid_search, rerank_results, get_qdrant_client
from .context_expansion import expand_context_with_neighbors
from .llm_client import build_rag_messages, stream_llm_response

# 로거 설정
logger = logging.getLogger(__name__)

# 로그 파일 설정
log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'logs')
os.makedirs(log_dir, exist_ok=True)


def _sanitize_collection_name(filename: str) -> str:
    """파일명을 Qdrant 컬렉션 이름으로 변환"""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    name = re.sub(r'_{2,}', '_', name)
    name = name.strip('_')
    if not name:
        name = "pdf_document"
    return name.lower()


class RAGPipeline:
    """
    RAG 파이프라인 메인 클래스
    
    Features:
    - 다중 쿼리 생성 (Query Expansion)
    - 하이브리드 검색 (Dense + Sparse + RRF)
    - Cross-Encoder Reranking
    - 이웃 청크 확장 (Context expansion)
    - 병렬 처리
    """

    def __init__(self):
        self.config = RAGConfig()
        logger.info(f"[RAGPipeline] Initialized with model: {self.config.LLM_MODEL}")



    async def generate_response_stream(
        self,
        pdf_id: str,
        filename: str,
        query: str,
        chat_history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        RAG 응답 스트리밍 생성

        Args:
            pdf_id: PDF ID
            filename: PDF 파일명
            query: 사용자 질문
            chat_history: 대화 히스토리

        Yields:
            스트리밍 응답 청크
        """
        try:
            # pdf_id를 컬렉션 이름으로 사용 (한글 파일명 충돌 방지)
            collection_name = pdf_id
            
            logger.info("="*80)
            logger.info(f"[NEW QUERY] PDF: {filename}, Query: {query}")
            logger.info("-"*80)

            # 1. 대화 히스토리 포맷팅 (최근 6턴 = 12개 메시지)
            prior_dialog = ""
            if chat_history:
                recent = chat_history[-12:]
                prior_dialog = "\n".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}"
                    for msg in recent
                ])

            # 2. 하이브리드 검색 (임베딩 생성 + 벡터 검색)
            t_search_start = time.time()
            search_results = await hybrid_search(collection_name, query)
            t_search = time.time() - t_search_start
            logger.info(f"⏱️  [1] Hybrid Search (Embedding + Retrieval): {t_search:.3f}s")

            if not search_results:
                logger.warning("No search results found")
                yield "검색된 문서가 없습니다. 다른 질문을 시도해보세요."
                return

            # 디버깅: RRF 융합 직후 결과 미리보기
            logger.info(f"[DEBUG] RRF Fusion results (top 10):")
            for i, result in enumerate(search_results[:10], 1):
                text_preview = result.get("text", "")[:150].replace("\n", " ")
                rrf_score = result.get("rrf_score", 0)
                payload = result.get("payload", {})
                metadata = payload.get("metadata", {})
                page_no = metadata.get("page_no", "N/A")
                logger.info(f"  [{i}] RRF={rrf_score:.4f} | Page={page_no} | Text: {text_preview}...")

            # 3. 리랭킹
            t_rerank_start = time.time()
            reranked_results = rerank_results(
                query=query,
                results=search_results,
                top_n=self.config.TOP_K_FINAL * 2
            )
            t_rerank = time.time() - t_rerank_start
            logger.info(f"⏱️  [2] Reranking: {t_rerank:.3f}s")
            logger.info(f"After reranking: {len(reranked_results)} results")

            # 디버깅: 리랭킹 후 결과 미리보기
            logger.info(f"[DEBUG] After reranking (top 10):")
            for i, result in enumerate(reranked_results[:10], 1):
                text_preview = result.get("text", "")[:150].replace("\n", " ")
                rerank_score = result.get("rerank_score", 0)
                payload = result.get("payload", {})
                metadata = payload.get("metadata", {})
                page_no = metadata.get("page_no", "N/A")
                logger.info(f"  [{i}] Rerank={rerank_score:.4f} | Page={page_no} | Text: {text_preview}...")

            # 4. 검색 결과 신뢰도 체크
            max_score = max([r.get("rerank_score") or r.get("rrf_score") or r.get("score", 0) for r in reranked_results])
            logger.info(f"[SCORE CHECK] Max rerank score: {max_score:.4f}, Threshold: {self.config.MIN_SEARCH_SCORE}")
            
            if max_score < self.config.MIN_SEARCH_SCORE:
                logger.warning(f"⚠️  Search scores too low (max={max_score:.4f} < {self.config.MIN_SEARCH_SCORE})")
                logger.warning("⚠️  Using conversation context only (no document context)")
                
                # 검색 결과 무시, 대화 히스토리만 사용
                messages = build_rag_messages(
                    query=query,
                    results=None,  # 검색 결과 없음
                    history=chat_history
                )
                
                t_llm_start = time.time()
                async for chunk in stream_llm_response(messages):
                    yield chunk
                t_llm = time.time() - t_llm_start
                logger.info(f"⏱️  [LLM] Response (conversation-only): {t_llm:.3f}s")
                return
            
            # 5. 이웃 청크 확장
            if self.config.NEIGHBOR_EXPAND > 0:
                expanded_results = expand_context_with_neighbors(
                    collection_name=collection_name,
                    results=reranked_results[:self.config.TOP_K_FINAL],
                    expand=self.config.NEIGHBOR_EXPAND
                )
            else:
                expanded_results = reranked_results[:self.config.TOP_K_FINAL]

            # 5. 컨텍스트 추출 (하위 호환성)
            contexts = []
            for result in expanded_results:
                text = result.get("expanded_text") or result.get("text", "")
                if text:
                    contexts.append(text)

            # 로그
            logger.info(f"[RETRIEVED] {len(expanded_results)} chunks")
            for i, r in enumerate(expanded_results, 1):
                score = r.get("rerank_score") or r.get("rrf_score") or r.get("score", 0)
                logger.info(f"  Chunk {i}: rerank_score={score:.6f}")
            logger.info("="*80)

            if not contexts:
                yield "검색된 문서가 없습니다. 다른 질문을 시도해보세요."
                return

            # 6. Messages 배열 생성 및 LLM 스트리밍 (멀티턴 대화 지원)
            messages = build_rag_messages(
                query=query,
                results=expanded_results,  # payload 포함된 전체 결과 전달
                history=chat_history
            )

            t_llm_start = time.time()
            async for chunk in stream_llm_response(messages):
                yield chunk
            t_llm = time.time() - t_llm_start
            logger.info(f"⏱️  [3] LLM Response Generation: {t_llm:.3f}s")

            # 전체 소요 시간
            t_total = t_search + t_rerank + t_llm
            logger.info(f"⏱️  [TOTAL] RAG Pipeline: {t_total:.3f}s")

        except Exception as e:
            import traceback
            logger.error(f"Pipeline error: {e}\n{traceback.format_exc()}")
            yield f"오류가 발생했습니다: {str(e)}"
