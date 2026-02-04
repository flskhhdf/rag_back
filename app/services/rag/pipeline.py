# -*- coding: utf-8 -*-
"""
RAG Pipeline - 메인 파이프라인 클래스
"""
import re
import os
import time
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator

from .config import RAGConfig
from .retriever import hybrid_search, rerank_results, get_qdrant_client
from .context_expansion import expand_context_with_neighbors
from .llm_client import build_rag_messages, stream_llm_response
from app.core.structured_logger import get_logger

# 구조화된 로거
logger = get_logger(__name__)


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
        logger.info_event(
            "pipeline_initialized",
            "RAG Pipeline initialized",
            data={"llm_model": self.config.LLM_MODEL}
        )



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

            logger.info_event(
                "request_received",
                "New RAG request received",
                data={
                    "pdf_id": pdf_id,
                    "filename": filename,
                    "query": query[:200],  # Truncate long queries
                    "has_history": bool(chat_history)
                }
            )

            # 검색 분석: 요청 시작 헤더 (search_analysis.log에 기록)
            logger.info(
                f"========== NEW REQUEST ==========",
                event='request_header',
                event_type='search_analysis'
            )
            logger.info(
                f"Query: {query}",
                event='request_query',
                event_type='search_analysis',
                data={"query": query}
            )
            logger.info(
                f"PDF: {filename} (ID: {pdf_id})",
                event='request_pdf',
                event_type='search_analysis',
                data={"filename": filename, "pdf_id": pdf_id}
            )

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

            # RRF 점수 통계 계산
            rrf_scores = [r.get("rrf_score", 0) for r in search_results]
            rrf_stats = {}
            if rrf_scores:
                rrf_stats = {
                    "max_score": max(rrf_scores),
                    "min_score": min(rrf_scores),
                    "avg_score": sum(rrf_scores) / len(rrf_scores),
                    "top_10_scores": rrf_scores[:10],
                    "score_distribution": {
                        "above_0.8": len([s for s in rrf_scores if s > 0.8]),
                        "0.5_to_0.8": len([s for s in rrf_scores if 0.5 < s <= 0.8]),
                        "below_0.5": len([s for s in rrf_scores if s <= 0.5])
                    }
                }

            logger.info_event(
                "search_completed",
                "Hybrid search completed",
                duration_ms=t_search * 1000,
                data={
                    "results_count": len(search_results),
                    **rrf_stats
                }
            )

            if not search_results:
                logger.warning_event(
                    "search_no_results",
                    "No search results found"
                )
                yield "검색된 문서가 없습니다. 다른 질문을 시도해보세요."
                return

            # 검색 분석: RRF 융합 결과 전체 로깅 (search_analysis.log에 기록)
            logger.info(
                f"=== RRF Results (Total: {len(search_results)}) ===",
                event='rrf_stage_start',
                event_type='search_analysis'
            )
            for i, result in enumerate(search_results, 1):
                full_text = result.get("text", "")
                rrf_score = result.get("rrf_score", 0)
                payload = result.get("payload", {})

                # pages는 배열 형태로 저장됨 (예: [54, 55])
                pages = payload.get("metadata", {}).get("pages", [])
                if pages and isinstance(pages, list):
                    if len(pages) == 1:
                        page_no = str(pages[0])
                    else:
                        page_no = f"{pages[0]}-{pages[-1]}"  # 54-55 형태
                else:
                    page_no = "N/A"

                logger.info(
                    f"RRF result {i}: score={rrf_score:.4f}, page={page_no}\n{full_text}",
                    event='rrf_result',
                    event_type='search_analysis',
                    data={
                        "rank": i,
                        "rrf_score": rrf_score,
                        "page_no": page_no,
                        "full_text": full_text
                    }
                )

            # 디버깅: RRF 융합 직후 결과 미리보기 (DEBUG 레벨)
            for i, result in enumerate(search_results[:10], 1):
                text_preview = result.get("text", "")[:100].replace("\n", " ")
                rrf_score = result.get("rrf_score", 0)
                payload = result.get("payload", {})

                # pages는 배열 형태로 저장됨
                pages = payload.get("metadata", {}).get("pages", [])
                if pages and isinstance(pages, list):
                    page_no = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])
                else:
                    page_no = "N/A"

                logger.debug_event(
                    "rrf_result",
                    f"RRF result {i}",
                    data={
                        "rank": i,
                        "rrf_score": rrf_score,
                        "page_no": page_no,
                        "text_preview": text_preview
                    }
                )

            # 3. 리랭킹
            t_rerank_start = time.time()
            reranked_results = rerank_results(
                query=query,
                results=search_results,
                top_n=self.config.TOP_K_FINAL * 2
            )
            t_rerank = time.time() - t_rerank_start

            # Rerank 점수 통계 계산
            rerank_scores = [r.get("rerank_score", 0) for r in reranked_results]
            max_score = max(rerank_scores) if rerank_scores else 0  # 이후 코드에서 사용
            rerank_stats = {}
            if rerank_scores:
                rerank_stats = {
                    "max_score": max_score,
                    "min_score": min(rerank_scores),
                    "avg_score": sum(rerank_scores) / len(rerank_scores),
                    "top_10_scores": rerank_scores[:10],
                    "score_distribution": {
                        "above_0.8": len([s for s in rerank_scores if s > 0.8]),
                        "0.5_to_0.8": len([s for s in rerank_scores if 0.5 < s <= 0.8]),
                        "below_0.5": len([s for s in rerank_scores if s <= 0.5])
                    }
                }

            logger.info_event(
                "rerank_completed",
                "Reranking completed",
                duration_ms=t_rerank * 1000,
                data={
                    "results_count": len(reranked_results),
                    **rerank_stats
                }
            )

            # 검색 분석: 리랭킹 결과 전체 로깅 (search_analysis.log에 기록)
            logger.info(
                f"=== Rerank Results (Total: {len(reranked_results)}) ===",
                event='rerank_stage_start',
                event_type='search_analysis'
            )
            for i, result in enumerate(reranked_results, 1):
                full_text = result.get("text", "")
                rerank_score = result.get("rerank_score", 0)
                payload = result.get("payload", {})

                # pages는 배열 형태로 저장됨
                pages = payload.get("metadata", {}).get("pages", [])
                if pages and isinstance(pages, list):
                    page_no = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])
                else:
                    page_no = "N/A"

                logger.info(
                    f"Rerank result {i}: score={rerank_score:.4f}, page={page_no}\n{full_text}",
                    event='rerank_result',
                    event_type='search_analysis',
                    data={
                        "rank": i,
                        "rerank_score": rerank_score,
                        "page_no": page_no,
                        "full_text": full_text
                    }
                )

            # 디버깅: 리랭킹 후 결과 미리보기 (DEBUG 레벨)
            for i, result in enumerate(reranked_results[:10], 1):
                text_preview = result.get("text", "")[:100].replace("\n", " ")
                rerank_score = result.get("rerank_score", 0)
                payload = result.get("payload", {})

                # pages는 배열 형태로 저장됨
                pages = payload.get("metadata", {}).get("pages", [])
                if pages and isinstance(pages, list):
                    page_no = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])
                else:
                    page_no = "N/A"

                logger.debug_event(
                    "rerank_result",
                    f"Rerank result {i}",
                    data={
                        "rank": i,
                        "rerank_score": rerank_score,
                        "page_no": page_no,
                        "text_preview": text_preview
                    }
                )

            # 4. 검색 결과 신뢰도 체크
            if max_score < self.config.MIN_SEARCH_SCORE:
                logger.warning_event(
                    "score_threshold_not_met",
                    "Search scores too low, using conversation context only",
                    data={
                        "max_score": max_score,
                        "threshold": self.config.MIN_SEARCH_SCORE
                    }
                )
                
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

                logger.info_event(
                    "llm_response",
                    "LLM response completed (conversation-only)",
                    duration_ms=t_llm * 1000
                )

                logger.log_metrics({
                    "total_duration_ms": (t_search + t_rerank + t_llm) * 1000,
                    "search_duration_ms": t_search * 1000,
                    "rerank_duration_ms": t_rerank * 1000,
                    "llm_duration_ms": t_llm * 1000
                })
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
            logger.info_event(
                "context_retrieved",
                f"Retrieved {len(expanded_results)} chunks",
                data={
                    "chunks_count": len(expanded_results),
                    "scores": [
                        r.get("rerank_score") or r.get("rrf_score") or r.get("score", 0)
                        for r in expanded_results
                    ]
                }
            )

            if not contexts:
                logger.warning_event("no_contexts", "No contexts available after retrieval")
                yield "검색된 문서가 없습니다. 다른 질문을 시도해보세요."
                return

            # 6. Messages 배열 생성 및 LLM 스트리밍 (멀티턴 대화 지원)
            messages = build_rag_messages(
                query=query,
                results=expanded_results,  # payload 포함된 전체 결과 전달
                history=chat_history
            )

            # 검색 분석: LLM에 전달되는 최종 컨텍스트 로깅 (search_analysis.log에 기록)
            logger.info(
                f"=== Final Context to LLM (Total chunks: {len(expanded_results)}) ===",
                event='llm_input_start',
                event_type='search_analysis'
            )
            for i, result in enumerate(expanded_results, 1):
                full_text = result.get("expanded_text") or result.get("text", "")
                score = result.get("rerank_score") or result.get("rrf_score", 0)
                payload = result.get("payload", {})

                # pages는 배열 형태로 저장됨
                pages = payload.get("metadata", {}).get("pages", [])
                if pages and isinstance(pages, list):
                    page_no = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])
                else:
                    page_no = "N/A"

                neighbors_count = result.get("neighbors_count", 0)

                logger.info(
                    f"LLM context {i}: score={score:.4f}, page={page_no}, neighbors={neighbors_count}, text_length={len(full_text)}\n{full_text}",
                    event='llm_context',
                    event_type='search_analysis',
                    data={
                        "rank": i,
                        "score": score,
                        "page_no": page_no,
                        "neighbors_count": neighbors_count,
                        "text_length": len(full_text),
                        "full_text": full_text
                    }
                )

            t_llm_start = time.time()
            async for chunk in stream_llm_response(messages):
                yield chunk
            t_llm = time.time() - t_llm_start

            logger.info_event(
                "llm_response",
                "LLM response completed",
                duration_ms=t_llm * 1000
            )

            # 전체 성능 메트릭
            logger.log_metrics({
                "total_duration_ms": (t_search + t_rerank + t_llm) * 1000,
                "search_duration_ms": t_search * 1000,
                "rerank_duration_ms": t_rerank * 1000,
                "llm_duration_ms": t_llm * 1000
            })

            # 검색 분석: 파이프라인 완료 요약 (search_analysis.log에 기록)
            logger.info(
                f"=== Pipeline Summary ===",
                event='pipeline_summary',
                event_type='search_analysis'
            )
            logger.info(
                f"Total duration: {(t_search + t_rerank + t_llm) * 1000:.2f}ms | "
                f"Search: {t_search * 1000:.2f}ms | Rerank: {t_rerank * 1000:.2f}ms | LLM: {t_llm * 1000:.2f}ms",
                event='pipeline_timing',
                event_type='search_analysis',
                data={
                    "total_ms": (t_search + t_rerank + t_llm) * 1000,
                    "search_ms": t_search * 1000,
                    "rerank_ms": t_rerank * 1000,
                    "llm_ms": t_llm * 1000
                }
            )
            logger.info(
                f"========== REQUEST COMPLETED ==========\n",
                event='request_footer',
                event_type='search_analysis'
            )

            logger.info_event(
                "pipeline_completed",
                "RAG pipeline completed successfully",
                duration_ms=(t_search + t_rerank + t_llm) * 1000
            )

        except Exception as e:
            import traceback
            logger.error_event(
                "pipeline_error",
                f"Pipeline error: {str(e)}",
                data={"traceback": traceback.format_exc()}
            )
            yield f"오류가 발생했습니다: {str(e)}"
