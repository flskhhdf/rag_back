# -*- coding: utf-8 -*-
"""
Retriever Module - 검색, RRF 융합, 리랭킹
"""
import asyncio
import logging
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import SparseVector
from sentence_transformers import CrossEncoder

try:
    from transformers import AutoModel
    import torch
    _HAS_JINA = True
except ImportError:
    _HAS_JINA = False
    logger.warning("transformers and torch not available. Jina Reranker will not be available.")

from .config import RAGConfig
from .embeddings import embed_query, get_sparse_embeddings, has_sparse_support

logger = logging.getLogger(__name__)

# Reranker 싱글톤
_reranker = None


class JinaReranker:
    """
    Jina Reranker v3 공식 API
    
    Hugging Face: https://huggingface.co/jinaai/jina-reranker-v3
    """
    
    def __init__(self, model_name: str = "jinaai/jina-reranker-v3", device: str = "cuda"):
        if not _HAS_JINA:
            raise ImportError("transformers와 torch가 필요합니다: pip install transformers torch")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"[Jina Reranker] 모델 로딩: {model_name} (device: {self.device})")
        
        # Jina 공식 API: AutoModel 사용
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"[Jina Reranker] 로딩 완료")
    
    def predict(self, pairs: List[tuple]) -> List[float]:
        """
        CrossEncoder 호환 인터페이스
        
        Args:
            pairs: [(query, document), ...] 형식의 리스트
            
        Returns:
            각 pair에 대한 relevance score (0~1 범위)
        """
        if not pairs:
            return []
        
        query = pairs[0][0]  # 모든 pair는 같은 query를 가짐
        documents = [doc for _, doc in pairs]
        
        # Jina 공식 API 호출
        results = self.model.rerank(query, documents)
        
        # relevance_score 추출 (float32 -> float 변환)
        return [float(result['relevance_score']) for result in results]


# Qdrant 클라이언트 싱글톤
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Qdrant 클라이언트 싱글톤"""
    global _qdrant_client
    
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            host=RAGConfig.QDRANT_HOST,
            port=RAGConfig.QDRANT_PORT,
            prefer_grpc=False,
            timeout=60.0,
        )
    return _qdrant_client


def get_reranker():
    """Reranker 모델 싱글톤 (CrossEncoder 또는 Jina Reranker)"""
    global _reranker
    
    if _reranker is None:
        logger.info("[RERANKER] 싱글톤이 None, 새로 로딩합니다")
        reranker_type = RAGConfig.RERANKER_TYPE.lower()
        
        if reranker_type == "jina":
            if not _HAS_JINA:
                logger.warning("Jina Reranker를 사용할 수 없습니다. CrossEncoder로 폴백합니다.")
                _reranker = CrossEncoder(RAGConfig.RERANKER_ID, max_length=512)
                logger.info(f"Reranker loaded (fallback): {RAGConfig.RERANKER_ID}")
            else:
                _reranker = JinaReranker(model_name=RAGConfig.JINA_RERANKER_MODEL)
        else:
            _reranker = CrossEncoder(RAGConfig.RERANKER_ID, max_length=512)
            logger.info(f"Reranker loaded: {RAGConfig.RERANKER_ID}")
    else:
        logger.info("[RERANKER] 싱글톤 재사용 (이미 로딩됨)")
    
    return _reranker


def rrf_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    w_dense: float = None,
    w_sparse: float = None,
    k: int = None
) -> List[Dict[str, Any]]:
    """
    개선된 RRF: 순위 기반 점수 + 원본 점수 활용

    - k 값을 60 → 5로 낮춰서 상위 순위 강조
    - 원본 점수를 30% 반영하여 품질 차별화
    """
    w_dense = w_dense or RAGConfig.W_DENSE
    w_sparse = w_sparse or RAGConfig.W_SPARSE
    k = k or RAGConfig.K_RRF
    use_score = RAGConfig.USE_SCORE_IN_RRF

    scores = defaultdict(float)
    result_map = {}

    for rank, r in enumerate(dense_results, 1):
        doc_id = r.get("doc_id") or f"dense_{rank}"

        # 순위 기반 점수
        rank_score = 1.0 / (k + rank)

        if use_score:
            # 원본 점수 (Dense는 이미 0~1 범위)
            original_score = r.get("score", 0.5)
            # 순위 70% + 원본 점수 30%
            combined = 0.7 * rank_score + 0.3 * original_score
        else:
            combined = rank_score

        scores[doc_id] += w_dense * combined
        result_map[doc_id] = r

    for rank, r in enumerate(sparse_results, 1):
        doc_id = r.get("doc_id") or f"sparse_{rank}"

        rank_score = 1.0 / (k + rank)

        if use_score:
            # Sparse 점수는 보통 0~20 범위이므로 정규화
            original_score = min(r.get("score", 0) / 20.0, 1.0)
            combined = 0.7 * rank_score + 0.3 * original_score
        else:
            combined = rank_score

        scores[doc_id] += w_sparse * combined
        if doc_id not in result_map:
            result_map[doc_id] = r

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids:
        result = result_map[doc_id].copy()
        result["rrf_score"] = scores[doc_id]
        fused_results.append(result)

    return fused_results


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_n: int = None
) -> List[Dict[str, Any]]:
    """Cross-encoder로 검색 결과 리랭킹 (필수)"""
    top_n = top_n or RAGConfig.TOP_K_FINAL
    
    if not results:
        return []
    
    reranker = get_reranker()
    
    pairs = [(query, r.get("text", "")) for r in results]
    scores = reranker.predict(pairs)
    
    for i, score in enumerate(scores):
        results[i]["rerank_score"] = float(score)
    
    reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
    filtered = [r for r in reranked if r.get("rerank_score", 0) >= RAGConfig.RERANK_THRESHOLD]
    
    logger.info(f"Reranked: {len(results)} -> {len(filtered)} (threshold: {RAGConfig.RERANK_THRESHOLD})")
    
    return filtered[:top_n] if filtered else reranked[:top_n]


async def dense_search(
    collection_name: str,
    query: str,
    top_k: int = None
) -> List[Dict[str, Any]]:
    """Dense 벡터 검색"""
    top_k = top_k or RAGConfig.K_PER_MODALITY
    client = get_qdrant_client()

    # 1. Dense 임베딩 생성
    t_embed_start = time.time()
    query_embedding = await embed_query(query)
    t_embed = time.time() - t_embed_start
    logger.info(f"    ↳ Dense embedding: {t_embed:.3f}s")

    # 2. 벡터 검색
    t_search_start = time.time()
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    t_search = time.time() - t_search_start
    logger.info(f"    ↳ Dense vector search: {t_search:.3f}s")

    return [
        {
            "text": hit.payload.get("page_content", hit.payload.get("text", "")),
            "score": hit.score,
            "doc_id": hit.payload.get("doc_id") or f"dense_{i}",
            "pdf_id": hit.payload.get("pdf_id"),
            "chunk_index": hit.payload.get("chunk_index"),
            "payload": hit.payload,
        }
        for i, hit in enumerate(results.points)
    ]


async def sparse_search(
    collection_name: str,
    query: str,
    top_k: int = None
) -> List[Dict[str, Any]]:
    """Sparse 벡터 검색 (BM25)"""
    if not has_sparse_support():
        return []

    top_k = top_k or RAGConfig.K_PER_MODALITY
    client = get_qdrant_client()
    sparse_model = get_sparse_embeddings()

    try:
        # 1. Sparse 임베딩 생성
        t_embed_start = time.time()
        sparse_vectors = sparse_model.embed_query(query)
        t_embed = time.time() - t_embed_start
        logger.info(f"    ↳ Sparse embedding: {t_embed:.3f}s")

        # indices와 values가 이미 리스트인 경우를 처리
        indices = sparse_vectors.indices if isinstance(sparse_vectors.indices, list) else sparse_vectors.indices.tolist()
        values = sparse_vectors.values if isinstance(sparse_vectors.values, list) else sparse_vectors.values.tolist()

        # 2. 벡터 검색
        t_search_start = time.time()
        results = client.query_points(
            collection_name=collection_name,
            query=SparseVector(
                indices=indices,
                values=values
            ),
            using="sparse",
            limit=top_k,
            with_payload=True,
        )
        t_search = time.time() - t_search_start
        logger.info(f"    ↳ Sparse vector search: {t_search:.3f}s")

        return [
            {
                "text": hit.payload.get("page_content", hit.payload.get("text", "")),
                "score": hit.score,
                "doc_id": hit.payload.get("doc_id") or f"sparse_{i}",
                "pdf_id": hit.payload.get("pdf_id"),
                "chunk_index": hit.payload.get("chunk_index"),
                "payload": hit.payload,
            }
            for i, hit in enumerate(results.points)
        ]

    except Exception as e:
        logger.warning(f"Sparse search failed: {e}")
        return []


async def hybrid_search(
    collection_name: str,
    query: str,
    top_k: int = None
) -> List[Dict[str, Any]]:
    """하이브리드 검색 (Dense + Sparse + RRF) - 병렬 실행"""
    # Dense와 Sparse 검색을 병렬로 실행
    dense_task = dense_search(collection_name, query, top_k)
    sparse_task = sparse_search(collection_name, query, top_k)
    
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
    
    logger.info(f"Dense search: {len(dense_results)} results")
    logger.info(f"Sparse search: {len(sparse_results)} results")
    
    if sparse_results:
        fused = rrf_fusion(dense_results, sparse_results)
        logger.info(f"RRF fusion: {len(fused)} results")
        return fused
    
    return dense_results
