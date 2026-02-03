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
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    import torch
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    logger.warning("transformers and torch not available. Advanced rerankers will not be available.")

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
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers와 torch가 필요합니다: pip install transformers torch")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"[Jina Reranker] 모델 로딩: {model_name} (device: {self.device})")
        
        # Jina 공식 API: AutoModel 사용
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype="auto",
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


class QwenReranker:
    """
    Qwen3-Reranker-8B 공식 API

    Hugging Face: https://huggingface.co/Qwen/Qwen3-Reranker-8B
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-8B", device: str = "cuda"):
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers와 torch가 필요합니다: pip install transformers torch")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.max_length = 8192

        logger.info(f"[Qwen Reranker] 모델 로딩: {model_name} (device: {self.device})")

        # Tokenizer 로딩 (padding_side='left')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

        # Qwen 공식 API: AutoModelForCausalLM 사용
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).cuda().eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()

        # Token IDs
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        # Prompt templates
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        logger.info(f"[Qwen Reranker] 로딩 완료")

    def format_instruction(self, instruction: str, query: str, doc: str) -> str:
        """Format query-document pair with instruction"""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )

    def process_inputs(self, pairs: List[str]) -> Dict[str, Any]:
        """Process formatted pairs into model inputs"""
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens

        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)

        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)

        return inputs

    @torch.no_grad()
    def compute_scores(self, inputs: Dict[str, Any]) -> List[float]:
        """Compute reranking scores from model logits"""
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def predict(self, pairs: List[tuple], instruction: str = None) -> List[float]:
        """
        CrossEncoder 호환 인터페이스

        Args:
            pairs: [(query, document), ...] 형식의 리스트
            instruction: 태스크별 instruction (선택사항)

        Returns:
            각 pair에 대한 relevance score (0~1 범위)
        """
        if not pairs:
            return []

        # Format all query-document pairs
        formatted_pairs = [
            self.format_instruction(instruction, query, doc)
            for query, doc in pairs
        ]

        # Process and compute scores
        inputs = self.process_inputs(formatted_pairs)
        scores = self.compute_scores(inputs)

        return scores


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
    """Reranker 모델 싱글톤 (CrossEncoder, Jina Reranker, 또는 Qwen Reranker)"""
    global _reranker

    if _reranker is None:
        logger.info("[RERANKER] 싱글톤이 None, 새로 로딩합니다")
        reranker_type = RAGConfig.RERANKER_TYPE.lower()

        if reranker_type == "jina":
            if not _HAS_TRANSFORMERS:
                logger.warning("Jina Reranker를 사용할 수 없습니다. CrossEncoder로 폴백합니다.")
                _reranker = CrossEncoder(RAGConfig.RERANKER_ID, max_length=512)
                logger.info(f"Reranker loaded (fallback): {RAGConfig.RERANKER_ID}")
            else:
                _reranker = JinaReranker(model_name=RAGConfig.JINA_RERANKER_MODEL)
        elif reranker_type == "qwen":
            if not _HAS_TRANSFORMERS:
                logger.warning("Qwen Reranker를 사용할 수 없습니다. CrossEncoder로 폴백합니다.")
                _reranker = CrossEncoder(RAGConfig.RERANKER_ID, max_length=512)
                logger.info(f"Reranker loaded (fallback): {RAGConfig.RERANKER_ID}")
            else:
                _reranker = QwenReranker(model_name=RAGConfig.QWEN_RERANKER_MODEL)
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

    # 디버깅: doc_id 중복 체크
    logger.info(f"[RRF FUSION] Dense results: {len(dense_results)}, Sparse results: {len(sparse_results)}")
    logger.info(f"[RRF FUSION] Unique doc_ids after fusion: {len(sorted_ids)}")

    # 처음 5개의 doc_id 로깅
    for i, doc_id in enumerate(sorted_ids[:5], 1):
        result = result_map[doc_id]
        text_preview = result.get("text", "")[:80] + "..."
        logger.info(f"  [{i}] doc_id={doc_id} | score={scores[doc_id]:.4f} | text={text_preview}")

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

    # 디버깅: 리랭킹 전 결과 확인
    logger.info(f"[BEFORE RERANK] Total results: {len(results)}")
    logger.info(f"[BEFORE RERANK] Unique doc_ids: {len(set(r.get('doc_id', 'N/A') for r in results))}")

    # 처음 5개의 doc_id 로깅
    for i, r in enumerate(results[:5], 1):
        text_preview = r.get("text", "")[:80] + "..."
        logger.info(f"  [{i}] doc_id={r.get('doc_id', 'N/A')} | rrf_score={r.get('rrf_score', 0):.4f} | text={text_preview}")

    reranker = get_reranker()
    
    pairs = [(query, r.get("text", "")) for r in results]
    scores = reranker.predict(pairs)
    
    for i, score in enumerate(scores):
        results[i]["rerank_score"] = float(score)
    
    # 점수 통계 로깅
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        logger.info(f"[RERANK SCORES] Min: {min_score:.4f}, Max: {max_score:.4f}, Avg: {avg_score:.4f}")
        logger.info(f"[RERANK SCORES] All scores: {[f'{s:.4f}' for s in sorted(scores, reverse=True)[:10]]}")
    
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

    dense_results = [
        {
            "text": (
                hit.payload.get("metadata", {}).get("content_for_llm") or
                hit.payload.get("content_for_llm") or
                hit.payload.get("page_content", hit.payload.get("text", ""))
            ),
            "score": hit.score,
            "doc_id": hit.payload.get("doc_id") or hit.payload.get("metadata", {}).get("doc_id") or f"dense_{i}",
            "pdf_id": hit.payload.get("pdf_id") or hit.payload.get("metadata", {}).get("pdf_id"),
            "chunk_index": hit.payload.get("chunk_index") or hit.payload.get("metadata", {}).get("chunk_index"),
            "payload": hit.payload,
        }
        for i, hit in enumerate(results.points)
    ]

    # 디버깅: 처음 3개의 doc_id 로깅
    logger.info(f"[DENSE] Top 3 doc_ids: {[r['doc_id'] for r in dense_results[:3]]}")

    return dense_results


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

        sparse_results = [
            {
                "text": (
                    hit.payload.get("metadata", {}).get("content_for_llm") or
                    hit.payload.get("content_for_llm") or
                    hit.payload.get("page_content", hit.payload.get("text", ""))
                ),
                "score": hit.score,
                "doc_id": hit.payload.get("doc_id") or hit.payload.get("metadata", {}).get("doc_id") or f"sparse_{i}",
                "pdf_id": hit.payload.get("pdf_id") or hit.payload.get("metadata", {}).get("pdf_id"),
                "chunk_index": hit.payload.get("chunk_index") or hit.payload.get("metadata", {}).get("chunk_index"),
                "payload": hit.payload,
            }
            for i, hit in enumerate(results.points)
        ]

        # 디버깅: 처음 3개의 doc_id 로깅
        logger.info(f"[SPARSE] Top 3 doc_ids: {[r['doc_id'] for r in sparse_results[:3]]}")

        return sparse_results

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
