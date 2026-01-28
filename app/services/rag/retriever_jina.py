#!/usr/bin/env python3
"""
Hybrid RAG Retriever with Jina Reranker (Official API)

검색 파이프라인:
1. Dense Search (BGE-M3): 원본 쿼리로 의미적 검색
2. Sparse Search (BM25): 원본 쿼리로 키워드 기반 검색
3. RRF Fusion: 1/(k+rank) 방식으로 두 결과 융합
4. Reranking: Jina Reranker 공식 API로 최종 정렬

Usage:
    python retriever_jina.py "What is the four-point probe method?" --collection semiconductor --top-k 20 --rerank-top 5
"""

import os
import json
import argparse
import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, SparseVector

# BGE-M3 Dense Embedding
try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    _HAS_BGE = True
except ImportError:
    _HAS_BGE = False
    print("[ERROR] transformers와 torch가 필요합니다: pip install transformers torch")
    sys.exit(1)

# BM25 Sparse
try:
    from langchain_qdrant import FastEmbedSparse
    _HAS_SPARSE = True
except Exception:
    _HAS_SPARSE = False
    print("[WARN] FastEmbedSparse를 사용할 수 없습니다. Dense only 모드로 진행합니다")

# Jina Reranker (공식 API)
_HAS_JINA_RERANKER = True
try:
    from transformers import AutoModel
except ImportError:
    _HAS_JINA_RERANKER = False
    print("[WARN] Jina reranker를 사용할 수 없습니다 (transformers 필요)")

# ===================== Config =====================
load_dotenv()

DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-m3")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
JINA_RERANKER_MODEL = os.getenv("JINA_RERANKER_MODEL", "jinaai/jina-reranker-v3")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:16333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

DEFAULT_TOP_K = 20
DEFAULT_RERANK_TOP = 5
DEFAULT_DENSE_WEIGHT = 0.7
DEFAULT_SPARSE_WEIGHT = 0.3
DEFAULT_RRF_K = 60


# ===================== BGE-M3 Embeddings =====================
class BGEM3Embeddings:
    """BGE-M3 Dense Embedding for queries"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        print(f"[BGE-M3] 모델 로딩: {model_name}")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[BGE-M3] 로딩 완료 (device: {self.device})")

    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 임베딩"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings[0].cpu().tolist()


# ===================== Jina Reranker (공식 API) =====================
class JinaReranker:
    """
    Jina Reranker v3 공식 API 사용

    Hugging Face 공식 문서:
    https://huggingface.co/jinaai/jina-reranker-v3

    사용법:
    - AutoModel.from_pretrained()로 모델 로드
    - model.rerank(query, documents) 메서드 사용
    - 내부적으로 sigmoid 적용하여 0~1 범위 relevance_score 반환
    """

    def __init__(self, model_name: str = "jinaai/jina-reranker-v3", device: str = "cuda"):
        if not _HAS_JINA_RERANKER:
            raise ImportError("transformers가 필요합니다: pip install transformers")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        print(f"[Jina Reranker] 모델 로딩: {model_name} (device: {device})")

        # Jina 공식 API: AutoModel 사용 (AutoModelForSequenceClassification 아님!)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"[Jina Reranker] 로딩 완료")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Jina 공식 API를 사용한 Reranking

        Args:
            query: 검색 쿼리
            documents: 문서 리스트 (각 문서는 'content' 키 포함)
            top_k: 상위 k개 반환

        Returns:
            relevance_score가 추가된 정렬된 문서 리스트
        """
        if not documents:
            return []

        # 문서 텍스트 추출
        doc_texts = [doc['content'] for doc in documents]

        # Jina 공식 API 호출
        # model.rerank()는 내부적으로 sigmoid를 적용하여 0~1 범위의 점수를 반환
        results = self.model.rerank(query, doc_texts)

        # 결과를 원본 문서에 매핑
        for doc, result in zip(documents, results):
            # Jina API는 다음 키를 반환:
            # - 'relevance_score': 0~1 범위의 관련성 점수 (높을수록 관련성 높음)
            # - 'index': 원본 문서 인덱스
            # - 'document': 문서 텍스트
            # float32 -> float 변환 (JSON 직렬화를 위해)
            doc['rerank_score'] = float(result['relevance_score'])

        # 점수 기준 내림차순 정렬
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k]


# ===================== RRF (Reciprocal Rank Fusion) =====================
def reciprocal_rank_fusion(
    dense_results: List[ScoredPoint],
    sparse_results: List[ScoredPoint],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    RRF를 사용한 하이브리드 검색 결과 융합

    RRF Formula:
        score = dense_weight * (1 / (k + dense_rank)) + sparse_weight * (1 / (k + sparse_rank))

    Args:
        dense_results: Dense 검색 결과
        sparse_results: Sparse 검색 결과
        dense_weight: Dense 가중치
        sparse_weight: Sparse 가중치
        k: RRF 상수 (기본값: 60, 논문 권장값)

    Returns:
        RRF 점수로 정렬된 문서 리스트
    """
    # 정규화: 가중치 합이 1이 되도록
    total_weight = dense_weight + sparse_weight
    if total_weight > 0:
        dense_weight = dense_weight / total_weight
        sparse_weight = sparse_weight / total_weight
    else:
        dense_weight = sparse_weight = 0.5

    # ID -> (rank, score, payload) 매핑
    dense_map = {str(p.id): (rank + 1, p.score, p.payload) for rank, p in enumerate(dense_results)}
    sparse_map = {str(p.id): (rank + 1, p.score, p.payload) for rank, p in enumerate(sparse_results)}

    # 모든 고유 ID 수집
    all_ids = set(dense_map.keys()) | set(sparse_map.keys())

    # RRF 점수 계산
    rrf_scores = {}
    for doc_id in all_ids:
        score = 0.0
        payload = None
        dense_rank = None
        sparse_rank = None
        dense_score = None
        sparse_score = None

        # Dense 기여도
        if doc_id in dense_map:
            dense_rank, dense_score, payload = dense_map[doc_id]
            score += dense_weight * (1.0 / (k + dense_rank))

        # Sparse 기여도
        if doc_id in sparse_map:
            sparse_rank, sparse_score, sparse_payload = sparse_map[doc_id]
            score += sparse_weight * (1.0 / (k + sparse_rank))
            if payload is None:
                payload = sparse_payload

        rrf_scores[doc_id] = {
            'id': doc_id,
            'rrf_score': score,
            'dense_rank': dense_rank,
            'sparse_rank': sparse_rank,
            'dense_score': dense_score,
            'sparse_score': sparse_score,
            'payload': payload or {}
        }

    # RRF 점수 기준 내림차순 정렬
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)

    return sorted_results


# ===================== Hybrid Retriever with Jina =====================
class HybridRetrieverJina:
    """
    Hybrid Retriever with Jina Reranker (공식 API)

    검색 파이프라인:
    1. Dense Search: 의미적 유사도 검색 (BGE-M3)
    2. Sparse Search: 키워드 매칭 검색 (BM25)
    3. RRF Fusion: 두 결과를 1/(k+rank) 방식으로 융합
    4. Reranking: Jina Reranker 공식 API로 최종 정렬
    """

    def __init__(
        self,
        collection_name: str,
        dense_model: str = DENSE_MODEL,
        sparse_model: str = SPARSE_MODEL,
        jina_reranker_model: str = JINA_RERANKER_MODEL,
        qdrant_url: str = QDRANT_URL,
        qdrant_api_key: Optional[str] = QDRANT_API_KEY,
        device: str = "cuda"
    ):
        """
        Args:
            collection_name: Qdrant collection 이름
            dense_model: Dense embedding 모델
            sparse_model: Sparse embedding 모델
            jina_reranker_model: Jina Reranker 모델
            qdrant_url: Qdrant URL
            qdrant_api_key: Qdrant API key
            device: 연산 디바이스 (cuda/cpu)
        """
        self.collection_name = collection_name
        self.device = device if torch.cuda.is_available() else "cpu"

        # Qdrant Client
        print(f"[Qdrant] 연결 중: {qdrant_url}")
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=True,
            grpc_port=16334,
            timeout=60.0
        )
        print(f"[Qdrant] 연결 완료")

        # Dense Embeddings
        self.dense_embeddings = BGEM3Embeddings(model_name=dense_model, device=self.device)

        # Sparse Embeddings
        if _HAS_SPARSE:
            print(f"[Sparse] 모델 초기화: {sparse_model}")
            self.sparse_embeddings = FastEmbedSparse(model_name=sparse_model)
            print(f"[Sparse] 초기화 완료")
        else:
            self.sparse_embeddings = None

        # Jina Reranker (공식 API)
        if _HAS_JINA_RERANKER:
            self.reranker = JinaReranker(model_name=jina_reranker_model, device=self.device)
        else:
            self.reranker = None

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        rerank_top: int = DEFAULT_RERANK_TOP,
        rrf_k: int = DEFAULT_RRF_K
    ) -> Dict[str, Any]:
        """
        Hybrid 검색 실행

        Args:
            query: 사용자 쿼리
            top_k: Dense/Sparse 각각에서 가져올 결과 개수
            dense_weight: Dense 가중치
            sparse_weight: Sparse 가중치
            rerank_top: 최종 반환 결과 개수
            rrf_k: RRF 상수 (논문 권장값: 60)

        Returns:
            검색 결과 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"[Hybrid Retriever - Jina] Dense + Sparse → RRF → Jina Reranker")
        print(f"Query: {query}")
        print(f"Params: top_k={top_k}, dense_w={dense_weight}, sparse_w={sparse_weight}, rrf_k={rrf_k}")
        print(f"{'='*60}\n")

        # Step 1: Dense 검색
        print("[Step 1/4] Dense 검색 (의미적 유사도)")
        dense_vector = self.dense_embeddings.embed_query(query)

        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k,
            with_payload=True
        ).points

        print(f"  → Dense 결과: {len(dense_results)}개")

        # Step 2: Sparse 검색
        sparse_query = query
        print(f"\n[Step 2/4] Sparse 검색 (키워드 매칭)")

        sparse_results = []
        if self.sparse_embeddings:
            sparse_vector = list(self.sparse_embeddings.embed_documents([sparse_query]))[0]

            # sparse_vector 형식 변환
            if hasattr(sparse_vector, 'indices') and hasattr(sparse_vector, 'values'):
                indices = sparse_vector.indices
                values = sparse_vector.values
            elif isinstance(sparse_vector, dict):
                indices = sparse_vector.get('indices', [])
                values = sparse_vector.get('values', [])
            elif isinstance(sparse_vector, tuple) and len(sparse_vector) == 2:
                indices, values = sparse_vector
            else:
                print(f"  [WARN] Sparse vector 형식 오류: {type(sparse_vector)}")
                indices = []
                values = []

            # Sparse 검색 실행
            if indices and values:
                sparse_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=top_k,
                    with_payload=True
                ).points

                print(f"  → Sparse 결과: {len(sparse_results)}개")
            else:
                print(f"  → Sparse 검색 스킵 (벡터 생성 실패)")
        else:
            print(f"  → Sparse 검색 스킵 (모델 없음)")

        # Step 3: RRF 융합
        print(f"\n[Step 3/4] RRF 융합 (k={rrf_k})")
        print(f"  공식: score = {dense_weight:.1f}/(k+dense_rank) + {sparse_weight:.1f}/(k+sparse_rank)")

        hybrid_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            k=rrf_k
        )

        print(f"  → RRF 융합 결과: {len(hybrid_results)}개")

        # Step 4: Jina Reranking
        print(f"\n[Step 4/4] Jina Reranking (공식 API)")

        # Reranker 입력 형식으로 변환
        rerank_input = [
            {
                'id': doc['id'],
                'rrf_score': doc['rrf_score'],
                'dense_rank': doc['dense_rank'],
                'sparse_rank': doc['sparse_rank'],
                'dense_score': doc['dense_score'],
                'sparse_score': doc['sparse_score'],
                'content': doc['payload'].get('page_content', ''),
                'payload': doc['payload']
            }
            for doc in hybrid_results
        ]

        if self.reranker and rerank_input:
            # Reranker에 충분한 후보 제공 (최소 rerank_top * 2)
            rerank_candidates = min(len(rerank_input), max(rerank_top * 2, 10))
            final_results = self.reranker.rerank(
                query=query,
                documents=rerank_input[:rerank_candidates],
                top_k=rerank_top
            )
            print(f"  → Jina Reranking 완료: {len(final_results)}개 (후보: {rerank_candidates}개)")
            print(f"  → Relevance scores (0~1 범위):")
            for i, doc in enumerate(final_results[:3], 1):
                print(f"     [{i}] {doc['rerank_score']:.4f}")
        else:
            final_results = rerank_input[:rerank_top]
            if self.reranker is None:
                print(f"  → Reranking 스킵 (모델 없음), 상위 {rerank_top}개 반환")
            else:
                print(f"  → Reranking 스킵 (후보 없음)")

        # 결과 정리
        result = {
            'query': query,
            'sparse_query': sparse_query,
            'search_params': {
                'collection': self.collection_name,
                'top_k': top_k,
                'dense_weight': dense_weight,
                'sparse_weight': sparse_weight,
                'rrf_k': rrf_k,
                'rerank_top': rerank_top,
                'method': 'Dense+Sparse→RRF→JinaReranker(Official API)'
            },
            'dense_results': [
                {
                    'id': str(p.id),
                    'rank': i + 1,
                    'score': p.score,
                    'page_content': p.payload.get('page_content', '')
                }
                for i, p in enumerate(dense_results)
            ],
            'sparse_results': [
                {
                    'id': str(p.id),
                    'rank': i + 1,
                    'score': p.score,
                    'page_content': p.payload.get('page_content', '')
                }
                for i, p in enumerate(sparse_results)
            ],
            'hybrid_results': hybrid_results,
            'final_results': final_results,
            'num_dense': len(dense_results),
            'num_sparse': len(sparse_results),
            'num_hybrid': len(hybrid_results),
            'num_final': len(final_results)
        }

        print(f"\n{'='*60}")
        print(f"[검색 완료]")
        print(f"  Dense: {result['num_dense']}개, Sparse: {result['num_sparse']}개")
        print(f"  RRF 융합: {result['num_hybrid']}개")
        print(f"  최종 결과: {result['num_final']}개")
        print(f"{'='*60}\n")

        return result


# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG Retriever with Jina Reranker (Official API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 사용
  python retriever_jina.py "What is the four-point probe method?" --collection semiconductor

  # 파라미터 조정
  python retriever_jina.py "resistivity measurement" \\
      --collection semiconductor \\
      --top-k 30 \\
      --dense 0.6 \\
      --sparse 0.4 \\
      --rerank-top 10

  # 결과 저장
  python retriever_jina.py "semiconductor characterization" \\
      --collection semiconductor \\
      --output results.json
        """
    )

    # Required
    parser.add_argument(
        "query",
        type=str,
        help="검색 쿼리"
    )

    # Collection
    parser.add_argument(
        "--collection",
        type=str,
        default="semiconductor",
        help="Qdrant collection 이름 (기본값: semiconductor)"
    )

    # Search params
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Dense/Sparse 각각 검색 결과 개수 (기본값: {DEFAULT_TOP_K})"
    )

    parser.add_argument(
        "--dense",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
        help=f"Dense 가중치 (기본값: {DEFAULT_DENSE_WEIGHT})"
    )

    parser.add_argument(
        "--sparse",
        type=float,
        default=DEFAULT_SPARSE_WEIGHT,
        help=f"Sparse 가중치 (기본값: {DEFAULT_SPARSE_WEIGHT})"
    )

    parser.add_argument(
        "--rerank-top",
        type=int,
        default=DEFAULT_RERANK_TOP,
        help=f"최종 반환 결과 개수 (기본값: {DEFAULT_RERANK_TOP})"
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=f"RRF 상수 (기본값: {DEFAULT_RRF_K}, 논문 권장값)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="결과 저장 JSON 파일 경로"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="연산 디바이스 (기본값: cuda)"
    )

    args = parser.parse_args()

    # Retriever 초기화
    try:
        retriever = HybridRetrieverJina(
            collection_name=args.collection,
            device=args.device
        )
    except Exception as e:
        print(f"\n[ERROR] Retriever 초기화 실패: {e}")
        sys.exit(1)

    # 검색 실행
    try:
        result = retriever.retrieve(
            query=args.query,
            top_k=args.top_k,
            dense_weight=args.dense,
            sparse_weight=args.sparse,
            rerank_top=args.rerank_top,
            rrf_k=args.rrf_k
        )
    except Exception as e:
        print(f"\n[ERROR] 검색 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 결과 출력
    print("\n" + "="*60)
    print("최종 결과 (Top 5)")
    print("="*60)
    for i, doc in enumerate(result['final_results'][:5], 1):
        print(f"\n[{i}] ID: {doc['id']}")
        print(f"    Rerank Score: {doc.get('rerank_score', 0):.4f}")
        print(f"    RRF Score: {doc.get('rrf_score', 0):.4f}")
        print(f"    Dense Rank: {doc.get('dense_rank', 'N/A')}, Sparse Rank: {doc.get('sparse_rank', 'N/A')}")
        print(f"    내용: {doc['content'][:150]}...")

    # 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([result], f, ensure_ascii=False, indent=2)

        print(f"\n✅ 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
