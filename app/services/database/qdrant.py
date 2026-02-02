import os
import re
import hashlib
from typing import List, Optional, Tuple, Dict
from uuid import uuid5, NAMESPACE_URL
from datetime import datetime

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams


# Sparse 임베딩 지원 여부
try:
    from langchain_qdrant import FastEmbedSparse
    _HAS_SPARSE = True
except Exception:
    _HAS_SPARSE = False


# ===== Config =====
EMBED_TYPE = os.getenv("EMBED_TYPE", "huggingface")
DENSE_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
DISABLE_SPARSE = os.getenv("DISABLE_SPARSE", "0").lower() in ("1", "true", "yes")


# ===== 전역 객체 (싱글톤) =====
_client: Optional[QdrantClient] = None
_dense_embeddings = None  # OllamaEmbeddings or HuggingFaceEmbeddings
_sparse_embeddings: Optional["FastEmbedSparse"] = None
_vectorstore_cache: Dict[str, QdrantVectorStore] = {}


def get_client() -> QdrantClient:
    """Qdrant 클라이언트 싱글톤"""
    global _client
    if _client is None:
        _client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False,
            timeout=60.0,
        )
    return _client


# vLLM용 LangChain Embeddings wrapper
class VLLMEmbeddings(Embeddings):
    """vLLM OpenAI client를 LangChain Embeddings 인터페이스로 래핑"""
    
    def __init__(self, client, model_name: str):
        super().__init__()
        self.client = client
        self.model_name = model_name
    
    def embed_query(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]


def get_embeddings() -> Tuple[any, Optional["FastEmbedSparse"]]:
    """임베딩 모델 싱글톤 (vLLM, Ollama, 또는 HuggingFace)"""
    global _dense_embeddings, _sparse_embeddings

    if _dense_embeddings is None:
        embed_type = EMBED_TYPE.lower()

        if embed_type == "vllm":
            # vLLM OpenAI client를 LangChain 인터페이스로 래핑
            from openai import OpenAI
            vllm_embed_url = os.getenv("VLLM_EMBED_URL", "http://localhost:8004")
            client = OpenAI(
                api_key="EMPTY",
                base_url=vllm_embed_url + "/v1",
            )
            _dense_embeddings = VLLMEmbeddings(client, DENSE_MODEL)
            print(f"[INFO] Dense embeddings initialized (vLLM): {DENSE_MODEL} at {vllm_embed_url}")

        elif embed_type == "ollama":
            _dense_embeddings = OllamaEmbeddings(
                model=DENSE_MODEL,
                base_url=OLLAMA_URL,
            )
            print(f"[INFO] Dense embeddings initialized (Ollama): {DENSE_MODEL}")

        elif embed_type == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            _dense_embeddings = HuggingFaceEmbeddings(
                model_name=DENSE_MODEL,
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"[INFO] Dense embeddings initialized (HuggingFace): {DENSE_MODEL}")

        else:
            raise ValueError(f"Unknown EMBED_TYPE: {embed_type}. Use 'vllm', 'ollama', or 'huggingface'")

    if not DISABLE_SPARSE and _HAS_SPARSE and _sparse_embeddings is None:
        try:
            _sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)
        except Exception as e:
            print(f"[WARN] Sparse 임베딩 비활성화: {e}")

    return _dense_embeddings, _sparse_embeddings


def sanitize_collection_name(filename: str) -> str:
    """파일명을 Qdrant 컬렉션 이름으로 변환 (특수문자 처리)"""
    # 확장자 제거
    name = os.path.splitext(filename)[0]
    
    # 특수문자를 언더스코어로 변환 (영문, 숫자, 언더스코어, 하이픈만 허용)
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # 연속된 언더스코어 제거
    name = re.sub(r'_{2,}', '_', name)
    
    # 앞뒤 언더스코어 제거
    name = name.strip('_')
    
    # 빈 문자열이면 기본값
    if not name:
        name = "pdf_document"
    
    # 소문자로 변환
    return name.lower()


def ensure_collection(dense_dim: int, collection_name: str):
    """컬렉션이 없으면 생성 (Hybrid Search 지원)"""
    client = get_client()

    try:
        client.get_collection(collection_name)
        return
    except Exception:
        pass

    vectors_cfg = {"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)}
    sparse_cfg = None

    if not DISABLE_SPARSE and _HAS_SPARSE:
        sparse_cfg = {"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_cfg,
        sparse_vectors_config=sparse_cfg,
    )


def get_vectorstore(collection_name: str) -> QdrantVectorStore:
    """VectorStore 생성/캐시 (LangChain)"""
    global _vectorstore_cache

    if collection_name in _vectorstore_cache:
        return _vectorstore_cache[collection_name]

    client = get_client()
    dense, sparse = get_embeddings()

    # Dense 차원 확인
    try:
        dense_dim = len(dense.embed_query("probe for dimension"))
    except Exception:
        dense_dim = 1024

    ensure_collection(dense_dim, collection_name)

    retrieval = RetrievalMode.DENSE
    kwargs = {
        "client": client,
        "collection_name": collection_name,
        "embedding": dense,
        "retrieval_mode": retrieval,
        "vector_name": "dense",
    }

    if sparse is not None:
        retrieval = RetrievalMode.HYBRID
        kwargs.update({
            "retrieval_mode": retrieval,
            "sparse_embedding": sparse,
            "sparse_vector_name": "sparse"
        })

    vectorstore = QdrantVectorStore(**kwargs)
    _vectorstore_cache[collection_name] = vectorstore
    return vectorstore


# ===== ID 생성 =====
def _build_doc_id(pdf_id: str, chunk_index: int) -> str:
    """문서 ID 생성"""
    return f"{pdf_id}|chunk_{chunk_index}"


def _stable_point_id(doc_id: str, content_for_hash: str) -> str:
    """안정적인 Point ID 생성 (중복 방지)"""
    h = hashlib.sha1((content_for_hash or "").encode("utf-8")).hexdigest()[:16]
    return str(uuid5(NAMESPACE_URL, f"{doc_id}|h={h}"))


def existing_ids(collection_name: str, ids: List[str], chunk_size: int = 2048) -> List[str]:
    """이미 존재하는 ID 목록 조회"""
    client = get_client()
    out: List[str] = []

    for i in range(0, len(ids), chunk_size):
        batch = ids[i:i+chunk_size]
        try:
            res = client.retrieve(
                collection_name=collection_name,
                ids=batch,
                with_payload=False,
                with_vectors=False
            )
            out.extend([str(p.id) for p in res])
        except Exception:
            pass

    return out


# ===== Upsert =====
def upsert_chunks(
    pdf_id: str,
    filename: str,
    chunks_data: list[dict],  # chunk_docling_json의 전체 반환값
) -> int:
    """청크와 메타데이터를 Qdrant에 저장 (중복 방지)"""
    # pdf_id를 컬렉션 이름으로 사용 (한글 파일명 충돌 방지)
    collection_name = pdf_id
    vs = get_vectorstore(collection_name)

    # Document 객체 생성
    docs: List[Document] = []
    ids: List[str] = []

    for i, chunk_dict in enumerate(chunks_data):
        # ✅ content 또는 page_content 둘 다 지원 (complete_chunker 호환)
        page_content = chunk_dict.get("page_content") or chunk_dict.get("content", "")
        if not page_content.strip():
            continue

        # 메타데이터 가져오기 (두 가지 형식 지원)
        if "metadata" in chunk_dict:
            # 형식 1: {page_content, metadata} 구조
            metadata = dict(chunk_dict.get("metadata", {}))
        else:
            # 형식 2: chunk_dict 전체가 메타데이터 (complete_chunker 형식)
            metadata = {k: v for k, v in chunk_dict.items()
                       if k not in ["page_content", "content", "content_for_llm"]}

        # content_for_llm을 raw_text로 저장 (LLM 응답 생성용)
        content_for_llm = chunk_dict.get("content_for_llm")
        if content_for_llm:
            metadata["raw_text"] = content_for_llm
        else:
            # 기존 방식: display_content를 raw_text로 변환
            display_content = metadata.pop("display_content", None) or page_content
            metadata["raw_text"] = display_content

        # 추가 필드 (기존 필드 유지)
        metadata["pdf_id"] = pdf_id
        metadata["filename"] = filename
        metadata["chunk_index"] = i
        metadata["uploaded_at"] = datetime.now().isoformat()
        metadata["collection_name"] = collection_name

        # doc_id 생성 (우선순위: chunk_id > doc_id > 자동 생성)
        if "doc_id" not in metadata:
            # chunk_id가 있으면 사용 (complete_chunker), 없으면 자동 생성
            doc_id = metadata.get("chunk_id") or _build_doc_id(pdf_id, i)
            metadata["doc_id"] = doc_id
        else:
            doc_id = metadata["doc_id"]
        
        point_id = _stable_point_id(doc_id, page_content)

        doc = Document(
            page_content=page_content,
            metadata=metadata
        )

        docs.append(doc)
        ids.append(point_id)

    if not docs:
        return 0

    # 중복 체크
    existing = existing_ids(collection_name, ids)
    if existing:
        mask = set(existing)
        docs_to_add = []
        ids_to_add = []

        for doc, point_id in zip(docs, ids):
            if point_id not in mask:
                docs_to_add.append(doc)
                ids_to_add.append(point_id)
    else:
        docs_to_add, ids_to_add = docs, ids

    if docs_to_add:
        vs.add_documents(docs_to_add, ids=ids_to_add, batch_size=BATCH_SIZE)
        return len(docs_to_add)

    return 0


# ===== Search =====
def search_similar(
    query_embedding: list[float],
    pdf_id: str,
    filename: str,
    top_k: int = 5,
) -> list[dict]:
    """유사 문서 검색"""
    # pdf_id를 컬렉션 이름으로 사용
    collection_name = pdf_id
    client = get_client()

    # query_points 사용 (최신 qdrant-client API)
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        using="dense",
        limit=top_k,
        with_payload=True,
    )

    return [
        {"text": hit.payload.get("page_content", hit.payload.get("text", "")), "score": hit.score}
        for hit in results.points
    ]


# ===== PDF 관리 =====
def get_pdf_list() -> list[dict]:
    """저장된 PDF 목록 조회 (모든 컬렉션 조회)"""
    client = get_client()

    try:
        # 모든 컬렉션 목록 가져오기
        collections = client.get_collections().collections
        
        pdf_map = {}
        for collection in collections:
            collection_name = collection.name
            
            try:
                results = client.scroll(
                    collection_name=collection_name,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in results[0]:
                    payload = point.payload

                    # 새 구조: pdf_id가 최상위에 있음
                    pdf_id = payload.get("pdf_id")
                    filename = payload.get("filename")
                    uploaded_at = payload.get("uploaded_at")

                    # 구 구조: metadata 안에 source_id로 저장
                    if not pdf_id:
                        metadata = payload.get("metadata", {})
                        pdf_id = metadata.get("source_id")
                        filename = metadata.get("source_name")

                    if not pdf_id:
                        continue

                    if pdf_id not in pdf_map:
                        pdf_map[pdf_id] = {
                            "pdf_id": pdf_id,
                            "filename": filename or "Unknown",
                            "uploaded_at": uploaded_at,  # None이면 None, 있으면 datetime string
                            "chunk_count": 0,
                        }
                    pdf_map[pdf_id]["chunk_count"] += 1
            except Exception:
                continue

        return list(pdf_map.values())
    except Exception:
        return []


def delete_pdf(pdf_id: str, filename: str = None) -> bool:
    """
    PDF 삭제 (Qdrant 컬렉션 전체 삭제)

    Args:
        pdf_id: PDF ID (컬렉션 이름으로 사용)
        filename: PDF 파일명 (로깅용, optional)

    Returns:
        True: 삭제 성공
        False: 삭제 실패
    """
    # pdf_id를 컬렉션 이름으로 사용
    collection_name = pdf_id
    client = get_client()

    try:
        # 컬렉션 존재 여부 확인
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if collection_name not in collection_names:
            print(f"[WARNING] Collection {collection_name} does not exist in Qdrant")
            return True  # 이미 없으므로 삭제 성공으로 처리

        # 컬렉션 전체 삭제
        client.delete_collection(collection_name=collection_name)
        print(f"[INFO] Deleted Qdrant collection: {collection_name} (file: {filename})")

        # 캐시에서도 제거
        global _vectorstore_cache
        if collection_name in _vectorstore_cache:
            del _vectorstore_cache[collection_name]
            print(f"[INFO] Removed from cache: {collection_name}")

        return True
    except Exception as e:
        print(f"[ERROR] Failed to delete collection {collection_name}: {e}")
        import traceback
        traceback.print_exc()
        return False