# -*- coding: utf-8 -*-
"""
Embeddings Module - Dense/Sparse 임베딩
"""
import logging
from typing import Optional, Tuple, List

from langchain_ollama import OllamaEmbeddings

from .config import RAGConfig

logger = logging.getLogger(__name__)

# 싱글톤 인스턴스
_dense_embeddings: Optional[OllamaEmbeddings] = None
_sparse_embeddings = None
_HAS_SPARSE = False

# Sparse 임베딩 지원 확인
try:
    from langchain_qdrant import FastEmbedSparse
    _HAS_SPARSE = True
except ImportError:
    logger.warning("FastEmbedSparse not available, sparse search disabled")


def get_dense_embeddings() -> OllamaEmbeddings:
    """Dense 임베딩 모델 싱글톤"""
    global _dense_embeddings
    
    if _dense_embeddings is None:
        _dense_embeddings = OllamaEmbeddings(
            model=RAGConfig.EMBED_MODEL,
            base_url=RAGConfig.OLLAMA_URL,
        )
        logger.info(f"Dense embeddings initialized: {RAGConfig.EMBED_MODEL}")
    
    return _dense_embeddings


def get_sparse_embeddings():
    """Sparse 임베딩 모델 싱글톤 (BM25)"""
    global _sparse_embeddings
    
    if _HAS_SPARSE and _sparse_embeddings is None:
        try:
            _sparse_embeddings = FastEmbedSparse(model_name=RAGConfig.SPARSE_MODEL)
            logger.info(f"Sparse embeddings initialized: {RAGConfig.SPARSE_MODEL}")
        except Exception as e:
            logger.warning(f"Sparse embedding init failed: {e}")
    
    return _sparse_embeddings


def has_sparse_support() -> bool:
    """Sparse 임베딩 지원 여부"""
    return _HAS_SPARSE and get_sparse_embeddings() is not None


async def embed_query(text: str) -> List[float]:
    """쿼리 텍스트의 Dense 임베딩 생성"""
    embeddings = get_dense_embeddings()
    return embeddings.embed_query(text)


async def embed_documents(texts: List[str]) -> List[List[float]]:
    """여러 텍스트의 Dense 임베딩 생성"""
    embeddings = get_dense_embeddings()
    return embeddings.embed_documents(texts)
