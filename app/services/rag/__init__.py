# -*- coding: utf-8 -*-
"""
RAG Service Package
===================

고급 RAG 파이프라인 모듈

Features:
- 다중 쿼리 생성 (Query Expansion)
- 하이브리드 검색 (Dense + Sparse + RRF)
- Cross-Encoder Reranking
- 이웃 청크 확장 (Context expansion)
- 병렬 처리
"""
from .config import RAGConfig
from .pipeline import RAGPipeline

# 싱글톤 인스턴스
_pipeline_instance = None


def get_rag_service() -> RAGPipeline:
    """
    RAGPipeline 싱글톤 인스턴스 반환
    
    Returns:
        RAGPipeline 인스턴스
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance


__all__ = [
    "RAGConfig",
    "RAGPipeline",
    "get_rag_service",
]
