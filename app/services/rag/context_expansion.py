# -*- coding: utf-8 -*-
"""
Context Expansion Module - 이웃 청크 확장
"""
import logging
from typing import List, Dict, Any

from .retriever import get_qdrant_client
from .config import RAGConfig

logger = logging.getLogger(__name__)


def fetch_neighbor_chunks(
    collection_name: str,
    current_chunk: Dict[str, Any],
    expand: int = None
) -> List[Dict[str, Any]]:
    """현재 청크의 이웃 청크들 조회 (범위 쿼리로 최적화)"""
    expand = expand or RAGConfig.NEIGHBOR_EXPAND

    try:
        pdf_id = current_chunk.get("pdf_id")
        chunk_index = current_chunk.get("chunk_index")

        if pdf_id is None or chunk_index is None:
            return []

        client = get_qdrant_client()

        # 범위 계산 (음수 방지)
        min_index = max(0, chunk_index - expand)
        max_index = chunk_index + expand

        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

        # 한 번의 쿼리로 범위 검색 (LangChain metadata 구조 고려)
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="metadata.pdf_id", match=MatchValue(value=pdf_id)),
                    FieldCondition(
                        key="metadata.chunk_index",
                        range=Range(gte=min_index, lte=max_index)
                    )
                ]
            ),
            limit=expand * 2 + 1,  # 최대 이웃 개수 (현재 청크 포함)
            with_payload=True,
            with_vectors=False
        )

        neighbors = []
        for point in results[0]:
            # LangChain metadata 구조에서 chunk_index 추출
            point_chunk_index = (
                point.payload.get("metadata", {}).get("chunk_index") or
                point.payload.get("chunk_index")
            )

            # chunk_index가 없거나 현재 청크는 제외
            if point_chunk_index is None or point_chunk_index == chunk_index:
                continue

            offset = point_chunk_index - chunk_index
            neighbors.append({
                "text": (
                    point.payload.get("metadata", {}).get("content_for_llm") or
                    point.payload.get("content_for_llm") or
                    point.payload.get("page_content", point.payload.get("text", ""))
                ),
                "chunk_index": point_chunk_index,
                "offset": offset
            })

        return sorted(neighbors, key=lambda x: x.get("chunk_index", 0))

    except Exception as e:
        logger.error(f"Neighbor fetch failed: {e}")
        return []


def expand_context_with_neighbors(
    collection_name: str,
    results: List[Dict[str, Any]],
    expand: int = None
) -> List[Dict[str, Any]]:
    """검색 결과에 이웃 청크 컨텍스트 추가"""
    expand = expand or RAGConfig.NEIGHBOR_EXPAND
    
    if expand <= 0:
        return results
    
    expanded_results = []
    
    for r in results:
        neighbors = fetch_neighbor_chunks(collection_name, r, expand)
        
        prev_texts = [n["text"] for n in neighbors if n.get("offset", 0) < 0]
        next_texts = [n["text"] for n in neighbors if n.get("offset", 0) > 0]
        
        parts = []
        if prev_texts:
            parts.extend(prev_texts)
        parts.append(r.get("text", ""))
        if next_texts:
            parts.extend(next_texts)
        
        expanded_text = "\n\n".join(parts)
        
        expanded = r.copy()
        expanded["expanded_text"] = expanded_text
        expanded["neighbors_count"] = len(neighbors)
        expanded_results.append(expanded)
    
    return expanded_results
