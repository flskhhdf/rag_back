# -*- coding: utf-8 -*-
"""
LLM Client Module - vLLM OpenAI-compatible API
"""
import logging
import re
from typing import List, Dict, AsyncGenerator, Tuple, Any   
import httpx

from .config import RAGConfig, SYS_PROMPT, SYS_PROMPT_NO_CTX, CTX_TEMPLATE

logger = logging.getLogger(__name__)


def clean_stream_chunk(text: str) -> str:
    """스트리밍 청크 정규화"""
    # 1. 태그 복구 (SOURCE 태그가 중요)
    # < S OURCE, <SOURCE, < S OUR CE 등 다양하게 깨질 수 있음
    text = re.sub(r'<\s*S\s*O\s*U\s*R\s*C\s*E', '<SOURCE', text)
    text = re.sub(r'S\s*O\s*U\s*R\s*C\s*E\s*>', 'SOURCE>', text)

    # 2. 영문 대문자 사이 공백 제거 (예: B M S -> BMS, U I -> UI)
    text = re.sub(r'([A-Z])\s+(?=[A-Z])', r'\1', text)

    # 3. 영문 단어 사이 하이픈 정규화 (Wake - up -> Wake-up)
    # 영문자 앞뒤로만 적용하여 한글/마크다운 테이블은 보호
    text = re.sub(r'([a-zA-Z])\s+([‑-])\s+(?=[a-zA-Z])', r'\1\2', text)

    # 4. 숫자 사이 공백 제거 (1 2 3 -> 123)
    text = re.sub(r'(\d)\s+(?=\d)', r'\1', text)

    # 5. 한글/영문 문자 사이에 불필요하게 삽입된 하이픈과 공백 제거
    # 예: "용 - 어" -> "용어", "설 - 명" -> "설명"
    # 단, 마크다운 테이블 구분자(|)와 함께 있는 경우는 보호
    text = re.sub(r'([가-힣a-zA-Z0-9])\s*-\s*(?=[가-힣a-zA-Z0-9])', r'\1', text)

    return text


def build_rag_messages(
    query: str,
    contexts: List[str] = None,
    history: List[Dict[str, str]] = None,
    results: List[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """RAG Messages 배열 생성 - OpenAI Chat API 표준 멀티턴 대화 지원
    
    Args:
        query: 사용자 질문
        contexts: 텍스트 컨텍스트 리스트 (하위 호환성, deprecated)
        history: 대화 이력 [{"role": "user/assistant", "content": "..."}]
        results: 검색 결과 (payload 포함)
    
    Returns:
        messages: OpenAI Chat API 형식의 메시지 배열
    """
    # results가 제공되면 이를 사용하여 메타데이터 포함 컨텍스트 생성
    if results:
        logger.info(f"[DEBUG] Building context from {len(results)} results")
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get("expanded_text") or result.get("text", "")
            payload = result.get("payload", {})
            
            # 디버깅: payload 확인
            logger.info(f"[Context {i}] Payload keys: {list(payload.keys())}")
            
            # 메타데이터 추출 (payload.metadata 안에 있음)
            metadata = payload.get("metadata", {})
            filename = metadata.get("filename", "알 수 없음")
            
            # page_no 또는 pages (set) 처리
            page_no = metadata.get("page_no")
            if page_no is None:
                pages_set = metadata.get("pages")
                if pages_set and isinstance(pages_set, set):
                    page_no = ", ".join(map(str, sorted(pages_set)))
                elif pages_set and isinstance(pages_set, list):
                    page_no = ", ".join(map(str, sorted(pages_set)))
                else:
                    page_no = "N/A"
            
            logger.info(f"[Context {i}] filename={filename}, page_no={page_no}")
            
            # 헤더 추가
            header = f"[파일명: {filename} | 페이지: {page_no}]"
            context_parts.append(f"[문서 {i}]\n{header}\n{text}")
        
        context_text = "\n\n".join(context_parts)
        logger.info(f"[DEBUG] Context preview (first 200 chars): {context_text[:200]}")
    elif contexts:
        # 하위 호환성: contexts만 제공된 경우 (메타데이터 없음)
        context_text = "\n\n".join([
            f"[문서 {i+1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])
    else:
        context_text = ""
    
    # Messages 배열 구성
    messages = []
    
    # 1. System message
    if context_text:
        messages.append({"role": "system", "content": SYS_PROMPT})
    else:
        messages.append({"role": "system", "content": SYS_PROMPT_NO_CTX})
    
    # 2. 대화 이력 추가 (최근 6턴 = 12개 메시지)
    if history:
        recent_history = history[-12:]
        for msg in recent_history:
            messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })
    
    # 3. 현재 사용자 질문 (컨텍스트 포함)
    if context_text:
        current_user_message = CTX_TEMPLATE.format(
            Lang=RAGConfig.SUMMARIZE_LANG,
            DOCUMENT_CONTEXT=context_text,
            RECENT_DIALOG="",  # 대화 이력은 messages에 이미 포함되어 있음
            QUERY=query
        )
    else:
        current_user_message = f"""<USER_QUERY>
{query}
</USER_QUERY>
"""
    
    messages.append({"role": "user", "content": current_user_message})
    
    logger.info(f"[DEBUG] Built {len(messages)} messages for LLM")
    return messages



async def stream_llm_response(
    messages: List[Dict[str, str]],
    model: str = None,
    vllm_url: str = None,
    max_tokens: int = None
) -> AsyncGenerator[str, None]:
    """vLLM Chat Completions API 응답 (멀티턴 대화 지원)
    
    Args:
        messages: OpenAI Chat API 형식의 메시지 배열
        model: LLM 모델명 (기본값: RAGConfig.LLM_MODEL)
        vllm_url: vLLM 서버 URL (기본값: RAGConfig.VLLM_URL)
        max_tokens: 최대 토큰 수 (기본값: RAGConfig.MAX_TOKENS)
    """
    import json
    import time

    model = model or RAGConfig.LLM_MODEL
    vllm_url = vllm_url or RAGConfig.VLLM_URL
    max_tokens = max_tokens or RAGConfig.MAX_TOKENS

    try:
        start_time = time.time()

        async with httpx.AsyncClient(timeout=240.0) as client:
            # vLLM OpenAI-compatible Chat API payload
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 1.0,
                "stream": False,  # 스트리밍 비활성화
            }

            # 요청
            response = await client.post(
                f"{vllm_url}/v1/chat/completions",
                json=payload,
                timeout=240.0
            )
            response.raise_for_status()

            # 응답 파싱
            data = response.json()
            choices = data.get("choices", [])

            if not choices:
                yield "응답을 생성할 수 없습니다."
                return

            # content 추출
            message = choices[0].get("message", {})
            content = message.get("content", "")

            if not content:
                yield "응답이 비어있습니다."
                return

            # 텍스트 정제
            cleaned = clean_stream_chunk(content)

            # 로깅
            total_time = time.time() - start_time
            logger.info(f"[FULL LLM RESPONSE]\n{cleaned}")
            logger.info(f"⏱️  Total Generation Time: {total_time:.3f}s")

            # 전체 응답을 한 번에 yield
            yield cleaned

    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        yield f"오류가 발생했습니다: {str(e)}"
