# -*- coding: utf-8 -*-
"""
LLM Client Module - vLLM OpenAI-compatible API
"""
import logging
import re
from typing import List, Dict, AsyncGenerator, Tuple
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


def build_rag_prompt(
    query: str,
    contexts: List[str],
    history: List[Dict[str, str]] = None
) -> Tuple[str, str]:
    """RAG 프롬프트 생성 - config_file.py 스타일
    
    Returns:
        (system_prompt, user_prompt) 튜플
    """
    # 컨텍스트 포맷팅
    context_text = "\n\n".join([
        f"[문서 {i+1}]\n{ctx}"
        for i, ctx in enumerate(contexts)
    ])
    
    # 대화 이력 포맷팅
    history_text = ""
    if history:
        recent_history = history[-6:]
        history_parts = []
        for msg in recent_history:
            role = "사용자" if msg.get("role") == "user" else "어시스턴트"
            content = msg.get("content", "")
            history_parts.append(f"{role}: {content}")
        history_text = "\n".join(history_parts)
    
    # 컨텍스트 유무에 따른 시스템 프롬프트 선택
    if contexts:
        system_prompt = SYS_PROMPT
        user_prompt = CTX_TEMPLATE.format(
            Lang=RAGConfig.SUMMARIZE_LANG,
            DOCUMENT_CONTEXT=context_text,
            RECENT_DIALOG=history_text or "(없음)",
            QUERY=query
        )
    else:
        system_prompt = SYS_PROMPT_NO_CTX
        user_prompt = f"""<RECENT_DIALOG>
{history_text or "(없음)"}
</RECENT_DIALOG>

<USER_QUERY>
{query}
</USER_QUERY>
"""
    
    return system_prompt, user_prompt



async def stream_llm_response(
    prompt: str,
    system_prompt: str = None,
    model: str = None,
    vllm_url: str = None,
    max_tokens: int = None
) -> AsyncGenerator[str, None]:
    """vLLM Chat Completions API 비스트리밍 응답 (전체 응답 후 후처리)"""
    model = model or RAGConfig.LLM_MODEL
    vllm_url = vllm_url or RAGConfig.VLLM_URL
    max_tokens = max_tokens or RAGConfig.MAX_TOKENS

    try:
        async with httpx.AsyncClient(timeout=240.0) as client:
            # messages 배열 구성
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # vLLM OpenAI-compatible Chat API payload
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 1.0,
                "stream": False,  # 스트리밍 비활성화
            }

            # 전체 응답을 한 번에 받기 (Chat Completions endpoint)
            response = await client.post(
                f"{vllm_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()

            import json
            result = response.json()

            # OpenAI Chat API 형식: choices[0].message.content
            choices = result.get("choices", [])
            if not choices:
                logger.warning("Empty response from LLM")
                yield "응답을 생성할 수 없습니다."
                return

            raw_response = choices[0].get("message", {}).get("content", "")

            if not raw_response:
                logger.warning("Empty response from LLM")
                yield "응답을 생성할 수 없습니다."
                return

            # 원본 응답 로깅
            logger.info(f"[RAW LLM RESPONSE]\n{raw_response}")

            # 후처리: 정규화 및 정제
            cleaned_response = clean_stream_chunk(raw_response)

            # 정제된 응답 로깅
            if cleaned_response != raw_response:
                logger.info(f"[CLEANED LLM RESPONSE]\n{cleaned_response}")

            # 정제된 응답 반환
            yield cleaned_response

    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        yield f"오류가 발생했습니다: {str(e)}"
