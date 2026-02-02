from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
import json
import uuid

from app.models.schemas import (
    ChatRequest,
    SaveChatRequest,
    ChatHistoryResponse,
    ChatHistoryMessage,
)
from app.services.rag import get_rag_service
from app.services.rag.llm_client import generate_follow_up_questions
from app.services.rag.config import RAGConfig
from app.services.database import mysql
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/messages")
async def create_chat_message(
    request: ChatRequest,
    use_parallel: bool = True  # 프론트엔드 호환성을 위해 받지만 항상 병렬 실행
):
    """
    채팅 메시지 생성 엔드포인트 - SSE 스트리밍 응답

    Features:
    - 하이브리드 검색 (Dense + Sparse vectors with RRF fusion) - 항상 병렬 실행
    - Cross-Encoder Reranking
    - 이웃 청크 확장 (Context expansion)
    - 채팅 기록 자동 저장 (user, assistant 메시지)
    """
    service = get_rag_service()
    
    # notebook_id가 있는지 확인 (없으면 요청 거부)
    notebook_id = getattr(request, 'notebook_id', None)
    if not notebook_id:
        raise HTTPException(status_code=400, detail="notebook_id is required")
    
    # 노트북 존재 여부 확인
    notebook = mysql.get_notebook_by_id(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail=f"Notebook not found: {notebook_id}")
    
    # 1. User 메시지 저장
    user_message_id = str(uuid.uuid4())
    user_metadata = json.dumps({
        "pdf_id": request.pdf_id,
        "filename": request.filename,
    }, ensure_ascii=False)

    if not mysql.create_chat_message(
        message_id=user_message_id,
        notebook_id=notebook_id,
        role="user",
        content=request.message,
        metadata=user_metadata,
    ):
        logger.error(f"[CHAT] Failed to save user message to DB for notebook {notebook_id}")
        # DB 저장 실패해도 계속 진행 (답변은 제공)
    
    # 2. Assistant 응답 생성 및 저장
    assistant_message_id = str(uuid.uuid4())
    assistant_response = ""
    
    async def event_generator():
        nonlocal assistant_response

        # 1. 답변 스트리밍
        async for chunk in service.generate_response_stream(
            pdf_id=request.pdf_id,
            filename=request.filename,
            query=request.message,
            chat_history=request.history or [],
        ):
            assistant_response += chunk
            yield {"event": "message", "data": chunk}

        # 2. 답변 저장
        assistant_metadata = json.dumps({
            "pdf_id": request.pdf_id,
            "filename": request.filename,
        }, ensure_ascii=False)

        if not mysql.create_chat_message(
            message_id=assistant_message_id,
            notebook_id=notebook_id,
            role="assistant",
            content=assistant_response,
            metadata=assistant_metadata,
        ):
            logger.error(f"[CHAT] Failed to save assistant message to DB for notebook {notebook_id}")

        # 3. 후속 질문 생성 (주석 처리: 로그가 너무 길어짐)
        # if RAGConfig.ENABLE_FOLLOW_UP_QUESTIONS and assistant_response.strip():
        #     try:
        #         logger.info("[FOLLOW-UP] Starting follow-up question generation")
        #         follow_up_questions = await generate_follow_up_questions(
        #             query=request.message,
        #             answer=assistant_response,
        #             top_k=RAGConfig.FOLLOW_UP_QUESTIONS_COUNT
        #         )
        #
        #         if follow_up_questions:
        #             logger.info(f"[FOLLOW-UP] Generated {len(follow_up_questions)} questions")
        #             yield {
        #                 "event": "follow_up",
        #                 "data": json.dumps({
        #                     "questions": follow_up_questions
        #                 }, ensure_ascii=False)
        #             }
        #         else:
        #             logger.warning("[FOLLOW-UP] No questions generated")
        #
        #     except Exception as e:
        #         logger.error(f"[FOLLOW-UP] Failed to generate follow-up questions: {e}")
        #         # 에러 발생해도 답변은 이미 전송되었으므로 계속 진행

    return EventSourceResponse(event_generator())


@router.get("/history/{notebook_id}", response_model=ChatHistoryResponse)
async def get_chat_history(notebook_id: str, limit: int = 100, offset: int = 0):
    """
    노트북의 채팅 기록 조회 (시간순 정렬)

    Args:
        notebook_id: 노트북 ID
        limit: 최대 조회 개수 (기본 100)
        offset: 오프셋 (기본 0)

    Returns:
        ChatHistoryResponse: 채팅 메시지 리스트
    """
    logger.info(f"[CHAT] Retrieving chat history for notebook: {notebook_id} (limit={limit}, offset={offset})")

    # 노트북 존재 여부 확인
    notebook = mysql.get_notebook_by_id(notebook_id)
    if not notebook:
        logger.warning(f"[CHAT] Notebook not found: {notebook_id}")
        raise HTTPException(status_code=404, detail=f"Notebook not found: {notebook_id}")

    # 채팅 기록 조회
    messages = mysql.get_chat_history(notebook_id, limit, offset)
    logger.info(f"[CHAT] Found {len(messages)} messages in DB")

    # 응답 변환
    chat_messages = []
    try:
        for msg in messages:
            chat_messages.append(ChatHistoryMessage(
                message_id=msg["id"],
                notebook_id=msg["notebook_id"],
                role=msg["role"],
                content=msg["content"],
                metadata=json.loads(msg["metadata"]) if msg.get("metadata") else None,
                created_at=msg.get("created_at"),
            ))
        logger.info(f"[CHAT] Successfully converted {len(chat_messages)} messages to response")
    except Exception as e:
        logger.error(f"[CHAT] Failed to convert messages to response: {e}")
        import traceback
        logger.error(f"[CHAT] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat history: {str(e)}")

    return ChatHistoryResponse(messages=chat_messages)


@router.get("/history/{notebook_id}/recent", response_model=ChatHistoryResponse)
async def get_recent_chat_history(notebook_id: str, limit: int = 10):
    """
    노트북의 최근 채팅 기록 조회
    
    Args:
        notebook_id: 노트북 ID
        limit: 최대 조회 개수 (기본 10)
    
    Returns:
        ChatHistoryResponse: 최근 채팅 메시지 리스트 (오래된 순서)
    """
    # 노트북 존재 여부 확인
    notebook = mysql.get_notebook_by_id(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail=f"Notebook not found: {notebook_id}")
    
    # 최근 채팅 기록 조회
    messages = mysql.get_recent_chat_history(notebook_id, limit)
    
    # 응답 변환
    chat_messages = []
    for msg in messages:
        chat_messages.append(ChatHistoryMessage(
            message_id=msg["id"],
            notebook_id=msg["notebook_id"],
            role=msg["role"],
            content=msg["content"],
            metadata=json.loads(msg["metadata"]) if msg.get("metadata") else None,
            created_at=msg.get("created_at"),
        ))
    
    return ChatHistoryResponse(messages=chat_messages)


@router.delete("/history/{notebook_id}")
async def delete_chat_history(notebook_id: str):
    """
    노트북의 모든 채팅 기록 삭제
    
    Args:
        notebook_id: 노트북 ID
    
    Returns:
        성공 메시지
    """
    # 노트북 존재 여부 확인
    notebook = mysql.get_notebook_by_id(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail=f"Notebook not found: {notebook_id}")
    
    # 채팅 기록 삭제
    success = mysql.delete_chat_history(notebook_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete chat history")
    
    return {"message": f"Chat history deleted for notebook {notebook_id}"}


@router.delete("/message/{message_id}")
async def delete_chat_message(message_id: str):
    """
    특정 채팅 메시지 삭제
    
    Args:
        message_id: 메시지 ID
    
    Returns:
        성공 메시지
    """
    success = mysql.delete_chat_message(message_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Message not found: {message_id}")
    
    return {"message": f"Message {message_id} deleted"}

