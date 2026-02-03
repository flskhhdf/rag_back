import uuid
import json
from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from app.models.schemas import (
    FeedbackCreate,
    FeedbackUpdate,
    FeedbackInfo,
    FeedbackListResponse,
    FeedbackSource,
)
from app.services.database import mysql

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=FeedbackInfo)
async def create_feedback(request: FeedbackCreate):
    """
    피드백 생성 또는 업데이트

    - message_id는 assistant 메시지의 ID여야 함
    - 동일한 message_id로 재요청하면 업데이트됨 (UPSERT)
    """
    # 1. 메시지 ID로 Q&A 페어 조회
    qa_pair = mysql.get_qa_pair_by_message_id(request.message_id)

    if not qa_pair:
        raise HTTPException(
            status_code=404,
            detail="메시지를 찾을 수 없거나 assistant 메시지가 아닙니다."
        )

    question = qa_pair["question"]
    answer = qa_pair["answer"]
    notebook_id = qa_pair["notebook_id"]

    # 2. 기존 피드백 확인 (UPSERT를 위해)
    existing_feedback = mysql.get_feedback_by_message_id(request.message_id)

    if existing_feedback:
        feedback_id = existing_feedback["id"]
        logger.info(f"[FEEDBACK] Updating existing feedback: {feedback_id}")
    else:
        feedback_id = str(uuid.uuid4())
        logger.info(f"[FEEDBACK] Creating new feedback: {feedback_id}")

    # 3. metadata에서 sources 추출
    sources_data = None
    if answer.get("metadata"):
        try:
            metadata = json.loads(answer["metadata"]) if isinstance(answer["metadata"], str) else answer["metadata"]
            sources = metadata.get("sources", [])

            if sources:
                # Clean and format sources
                cleaned_sources = []
                for src in sources:
                    cleaned_sources.append({
                        "text": src.get("text", "")[:500],  # Limit text length
                        "expanded_text": src.get("expanded_text", "")[:1000] if src.get("expanded_text") else None,
                        "rerank_score": src.get("rerank_score"),
                        "rrf_score": src.get("rrf_score"),
                        "metadata": {
                            "filename": src.get("metadata", {}).get("filename"),
                            "page_no": src.get("metadata", {}).get("page_no"),
                            "chunk_index": src.get("metadata", {}).get("chunk_index"),
                        }
                    })
                sources_data = json.dumps(cleaned_sources, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[FEEDBACK] Failed to extract sources from metadata: {e}")

    # 4. 피드백 저장
    success = mysql.create_feedback(
        feedback_id=feedback_id,
        message_id=request.message_id,
        notebook_id=notebook_id,
        is_positive=request.is_positive,
        comment=request.comment,
        question_content=question["content"],
        answer_content=answer["content"],
        sources=sources_data,
    )

    if not success:
        raise HTTPException(status_code=500, detail="피드백 생성 실패")

    # 5. 생성된 피드백 조회 및 반환
    created = mysql.get_feedback_by_id(feedback_id)

    # Parse sources from JSON
    sources_list = None
    if created.get("sources"):
        try:
            sources_json = json.loads(created["sources"]) if isinstance(created["sources"], str) else created["sources"]
            sources_list = [FeedbackSource(**src) for src in sources_json]
        except Exception as e:
            logger.warning(f"[FEEDBACK] Failed to parse sources: {e}")

    return FeedbackInfo(
        feedback_id=created["id"],
        message_id=created["message_id"],
        notebook_id=created["notebook_id"],
        is_positive=created["is_positive"],
        comment=created.get("comment"),
        question_content=created["question_content"],
        answer_content=created["answer_content"],
        sources=sources_list,
        created_at=created.get("created_at"),
        updated_at=created.get("updated_at"),
    )


@router.get("/message/{message_id}", response_model=Optional[FeedbackInfo])
async def get_feedback_by_message(message_id: str):
    """특정 메시지의 피드백 조회"""
    feedback = mysql.get_feedback_by_message_id(message_id)

    if not feedback:
        return None

    # Parse sources from JSON
    sources_list = None
    if feedback.get("sources"):
        try:
            sources_json = json.loads(feedback["sources"]) if isinstance(feedback["sources"], str) else feedback["sources"]
            sources_list = [FeedbackSource(**src) for src in sources_json]
        except Exception as e:
            logger.warning(f"[FEEDBACK] Failed to parse sources: {e}")

    return FeedbackInfo(
        feedback_id=feedback["id"],
        message_id=feedback["message_id"],
        notebook_id=feedback["notebook_id"],
        is_positive=feedback["is_positive"],
        comment=feedback.get("comment"),
        question_content=feedback["question_content"],
        answer_content=feedback["answer_content"],
        sources=sources_list,
        created_at=feedback.get("created_at"),
        updated_at=feedback.get("updated_at"),
    )


@router.put("/{feedback_id}", response_model=FeedbackInfo)
async def update_feedback(feedback_id: str, update: FeedbackUpdate):
    """피드백 수정 (thumbs up/down 변경 또는 코멘트 수정)"""
    existing = mysql.get_feedback_by_id(feedback_id)
    if not existing:
        raise HTTPException(status_code=404, detail="피드백을 찾을 수 없습니다.")

    success = mysql.update_feedback(
        feedback_id=feedback_id,
        is_positive=update.is_positive,
        comment=update.comment,
    )

    if not success:
        raise HTTPException(status_code=500, detail="피드백 수정 실패")

    updated = mysql.get_feedback_by_id(feedback_id)

    # Parse sources from JSON
    sources_list = None
    if updated.get("sources"):
        try:
            sources_json = json.loads(updated["sources"]) if isinstance(updated["sources"], str) else updated["sources"]
            sources_list = [FeedbackSource(**src) for src in sources_json]
        except Exception as e:
            logger.warning(f"[FEEDBACK] Failed to parse sources: {e}")

    return FeedbackInfo(
        feedback_id=updated["id"],
        message_id=updated["message_id"],
        notebook_id=updated["notebook_id"],
        is_positive=updated["is_positive"],
        comment=updated.get("comment"),
        question_content=updated["question_content"],
        answer_content=updated["answer_content"],
        sources=sources_list,
        created_at=updated.get("created_at"),
        updated_at=updated.get("updated_at"),
    )


@router.get("/notebook/{notebook_id}", response_model=FeedbackListResponse)
async def get_notebook_feedbacks(
    notebook_id: str,
    is_positive: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
):
    """
    노트북의 피드백 목록 조회 (분석용)

    Args:
        notebook_id: 노트북 ID
        is_positive: 필터링 (None=전체, true=긍정만, false=부정만)
        limit: 최대 조회 개수
        offset: 오프셋
    """
    # 노트북 존재 여부 확인
    notebook = mysql.get_notebook_by_id(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="노트북을 찾을 수 없습니다.")

    feedbacks = mysql.get_feedbacks_by_notebook(
        notebook_id=notebook_id,
        is_positive=is_positive,
        limit=limit,
        offset=offset,
    )

    feedback_list = []
    for fb in feedbacks:
        # Parse sources from JSON
        sources_list = None
        if fb.get("sources"):
            try:
                sources_json = json.loads(fb["sources"]) if isinstance(fb["sources"], str) else fb["sources"]
                sources_list = [FeedbackSource(**src) for src in sources_json]
            except Exception as e:
                logger.warning(f"[FEEDBACK] Failed to parse sources: {e}")

        feedback_list.append(FeedbackInfo(
            feedback_id=fb["id"],
            message_id=fb["message_id"],
            notebook_id=fb["notebook_id"],
            is_positive=fb["is_positive"],
            comment=fb.get("comment"),
            question_content=fb["question_content"],
            answer_content=fb["answer_content"],
            sources=sources_list,
            created_at=fb.get("created_at"),
            updated_at=fb.get("updated_at"),
        ))

    return FeedbackListResponse(
        feedbacks=feedback_list,
        total_count=len(feedback_list),
    )


@router.delete("/{feedback_id}")
async def delete_feedback(feedback_id: str):
    """피드백 삭제"""
    existing = mysql.get_feedback_by_id(feedback_id)
    if not existing:
        raise HTTPException(status_code=404, detail="피드백을 찾을 수 없습니다.")

    success = mysql.delete_feedback(feedback_id)
    if not success:
        raise HTTPException(status_code=500, detail="피드백 삭제 실패")

    return {"message": "피드백이 삭제되었습니다."}
