from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, Any, Dict, List
from datetime import datetime


class PDFUploadResponse(BaseModel):
    pdf_id: str
    filename: str
    chunk_count: int
    message: str
    task_id: Optional[str] = None  # Celery task ID (비동기 처리용)


class PDFInfo(BaseModel):
    pdf_id: str
    filename: str
    file_path: Optional[str] = None
    created_at: Optional[datetime] = None


class PDFListResponse(BaseModel):
    pdfs: list[PDFInfo]



class ChatRequest(BaseModel):
    notebook_id: str  # 노트북 ID (필수)
    pdf_id: str
    filename: str
    message: str
    history: Optional[list[dict]] = []


class ChatMessage(BaseModel):
    role: str
    content: str


# ===== Notebook Schemas =====

class NotebookCreate(BaseModel):
    title: str
    created_by: str


class NotebookUpdate(BaseModel):
    title: str


class NotebookInfo(BaseModel):
    notebook_id: str
    title: str
    created_by: str
    created_at: Optional[datetime] = None


class NotebookListResponse(BaseModel):
    notebooks: list[NotebookInfo]


# ===== User Schemas =====

class UserCreate(BaseModel):
    username: str
    email: str


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None


class UserInfo(BaseModel):
    user_id: str
    username: str
    email: str
    created_at: Optional[datetime] = None


# ===== Chat History Schemas =====

class ChatHistoryMessage(BaseModel):
    """채팅 메시지 (저장용)"""
    message_id: str
    notebook_id: str
    role: str  # 'user' or 'assistant'
    content: str
    metadata: Optional[dict] = None
    created_at: Optional[datetime] = None
    feedback: Optional[FeedbackInfo] = None  # 피드백 정보 (있을 경우)


class ChatHistoryResponse(BaseModel):
    """채팅 기록 응답"""
    messages: list[ChatHistoryMessage]


class SaveChatRequest(BaseModel):
    """채팅 저장 요청"""
    notebook_id: str
    role: str
    content: str
    metadata: Optional[dict] = None


# ===== Task Schemas =====

class TaskStatusResponse(BaseModel):
    """Celery 작업 상태 응답"""
    task_id: str
    state: str  # PENDING, PROCESSING, SUCCESS, FAILURE, RETRY
    status: str
    progress: int  # 0-100
    filename: Optional[str] = None
    chunk_count: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ===== Feedback Schemas =====

class FeedbackSource(BaseModel):
    """피드백에 포함된 소스 정보 (RAG 검색 결과)"""
    text: str
    expanded_text: Optional[str] = None
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None
    metadata: Optional[dict] = None  # filename, page_no, chunk_index


class FeedbackCreate(BaseModel):
    """피드백 생성 요청"""
    message_id: str  # assistant 메시지의 ID
    is_positive: bool  # True=👍, False=👎
    comment: Optional[str] = None


class FeedbackUpdate(BaseModel):
    """피드백 수정 요청"""
    is_positive: Optional[bool] = None
    comment: Optional[str] = None


class FeedbackInfo(BaseModel):
    """피드백 정보 (응답용)"""
    feedback_id: str
    message_id: str
    notebook_id: str
    is_positive: bool
    comment: Optional[str] = None
    question_content: str
    answer_content: str
    sources: Optional[list[FeedbackSource]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class FeedbackListResponse(BaseModel):
    """피드백 목록 응답"""
    feedbacks: list[FeedbackInfo]
    total_count: Optional[int] = None
