import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.routers import pdf, chat, notebook, user, tasks, feedback, log_viewer
from app.core.logging_config import setup_logging
from app.middleware.request_id import RequestIDMiddleware
from app.services.rag.sleep_manager import initialize_sleep_manager

load_dotenv()

# Initialize structured logging
setup_logging()

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Initialize vLLM Sleep Manager (if enabled)
VLLM_SLEEP_ENABLED = os.getenv("VLLM_SLEEP_ENABLED", "false").lower() == "true"
VLLM_DEV_URL = os.getenv("VLLM_DEV_URL", "http://localhost:8000")
VLLM_SLEEP_IDLE_TIMEOUT = int(os.getenv("VLLM_SLEEP_IDLE_TIMEOUT", "300"))  # 5분 기본값
VLLM_SLEEP_CHECK_INTERVAL = int(os.getenv("VLLM_SLEEP_CHECK_INTERVAL", "60"))  # 1분 체크 주기

sleep_manager = initialize_sleep_manager(
    vllm_dev_url=VLLM_DEV_URL,
    idle_timeout=VLLM_SLEEP_IDLE_TIMEOUT,
    check_interval=VLLM_SLEEP_CHECK_INTERVAL,
    enabled=VLLM_SLEEP_ENABLED
)

# Add Request ID middleware (BEFORE other middlewares)
app.add_middleware(RequestIDMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdf.router, prefix="/api/v1/file", tags=["File"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(notebook.router, prefix="/api/v1/notebook", tags=["Notebook"])
app.include_router(user.router, prefix="/api/v1/user", tags=["User"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["Tasks"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])
app.include_router(log_viewer.router)  # Log viewer at root level


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 초기화"""
    if sleep_manager:
        # 서버 재시작 시 vLLM이 sleep 상태일 수 있으므로 wake 시도
        try:
            await sleep_manager.wake()
        except Exception:
            # Wake 실패는 무시 (모델이 이미 active 상태일 수 있음)
            pass

        sleep_manager.start_background_task()


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 정리"""
    if sleep_manager:
        sleep_manager.stop_background_task()


@app.get("/health")
async def health_check():
    return {"status": "ok"}
