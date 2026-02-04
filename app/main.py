from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.routers import pdf, chat, notebook, user, tasks, feedback
from app.core.logging_config import setup_logging
from app.middleware.request_id import RequestIDMiddleware

load_dotenv()

# Initialize structured logging
setup_logging()

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

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


@app.get("/health")
async def health_check():
    return {"status": "ok"}
