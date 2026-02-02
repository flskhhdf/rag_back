import uuid
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    NotebookCreate, NotebookUpdate, NotebookInfo, NotebookListResponse, PDFListResponse, PDFInfo
)
from app.services.database import mysql as mysql_service

router = APIRouter()


@router.post("", response_model=NotebookInfo)
async def create_notebook(notebook: NotebookCreate):
    """노트북 생성"""
    notebook_id = str(uuid.uuid4())
    
    success = mysql_service.create_notebook(
        notebook_id=notebook_id,
        title=notebook.title,
        created_by=notebook.created_by,
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="노트북 생성 실패")
    
    created = mysql_service.get_notebook_by_id(notebook_id)
    return NotebookInfo(
        notebook_id=created["id"],
        title=created["title"],
        created_by=created["created_by"],
        created_at=created.get("created_at"),
    )


@router.get("", response_model=NotebookListResponse)
async def get_notebooks(user_id: str = None):
    """노트북 목록 조회"""
    if user_id:
        notebooks = mysql_service.get_notebooks_by_user(user_id)
    else:
        notebooks = mysql_service.get_all_notebooks()
    
    return NotebookListResponse(
        notebooks=[
            NotebookInfo(
                notebook_id=n["id"],
                title=n["title"],
                created_by=n["created_by"],
                created_at=n.get("created_at"),
            )
            for n in notebooks
        ]
    )


@router.get("/{notebook_id}", response_model=NotebookInfo)
async def get_notebook(notebook_id: str):
    """노트북 상세 조회"""
    notebook = mysql_service.get_notebook_by_id(notebook_id)
    
    if not notebook:
        raise HTTPException(status_code=404, detail="노트북을 찾을 수 없습니다.")
    
    return NotebookInfo(
        notebook_id=notebook["id"],
        title=notebook["title"],
        created_by=notebook["created_by"],
        created_at=notebook.get("created_at"),
    )


@router.put("/{notebook_id}", response_model=NotebookInfo)
async def update_notebook(notebook_id: str, update: NotebookUpdate):
    """노트북 수정"""
    existing = mysql_service.get_notebook_by_id(notebook_id)
    if not existing:
        raise HTTPException(status_code=404, detail="노트북을 찾을 수 없습니다.")
    
    success = mysql_service.update_notebook(notebook_id, update.title)
    if not success:
        raise HTTPException(status_code=500, detail="노트북 수정 실패")
    
    updated = mysql_service.get_notebook_by_id(notebook_id)
    return NotebookInfo(
        notebook_id=updated["id"],
        title=updated["title"],
        created_by=updated["created_by"],
        created_at=updated.get("created_at"),
    )


@router.delete("/{notebook_id}")
async def delete_notebook(notebook_id: str):
    """
    노트북 삭제 (파일은 유지, 링크만 끊음)

    삭제되는 것:
    - notebook 레코드
    - notebook_file_link (파일 연결)
    - chat_history (채팅 기록)

    유지되는 것:
    - file_info (파일 메타데이터)
    - 파일시스템의 실제 파일
    - Qdrant의 벡터 데이터
    """
    import logging
    logger = logging.getLogger(__name__)

    # 노트북 존재 여부 확인
    existing = mysql_service.get_notebook_by_id(notebook_id)
    if not existing:
        raise HTTPException(status_code=404, detail="노트북을 찾을 수 없습니다.")

    logger.info(f"[NOTEBOOK] Deleting notebook: {notebook_id}")

    # 1. 채팅 기록 삭제
    try:
        mysql_service.delete_chat_history(notebook_id)
        logger.info(f"[NOTEBOOK] Deleted chat history for notebook: {notebook_id}")
    except Exception as e:
        logger.error(f"[NOTEBOOK] Failed to delete chat history: {e}")
        # 에러가 나도 계속 진행

    # 2. 노트북 삭제 (notebook_file_link도 CASCADE로 함께 삭제됨)
    success = mysql_service.delete_notebook(notebook_id)
    if not success:
        raise HTTPException(status_code=500, detail="노트북 삭제 실패")

    logger.info(f"[NOTEBOOK] Successfully deleted notebook: {notebook_id}")

    return {
        "message": "노트북이 삭제되었습니다. (연결된 파일은 유지됩니다)"
    }


# ===== 노트북-파일 연결 =====

@router.get("/{notebook_id}/files", response_model=PDFListResponse)
async def get_notebook_files(notebook_id: str):
    """노트북에 연결된 파일 목록"""
    existing = mysql_service.get_notebook_by_id(notebook_id)
    if not existing:
        raise HTTPException(status_code=404, detail="노트북을 찾을 수 없습니다.")
    
    files = mysql_service.get_files_by_notebook(notebook_id)
    return PDFListResponse(
        pdfs=[
            PDFInfo(
                pdf_id=f["id"],
                filename=f["file_name"],
                file_path=f.get("file_path"),
                created_at=f.get("created_at"),
            )
            for f in files
        ]
    )


@router.post("/{notebook_id}/files/{file_id}")
async def link_file_to_notebook(notebook_id: str, file_id: str):
    """파일을 노트북에 연결"""
    notebook = mysql_service.get_notebook_by_id(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="노트북을 찾을 수 없습니다.")
    
    file = mysql_service.get_file_by_id(file_id)
    if not file:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    success = mysql_service.link_file_to_notebook(notebook_id, file_id)
    if not success:
        raise HTTPException(status_code=500, detail="파일 연결 실패")
    
    return {"message": "파일이 노트북에 연결되었습니다."}


@router.delete("/{notebook_id}/files/{file_id}")
async def unlink_file_from_notebook(notebook_id: str, file_id: str):
    """노트북에서 파일 연결 해제"""
    success = mysql_service.unlink_file_from_notebook(notebook_id, file_id)
    if not success:
        raise HTTPException(status_code=500, detail="파일 연결 해제 실패")
    
    return {"message": "파일 연결이 해제되었습니다."}
