import uuid
import shutil
import base64
from pathlib import Path
from urllib.parse import unquote
from fastapi import APIRouter, UploadFile, File, HTTPException, Form

from app.models.schemas import PDFUploadResponse, PDFListResponse, PDFInfo
from app.services.database import qdrant as qdrant_service
from app.services.database import mysql as mysql_service
from app.tasks import process_pdf_task, calculate_priority_from_size

router = APIRouter()

# 베이스 저장 디렉토리
BASE_STORAGE_DIR = Path("./storage")


@router.post("", response_model=PDFUploadResponse)
async def create_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    notebook_id: str = Form(...),
    generate_image_description: bool = Form(default=False)
):
    """
    PDF 업로드 및 비동기 처리

    Args:
        file: PDF 파일
        user_id: 사용자 ID
        notebook_id: 노트북 ID
        generate_image_description: 이미지 description 생성 여부 (기본값: False)

    저장 구조: ./storage/{user_id}/{file_name}/{file}

    Note: PDF 처리는 항상 백그라운드에서 비동기로 처리됩니다.
    """
    # URL 인코딩된 파일명 디코딩 (한글 파일명 지원)
    original_filename = unquote(file.filename) if file.filename else file.filename

    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    content = await file.read()
    file_size = len(content)
    pdf_id = str(uuid.uuid4())

    # 파일 해시 계산 (중복 체크용)
    file_hash = mysql_service.calculate_file_hash(content)

    # 중복 파일 체크
    existing_file = mysql_service.get_file_by_hash(file_hash)
    if existing_file:
        return PDFUploadResponse(
            pdf_id=existing_file["id"],
            filename=existing_file["file_name"],
            chunk_count=0,
            message=f"동일한 파일이 이미 존재합니다: {existing_file['file_name']}",
        )

    # PDF 파일명에서 확장자 제거
    pdf_name = Path(original_filename).stem

    # 디렉토리 구조 생성: storage/{user_id}/{pdf_name}/
    user_dir = BASE_STORAGE_DIR / user_id
    pdf_dir = user_dir / pdf_name
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # PDF 파일 저장
    pdf_file_path = pdf_dir / original_filename
    try:
        with open(pdf_file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 파일 저장 실패: {str(e)}")

    # 파일 크기 기반 우선순위 계산
    priority = calculate_priority_from_size(file_size)

    # Base64 인코딩 (Celery 전달용)
    content_b64 = base64.b64encode(content).decode('utf-8')

    # Celery 작업 생성 (비동기 처리)
    task = process_pdf_task.apply_async(
        args=[
            pdf_id,
            content_b64,
            original_filename,
            str(pdf_file_path),
            file_hash,
            user_id,
            notebook_id,
            generate_image_description,
        ],
        priority=priority,
        queue='pdf_processing',
    )

    return PDFUploadResponse(
        pdf_id=pdf_id,
        filename=original_filename,
        chunk_count=0,
        message=f"PDF 업로드 완료. 백그라운드에서 처리 중입니다. (우선순위: {priority}/9, 파일 크기: {file_size / 1024 / 1024:.2f}MB)",
        task_id=task.id,
    )


@router.get("", response_model=PDFListResponse)
async def get_pdfs(user_id: str = None):
    """저장된 PDF 목록 조회 (MySQL에서 조회)"""
    files = mysql_service.get_file_list(uploaded_by=user_id)
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


@router.get("/notebook/{notebook_id}", response_model=PDFListResponse)
async def get_pdfs_by_notebook(notebook_id: str):
    """노트북에 연결된 PDF 목록 조회"""
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


@router.delete("/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """
    PDF 삭제 (Qdrant, MySQL, 파일 시스템에서 모두 삭제)

    삭제 순서:
    1. 파일 정보 조회 (MySQL)
    2. Qdrant에서 컬렉션 삭제
    3. MySQL에서 파일 정보 삭제
    4. 파일 시스템에서 디렉토리 삭제
    """
    # 1. MySQL에서 파일 정보 조회
    file_info = mysql_service.get_file_by_id(pdf_id)

    if not file_info:
        raise HTTPException(status_code=404, detail="PDF를 찾을 수 없습니다.")

    filename = file_info["file_name"]
    file_path = file_info.get("file_path")

    errors = []

    # 2. Qdrant에서 컬렉션 삭제
    try:
        qdrant_deleted = qdrant_service.delete_pdf(pdf_id, filename)
        if not qdrant_deleted:
            errors.append("Qdrant 컬렉션 삭제 실패")
    except Exception as e:
        errors.append(f"Qdrant 삭제 중 오류: {str(e)}")

    # 3. MySQL에서 파일 정보 삭제 (notebook_file_link와 file_info)
    try:
        mysql_deleted = mysql_service.delete_file_info(pdf_id)
        if not mysql_deleted:
            errors.append("MySQL 파일 정보 삭제 실패")
    except Exception as e:
        errors.append(f"MySQL 삭제 중 오류: {str(e)}")

    # 4. 파일 시스템에서 삭제
    if file_path:
        try:
            pdf_path = Path(file_path)
            if pdf_path.exists():
                # PDF 파일이 있는 디렉토리 전체 삭제
                pdf_dir = pdf_path.parent
                if pdf_dir.exists():
                    shutil.rmtree(pdf_dir)
        except Exception as e:
            errors.append(f"파일 시스템 삭제 중 오류: {str(e)}")

    # 에러가 있으면 경고와 함께 응답
    if errors:
        return {
            "message": "PDF 삭제가 부분적으로 완료되었습니다.",
            "warnings": errors
        }

    return {"message": "PDF가 성공적으로 삭제되었습니다."}
