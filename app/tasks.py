# -*- coding: utf-8 -*-
"""
Celery Tasks - PDF 처리 비동기 작업
"""
import logging
from pathlib import Path
from typing import Dict, Any

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.celery_app import celery_app
from app.services.document import process_pdf_to_chunks, IntegratedParserConfig
from app.services.database import qdrant as qdrant_service
from app.services.database import mysql as mysql_service

log = logging.getLogger(__name__)


class CallbackTask(Task):
    """작업 상태 업데이트를 위한 커스텀 Task 클래스"""

    def on_success(self, retval, task_id, args, kwargs):
        """작업 성공 시 호출"""
        log.info(f"Task {task_id} completed successfully")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """작업 실패 시 호출"""
        log.error(f"Task {task_id} failed: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """작업 재시도 시 호출"""
        log.warning(f"Task {task_id} retrying: {exc}")


@celery_app.task(
    base=CallbackTask,
    bind=True,
    name="app.tasks.process_pdf_task",
    max_retries=3,
    default_retry_delay=60,
)
def process_pdf_task(
    self,
    pdf_id: str,
    file_content_b64: str,
    filename: str,
    file_path: str,
    file_hash: str,
    user_id: str,
    notebook_id: str,
    generate_image_description: bool = False,
) -> Dict[str, Any]:
    """
    PDF 처리 비동기 작업

    Args:
        self: Celery task instance (bind=True)
        pdf_id: PDF UUID
        file_content_b64: Base64 인코딩된 PDF 바이너리
        filename: 파일명
        file_path: 파일 저장 경로
        file_hash: 파일 해시
        user_id: 사용자 ID
        notebook_id: 노트북 ID
        generate_image_description: 이미지 description 생성 여부

    Returns:
        처리 결과 딕셔너리
    """
    try:
        # 작업 시작 상태 업데이트
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Processing PDF with Docling',
                'progress': 10,
                'filename': filename,
            }
        )

        # Base64 디코딩
        import base64
        file_content = base64.b64decode(file_content_b64)

        # 출력 디렉토리 설정
        pdf_dir = Path(file_path).parent
        output_path = pdf_dir / "docling_output"
        output_path.mkdir(parents=True, exist_ok=True)

        # 파서 설정 (고급 모드: VLM/LLM description)
        config = IntegratedParserConfig(
            enable_image_description=generate_image_description,
            enable_table_description=generate_image_description,  # 임시: 같은 값 사용
        )

        # 디버깅: config 확인
        log.info(f"Parser config: enable_image_description={config.enable_image_description}, enable_table_description={config.enable_table_description}")

        # 상태 업데이트: Docling 처리 중
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Running Docling parser (OCR detection)',
                'progress': 30,
                'filename': filename,
            }
        )

        # Progress callback 정의
        def progress_callback(meta):
            """Docling 처리 진행 상황을 Celery task 상태로 업데이트"""
            self.update_state(
                state='PROCESSING',
                meta=meta
            )

        # PDF 처리
        chunks_data, metadata = process_pdf_to_chunks(
            file_content=file_content,
            filename=filename,
            output_dir=output_path,
            source_id=pdf_id,
            config=config,
            progress_callback=progress_callback,
        )

        if not chunks_data:
            raise ValueError("PDF에서 청크를 생성할 수 없습니다.")

        # Chunking 결과를 JSON으로 저장 (디버깅 및 백업용)
        import json
        chunks_json_path = output_path / f"{Path(filename).stem}_chunks.json"
        try:
            with open(chunks_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': metadata,
                    'chunks': chunks_data,
                }, f, ensure_ascii=False, indent=2)
            log.info(f"Chunks JSON saved: {chunks_json_path}")
        except Exception as e:
            log.warning(f"Failed to save chunks JSON: {e}")

        # 상태 업데이트: Qdrant 업로드 중
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Uploading chunks to Qdrant vector DB',
                'progress': 70,
                'filename': filename,
                'chunk_count': len(chunks_data),
            }
        )

        # Qdrant에 저장
        chunk_count = qdrant_service.upsert_chunks(
            pdf_id=pdf_id,
            filename=filename,
            chunks_data=chunks_data,
        )

        # 상태 업데이트: MySQL 메타데이터 저장 중
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Saving metadata to MySQL',
                'progress': 90,
                'filename': filename,
                'chunk_count': chunk_count,
            }
        )

        # MySQL에 파일 메타데이터 저장
        mysql_service.create_file_info(
            file_id=pdf_id,
            file_name=filename,
            file_path=file_path,
            file_hash=file_hash,
            uploaded_by=user_id,
            extend=Path(filename).suffix.lstrip("."),
        )

        # 노트북에 파일 연결
        if notebook_id:
            mysql_service.link_file_to_notebook(notebook_id, pdf_id)

        # 최종 결과 반환
        result = {
            'status': 'SUCCESS',
            'pdf_id': pdf_id,
            'filename': filename,
            'chunk_count': chunk_count,
            'table_count': metadata['table_count'],
            'picture_count': metadata['picture_count'],
            'ocr_used': metadata['ocr_used'],
            'ocr_reason': metadata['ocr_reason'],
            'message': f'PDF 처리 완료 (청크: {chunk_count}, 테이블: {metadata["table_count"]}, 이미지: {metadata["picture_count"]})',
        }

        log.info(f"Task {self.request.id} completed: {result}")
        return result

    except SoftTimeLimitExceeded:
        # Soft timeout 발생
        log.error(f"Task {self.request.id} exceeded soft time limit")
        raise self.retry(countdown=300, exc=SoftTimeLimitExceeded())

    except Exception as e:
        log.error(f"Task {self.request.id} failed: {str(e)}", exc_info=True)

        # 재시도 로직
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60, exc=e)

        # 최대 재시도 초과 시 실패 반환
        return {
            'status': 'FAILURE',
            'error': str(e),
            'error_type': type(e).__name__,
            'filename': filename,
        }


@celery_app.task(name="app.tasks.get_file_size_priority")
def calculate_priority_from_size(file_size_bytes: int) -> int:
    """
    파일 크기에 따라 우선순위 계산

    작은 파일일수록 높은 우선순위 (낮은 숫자)
    - < 1MB: priority 9 (최우선)
    - 1-5MB: priority 7
    - 5-10MB: priority 5 (기본)
    - 10-50MB: priority 3
    - > 50MB: priority 1 (최저)
    """
    MB = 1024 * 1024

    if file_size_bytes < MB:
        return 9
    elif file_size_bytes < 5 * MB:
        return 7
    elif file_size_bytes < 10 * MB:
        return 5
    elif file_size_bytes < 50 * MB:
        return 3
    else:
        return 1
