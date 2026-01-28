# -*- coding: utf-8 -*-
"""
Task 상태 조회 라우터
"""
from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult

from app.celery_app import celery_app
from app.models.schemas import TaskStatusResponse

router = APIRouter()


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Celery 작업 상태 조회

    Args:
        task_id: Celery task ID

    Returns:
        작업 상태 정보
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state == 'PENDING':
        # 작업이 아직 시작되지 않았거나 존재하지 않음
        response = {
            'task_id': task_id,
            'state': 'PENDING',
            'status': 'Task is waiting in queue',
            'progress': 0,
        }
    elif task_result.state == 'PROCESSING':
        # 작업 진행 중
        response = {
            'task_id': task_id,
            'state': 'PROCESSING',
            'status': task_result.info.get('status', 'Processing'),
            'progress': task_result.info.get('progress', 50),
            'filename': task_result.info.get('filename'),
            'chunk_count': task_result.info.get('chunk_count'),
        }
    elif task_result.state == 'SUCCESS':
        # 작업 완료
        result = task_result.result
        response = {
            'task_id': task_id,
            'state': 'SUCCESS',
            'status': 'Task completed successfully',
            'progress': 100,
            'result': result,
        }
    elif task_result.state == 'FAILURE':
        # 작업 실패
        response = {
            'task_id': task_id,
            'state': 'FAILURE',
            'status': str(task_result.info),
            'progress': 0,
            'error': str(task_result.info),
        }
    elif task_result.state == 'RETRY':
        # 재시도 중
        response = {
            'task_id': task_id,
            'state': 'RETRY',
            'status': 'Task is retrying after failure',
            'progress': 0,
        }
    else:
        # 기타 상태
        response = {
            'task_id': task_id,
            'state': task_result.state,
            'status': str(task_result.info),
            'progress': 0,
        }

    return response


@router.get("/result/{task_id}")
async def get_task_result(task_id: str):
    """
    작업 결과 조회 (완료된 작업만)

    Args:
        task_id: Celery task ID

    Returns:
        작업 결과
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result.ready():
        raise HTTPException(
            status_code=202,
            detail=f"Task is still processing. Current state: {task_result.state}"
        )

    if task_result.successful():
        return {
            'task_id': task_id,
            'state': 'SUCCESS',
            'result': task_result.result,
        }
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Task failed: {str(task_result.info)}"
        )


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """
    작업 취소

    Args:
        task_id: Celery task ID

    Returns:
        취소 결과
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state in ['SUCCESS', 'FAILURE']:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task in state: {task_result.state}"
        )

    task_result.revoke(terminate=True)

    return {
        'task_id': task_id,
        'message': 'Task cancellation requested',
    }
