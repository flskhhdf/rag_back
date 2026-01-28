# -*- coding: utf-8 -*-
"""
Celery Application 설정
"""
import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Redis URL 설정 (환경변수에서 가져오기)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery 앱 생성
celery_app = Celery(
    "rag_backend",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks"]  # tasks 모듈 포함
)

# Celery 설정
celery_app.conf.update(
    # 작업 결과 만료 시간 (24시간)
    result_expires=86400,

    # 작업 타임아웃 (2시간)
    task_time_limit=7200,

    # Soft 타임아웃 (1시간 50분)
    task_soft_time_limit=6600,

    # 작업 승인 (late ack - 작업 완료 후 승인)
    task_acks_late=True,

    # Worker당 동시 작업 수 (CPU 코어 수에 따라 조정)
    worker_concurrency=2,

    # Worker prefetch multiplier (메모리 절약)
    worker_prefetch_multiplier=1,

    # 작업 우선순위 활성화
    task_default_priority=5,
    broker_transport_options={
        'priority_steps': list(range(10)),  # 0-9 우선순위
        'queue_order_strategy': 'priority',
    },

    # 결과 직렬화
    result_serializer='json',
    task_serializer='json',
    accept_content=['json'],

    # 타임존
    timezone='Asia/Seoul',
    enable_utc=True,
)

# 큐 라우팅 설정
celery_app.conf.task_routes = {
    'app.tasks.process_pdf_task': {
        'queue': 'pdf_processing',
        'routing_key': 'pdf.process',
    }
}
