# -*- coding: utf-8 -*-
"""
Structured Logging Configuration

Features:
- JSON Lines (JSONL) format for analysis
- Human-readable text format for monitoring
- Request ID tracking via contextvars
- Log rotation and retention
- Environment-based log level (DEBUG in dev, INFO in prod)
- Date-based folder organization (logs/2026_02_04/)
"""
import os
import logging
import sys
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from contextvars import ContextVar
from pathlib import Path
from pythonjsonlogger import jsonlogger

# Context variable for request ID tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')


class DateFolderTimedRotatingFileHandler(TimedRotatingFileHandler):
    """날짜별 폴더에 로그 파일을 저장하는 핸들러

    매일 자정에 새로운 날짜 폴더(예: logs/2026_02_04/)를 생성하고
    그 안에 로그 파일(예: rag.log)을 저장합니다.
    """

    def __init__(self, base_dir, filename, when='midnight', interval=1, backupCount=30, encoding='utf-8'):
        self.base_dir = Path(base_dir)
        self.filename = filename
        self.backupCount = backupCount

        # 현재 날짜 폴더 생성
        current_date_folder = self.base_dir / datetime.now().strftime("%Y_%m_%d")
        current_date_folder.mkdir(parents=True, exist_ok=True)

        # 전체 경로
        full_path = current_date_folder / filename

        super().__init__(
            str(full_path),
            when=when,
            interval=interval,
            backupCount=0,  # 우리가 직접 관리
            encoding=encoding
        )
        self.suffix = ""  # suffix 사용 안 함

    def doRollover(self):
        """날짜가 바뀌면 새로운 날짜 폴더에 로그 파일 생성"""
        # 기존 파일 닫기
        if self.stream:
            self.stream.close()
            self.stream = None

        # 새로운 날짜 폴더 생성
        new_date_folder = self.base_dir / datetime.now().strftime("%Y_%m_%d")
        new_date_folder.mkdir(parents=True, exist_ok=True)

        # 새 파일 경로
        self.baseFilename = str(new_date_folder / self.filename)

        # 파일 열기
        self.stream = self._open()

        # 롤오버 시간 계산
        currentTime = int(time.time())
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        self.rolloverAt = newRolloverAt

        # 오래된 폴더 삭제 (backupCount 초과 시)
        self._delete_old_folders()

    def _delete_old_folders(self):
        """backupCount를 초과하는 오래된 날짜 폴더 삭제"""
        if self.backupCount <= 0:
            return

        # 날짜 폴더 목록 가져오기 (형식: 2026_02_04)
        date_folders = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and len(item.name) == 10 and item.name.count('_') == 2:
                try:
                    # 폴더명이 날짜 형식인지 확인
                    datetime.strptime(item.name, "%Y_%m_%d")
                    date_folders.append(item)
                except ValueError:
                    continue

        # 날짜 기준 정렬 (오래된 것부터)
        date_folders.sort()

        # backupCount를 초과하는 오래된 폴더 삭제
        while len(date_folders) > self.backupCount:
            old_folder = date_folders.pop(0)
            try:
                # 폴더 내 모든 파일 삭제
                for file in old_folder.iterdir():
                    file.unlink()
                # 폴더 삭제
                old_folder.rmdir()
            except Exception as e:
                # 삭제 실패는 무시
                pass


def get_request_id() -> str:
    """Get current request ID from context"""
    return request_id_var.get('')


def set_request_id(request_id: str):
    """Set request ID in context"""
    request_id_var.set(request_id)


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter that includes request_id from context
    """
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        # Add request_id from context
        log_record['request_id'] = get_request_id()

        # Add timestamp in ISO format
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'

        # Add logger name
        log_record['logger'] = record.name

        # Add level
        log_record['level'] = record.levelname


class CustomTextFormatter(logging.Formatter):
    """
    Custom text formatter for human-readable logs
    Format: 2026-01-07 10:19:31.234 [INFO] [request-id] event_name | message
    """
    def format(self, record):
        request_id = get_request_id()
        request_id_str = f"[{request_id[:8]}]" if request_id else "[no-req]"

        # Extract event and message from record
        event = getattr(record, 'event', '')
        event_str = f"{event} | " if event else ""

        # Format: timestamp [LEVEL] [req-id] event | message
        formatted = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S.%f')[:-3]} "
            f"[{record.levelname}] {request_id_str} {event_str}{record.getMessage()}"
        )

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def setup_logging(log_level: str = None, log_dir: str = None):
    """
    Setup structured logging configuration

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR).
                   Defaults to LOG_LEVEL env var or INFO
        log_dir: Log directory path. Defaults to ./logs
    """
    # Determine log level from env or parameter
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    # Determine log directory
    if log_dir is None:
        log_dir = os.getenv('LOG_DIR', 'logs')

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all, filter in handlers

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 1. Console Handler (Human-readable, INFO+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(CustomTextFormatter())
    root_logger.addHandler(console_handler)

    # 2. Text File Handler (Human-readable, INFO+)
    text_handler = DateFolderTimedRotatingFileHandler(
        base_dir=log_path,
        filename='rag.log',
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    text_handler.setLevel(logging.INFO)
    text_handler.setFormatter(CustomTextFormatter())

    # search_analysis 로그는 제외 (search_analysis.log에만 기록)
    class ExcludeSearchAnalysisFilter(logging.Filter):
        def filter(self, record):
            return getattr(record, 'event_type', '') != 'search_analysis'

    text_handler.addFilter(ExcludeSearchAnalysisFilter())
    root_logger.addHandler(text_handler)

    # 3. JSON File Handler (Machine-readable, INFO+)
    json_handler = DateFolderTimedRotatingFileHandler(
        base_dir=log_path,
        filename='rag.jsonl',
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(CustomJsonFormatter(
        '%(timestamp)s %(level)s %(logger)s %(request_id)s %(message)s'
    ))
    # search_analysis 로그는 제외
    json_handler.addFilter(ExcludeSearchAnalysisFilter())
    root_logger.addHandler(json_handler)

    # 4. DEBUG File Handler (Detailed logs, DEBUG+)
    if log_level == 'DEBUG':
        debug_handler = DateFolderTimedRotatingFileHandler(
            base_dir=log_path,
            filename='debug.log',
            when='midnight',
            interval=1,
            backupCount=10,  # 10일치 보관 (DEBUG는 용량이 크므로 짧게)
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(CustomTextFormatter())
        root_logger.addHandler(debug_handler)

    # 5. ERROR File Handler (Errors only)
    error_handler = DateFolderTimedRotatingFileHandler(
        base_dir=log_path,
        filename='error.log',
        when='midnight',
        interval=1,
        backupCount=30,  # 30일치 보관
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(CustomTextFormatter())
    root_logger.addHandler(error_handler)

    # 6. Search Analysis Handler (검색 결과 분석용, 항상 생성)
    search_handler = DateFolderTimedRotatingFileHandler(
        base_dir=log_path,
        filename='search_analysis.log',
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    search_handler.setLevel(logging.INFO)
    search_handler.setFormatter(CustomTextFormatter())

    # 검색 분석 로그만 필터링 (event_type이 search_analysis인 것만)
    class SearchAnalysisFilter(logging.Filter):
        def filter(self, record):
            return getattr(record, 'event_type', '') == 'search_analysis'

    search_handler.addFilter(SearchAnalysisFilter())
    root_logger.addHandler(search_handler)

    # Log initialization message
    root_logger.info(
        f"Logging initialized: level={log_level}, dir={log_path}",
        extra={'event': 'logging_initialized'}
    )

    return root_logger
