# -*- coding: utf-8 -*-
"""
Structured Logging Configuration

Features:
- JSON Lines (JSONL) format for analysis
- Human-readable text format for monitoring
- Request ID tracking via contextvars
- Log rotation and retention
- Environment-based log level (DEBUG in dev, INFO in prod)
"""
import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from contextvars import ContextVar
from pathlib import Path
from pythonjsonlogger import jsonlogger

# Context variable for request ID tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')


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
    text_file = log_path / 'rag.log'
    text_handler = RotatingFileHandler(
        text_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=30,  # 30 files
        encoding='utf-8'
    )
    text_handler.setLevel(logging.INFO)
    text_handler.setFormatter(CustomTextFormatter())
    root_logger.addHandler(text_handler)

    # 3. JSON File Handler (Machine-readable, INFO+)
    json_file = log_path / 'rag.jsonl'
    json_handler = RotatingFileHandler(
        json_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=30,
        encoding='utf-8'
    )
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(CustomJsonFormatter(
        '%(timestamp)s %(level)s %(logger)s %(request_id)s %(message)s'
    ))
    root_logger.addHandler(json_handler)

    # 4. DEBUG File Handler (Detailed logs, DEBUG+)
    if log_level == 'DEBUG':
        debug_file = log_path / 'debug.log'
        debug_handler = RotatingFileHandler(
            debug_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=10,
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(CustomTextFormatter())
        root_logger.addHandler(debug_handler)

    # 5. ERROR File Handler (Errors only)
    error_file = log_path / 'error.log'
    error_handler = RotatingFileHandler(
        error_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=30,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(CustomTextFormatter())
    root_logger.addHandler(error_handler)

    # Log initialization message
    root_logger.info(
        f"Logging initialized: level={log_level}, dir={log_path}",
        extra={'event': 'logging_initialized'}
    )

    return root_logger
