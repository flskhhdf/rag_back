# -*- coding: utf-8 -*-
"""
Structured Logger Wrapper

Provides event-based logging with automatic request_id injection
and structured data handling.
"""
import logging
import time
from typing import Any, Dict, Optional
from .logging_config import get_request_id


class StructuredLogger:
    """
    Event-based structured logger wrapper

    Usage:
        logger = StructuredLogger(__name__)
        logger.info_event("search_completed", "Search completed", results_count=10, duration_ms=234.5)
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name

    def _log_event(
        self,
        level: int,
        event: str,
        message: str,
        duration_ms: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Internal method to log structured events

        Args:
            level: Log level (logging.INFO, logging.DEBUG, etc.)
            event: Event type (e.g., "search_completed")
            message: Human-readable message
            duration_ms: Optional duration in milliseconds
            data: Additional structured data
            **kwargs: Additional fields to include
        """
        extra = {
            'event': event,
            'request_id': get_request_id(),
        }

        if duration_ms is not None:
            extra['duration_ms'] = duration_ms

        if data:
            extra['data'] = data

        # Add any additional kwargs
        extra.update(kwargs)

        self.logger.log(level, message, extra=extra)

    # Convenience methods for different log levels

    def debug_event(
        self,
        event: str,
        message: str,
        duration_ms: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log DEBUG level event"""
        self._log_event(logging.DEBUG, event, message, duration_ms, data, **kwargs)

    def info_event(
        self,
        event: str,
        message: str,
        duration_ms: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log INFO level event"""
        self._log_event(logging.INFO, event, message, duration_ms, data, **kwargs)

    def warning_event(
        self,
        event: str,
        message: str,
        duration_ms: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log WARNING level event"""
        self._log_event(logging.WARNING, event, message, duration_ms, data, **kwargs)

    def error_event(
        self,
        event: str,
        message: str,
        duration_ms: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
        **kwargs
    ):
        """Log ERROR level event"""
        extra = {
            'event': event,
            'request_id': get_request_id(),
        }

        if duration_ms is not None:
            extra['duration_ms'] = duration_ms

        if data:
            extra['data'] = data

        extra.update(kwargs)

        self.logger.error(message, extra=extra, exc_info=exc_info)

    # Performance tracking helper

    def time_function(self, event: str, level: int = logging.INFO):
        """
        Decorator to time a function and log performance

        Usage:
            @logger.time_function("search_operation")
            def my_search():
                ...
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    self._log_event(
                        level,
                        event,
                        f"{func.__name__} completed",
                        duration_ms=duration_ms
                    )
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    self.error_event(
                        f"{event}_error",
                        f"{func.__name__} failed: {str(e)}",
                        duration_ms=duration_ms
                    )
                    raise
            return wrapper
        return decorator

    # Metrics logging

    def log_metrics(self, metrics: Dict[str, float], event: str = "performance_metrics"):
        """
        Log performance metrics

        Usage:
            logger.log_metrics({
                "total_duration_ms": 1234.5,
                "search_duration_ms": 234.5,
                "rerank_duration_ms": 456.7
            })
        """
        self.info_event(
            event,
            "Performance metrics",
            data={"metrics": metrics}
        )

    # Standard logging methods (backward compatibility)

    def debug(self, message: str, **kwargs):
        """Standard DEBUG log"""
        self.logger.debug(message, extra={'request_id': get_request_id(), **kwargs})

    def info(self, message: str, **kwargs):
        """Standard INFO log"""
        self.logger.info(message, extra={'request_id': get_request_id(), **kwargs})

    def warning(self, message: str, **kwargs):
        """Standard WARNING log"""
        self.logger.warning(message, extra={'request_id': get_request_id(), **kwargs})

    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Standard ERROR log"""
        self.logger.error(message, extra={'request_id': get_request_id(), **kwargs}, exc_info=exc_info)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance

    Usage:
        logger = get_logger(__name__)
        logger.info_event("request_received", "New request received", data={"user_id": 123})
    """
    return StructuredLogger(name)
