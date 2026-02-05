# -*- coding: utf-8 -*-
"""
Request Logging Middleware

Logs all POST requests with details like path, query params, and processing time.
"""
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.structured_logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all POST requests

    Logs:
    - Request method and path
    - Query parameters
    - Processing time
    - Response status code
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only log POST requests
        if request.method != "POST":
            return await call_next(request)

        # Record start time
        start_time = time.time()

        # Extract request details
        path = request.url.path
        method = request.method
        query_params = dict(request.query_params) if request.query_params else {}
        content_type = request.headers.get("content-type", "")

        # Log the incoming request
        logger.info_event(
            "post_request",
            f"POST {path}",
            data={
                "method": method,
                "path": path,
                "query_params": query_params,
                "content_type": content_type,
            }
        )

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log the response
        logger.info_event(
            "post_response",
            f"POST {path} completed",
            data={
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
            }
        )

        # Add processing time to response headers
        response.headers["X-Process-Time"] = str(process_time)

        return response
