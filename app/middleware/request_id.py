# -*- coding: utf-8 -*-
"""
Request ID Middleware

Generates a unique request ID for each HTTP request and stores it in context.
The request ID is included in all logs and can be returned in response headers.
"""
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging_config import set_request_id


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to generate and track request IDs

    - Generates a UUID for each request
    - Stores it in context for logging
    - Adds it to response headers (X-Request-ID)
    """

    async def dispatch(self, request: Request, call_next):
        # Generate request ID (or use existing from header)
        request_id = request.headers.get('X-Request-ID')
        if not request_id:
            request_id = str(uuid.uuid4())

        # Set request ID in context
        set_request_id(request_id)

        # Call next middleware/endpoint
        response = await call_next(request)

        # Add request ID to response headers
        if isinstance(response, Response):
            response.headers['X-Request-ID'] = request_id

        return response
