"""
Document processing services

This package contains services for parsing and processing documents.
"""

# complete_chunker를 기본 파서로 사용 (parser.py 대신)
from .parser import (
    IntegratedParserConfig,
    OCRDetectionResult,
    process_pdf_to_chunks,
)

__all__ = [
    "IntegratedParserConfig",
    "OCRDetectionResult",
    "process_pdf_to_chunks",
]
