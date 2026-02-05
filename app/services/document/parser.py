# -*- coding: utf-8 -*-
"""
Docling Complete + Chunking 통합 파서

docling_complete.py의 Docling 전처리 + chunk_docling.py의 Dual Content 청킹을
하나로 통합한 원스톱 PDF 처리 모듈입니다.

Features:
- Docling Complete의 모든 기능 (OCR 자동 감지, VLM/LLM description, 이미지/테이블 PNG 저장)
- Dual Content 청킹 (검색용 content + LLM용 content_for_llm)
- 메모리 기반 처리 (파일 I/O 최소화)
- 단일 함수 호출로 PDF → Chunks까지 완료

Usage:
    >>> from complete_chunker import DoclingChunker
    >>>
    >>> # 기본 모드 (PNG 저장만)
    >>> chunker = DoclingChunker()
    >>> chunks, metadata = chunker.process_pdf_to_chunks(pdf_bytes, "test.pdf", output_dir)
    >>>
    >>> # 고급 모드 (VLM/LLM description 추가)
    >>> chunker = DoclingChunker(advanced_mode=True)
    >>> chunks, metadata = chunker.process_pdf_to_chunks(pdf_bytes, "test.pdf", output_dir)
"""
import json
import logging
import re
import io
import base64
import unicodedata
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass
from uuid import uuid5, NAMESPACE_DNS

import fitz  # PyMuPDF
import requests
from PIL import Image
import nltk
from nltk.tokenize import sent_tokenize

# Config import
from app.services.rag.config import RAGConfig

# Note: CUDA_VISIBLE_DEVICES 설정 제거
# VLM/LLM은 vLLM API를 통해 호출되므로 로컬 GPU 설정 불필요
# 전역 CUDA_VISIBLE_DEVICES 설정은 다른 모듈(retriever 등)의 GPU 접근을 방해함

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    EasyOcrOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, PictureItem, TableItem

# NLTK 데이터 다운로드 (최초 1회만 필요)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


@dataclass
class OCRDetectionResult:
    """OCR 감지 결과"""
    needs_ocr: bool
    reason: str
    avg_chars_per_page: float
    has_images: bool
    has_corrupted_text: bool
    recommended_lang: list[str]
    total_pages: int
    text_layer_ratio: float  # 텍스트 레이어가 있는 페이지 비율


# ===== 헬퍼 함수 =====
def _collapse_ws_keep_newlines(text: str) -> str:
    """공백을 압축하되 줄바꿈은 유지"""
    if not text:
        return ""
    lines = text.split('\n')
    collapsed = []
    for line in lines:
        collapsed.append(' '.join(line.split()))
    return '\n'.join(collapsed)


def _docling_table_to_markdown(table_data: Dict[str, Any]) -> str:
    """Docling 테이블 데이터를 Markdown으로 변환"""
    # table_cells는 data 안에 있음
    data = table_data.get("data", {})
    cells = data.get("table_cells")
    if not cells:
        return ""

    data_list = []
    max_row, max_col = 0, 0

    for cell in cells:
        r_start = cell.get("start_row_offset_idx", 0)
        c_start = cell.get("start_col_offset_idx", 0)
        text = (_collapse_ws_keep_newlines(cell.get("text", "")) or "").replace('\n', ' ').strip()

        # 파이프 문자 이스케이프
        text = text.replace("|", "\\|")

        if text:
            data_list.append({"row": r_start, "col": c_start, "text": text})
            max_row = max(max_row, r_start)
            max_col = max(max_col, c_start)

    if not data_list:
        return ""

    table_array = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for item in data_list:
        r, c = item['row'], item['col']
        if 0 <= r <= max_row and 0 <= c <= max_col:
            table_array[r][c] = item['text']

    # Markdown 테이블 생성
    if not table_array:
        return ""

    lines = []
    for idx, row in enumerate(table_array):
        lines.append("| " + " | ".join(row) + " |")
        if idx == 0:
            lines.append("|" + "|".join([" --- " for _ in row]) + "|")

    return "\n".join(lines)


class DoclingChunker:

    """
    Docling Complete + Dual Content 청킹 통합 클래스

    docling_complete.py와 chunk_docling.py의 로직을 하나로 통합하여
    PDF → JSON → Chunks까지 원스톱으로 처리합니다.

    Example:
        >>> # 기본 모드
        >>> chunker = DoclingChunker()
        >>> chunks, metadata = chunker.process_pdf_to_chunks(pdf_bytes, "test.pdf", output_dir)
        >>>
        >>> # 고급 모드 (VLM/LLM description 생성)
        >>> chunker = DoclingChunker(advanced_mode=True)
        >>> chunks, metadata = chunker.process_pdf_to_chunks(pdf_bytes, "test.pdf", output_dir)
    """

    def __init__(
        self,
        # Docling Complete 옵션 (config 기본값 사용)
        image_scale: float = None,
        enable_table_structure: bool = True,
        auto_detect_ocr: bool = True,
        force_ocr: bool = False,
        force_no_ocr: bool = False,
        ocr_engine: str = "tesseract",
        ocr_threshold: float = None,

        # 고급 모드 옵션
        advanced_mode: bool = False,
        enable_image_description: bool = True,
        enable_table_description: bool = True,
        filter_junk_images: bool = True,

        # LLM/VLM 설정 (config 기본값 사용)
        llm_model: str = None,
        vision_model: str = None,
        ollama_url: str = None,

        # 프롬프트
        image_description_prompt: str = """Analyze this image and determine if it's meaningful content or junk.

JUNK images include: QR codes, barcodes, logos, decorative elements, page numbers, headers/footers, book covers, irrelevant graphics.

If the image is JUNK, respond with exactly: "JUNK"

If the image contains meaningful technical/scientific content (diagrams, charts, photos, illustrations), provide a concise description in English focusing on key components and technical details. Limit to 3-5 sentences.""",
        table_description_prompt: str = "Analyze this table (provided in markdown format) and provide a concise summary in English. Explain what data it contains and any key insights. Limit to 3-5 sentences.",

        # 청킹 옵션 (config 기본값 사용)
        max_tokens: int = None,
        min_chunk_tokens: int = None,
        include_descriptions: bool = True,
        embed_with_assets: bool = False,

        # Progress callback
        progress_callback: Optional[callable] = None,
    ):
        """
        Args:
            # Docling Complete 옵션
            image_scale: 이미지 해상도 스케일 (기본: 2.0)
            enable_table_structure: 테이블 구조 분석 활성화 (기본: True)
            auto_detect_ocr: OCR 필요 여부 자동 감지 (기본: True)
            force_ocr: OCR 강제 활성화 (기본: False)
            force_no_ocr: OCR 강제 비활성화 (기본: False)
            ocr_engine: OCR 엔진 선택 - tesseract(한국어 우수), easyocr(다국어), rapidocr(빠름)
            ocr_threshold: OCR 필요 판단 임계값 - 텍스트 레이어 비율 (기본: 0.4 = 40%)

            # 고급 모드
            advanced_mode: 고급 모드 활성화 (기본 모드: 이미지/테이블 PNG 저장만, 고급 모드: VLM/LLM description 추가 생성)
            enable_image_description: 이미지 VLM description 생성 (advanced_mode=True일 때만 동작)
            enable_table_description: 테이블 LLM description 생성 (advanced_mode=True일 때만 동작)
            filter_junk_images: JUNK으로 분류된 이미지 필터링 (기본: True)

            # LLM/VLM 설정
            llm_model: LLM 모델명
            vision_model: VLM vision 모델명
            ollama_url: vLLM 서버 URL

            # 프롬프트
            image_description_prompt: 이미지 description 프롬프트
            table_description_prompt: 테이블 description 프롬프트

            # 청킹 옵션
            max_tokens: 텍스트 청크당 최대 토큰 수 (기본: 400, asset 추가 전)
            min_chunk_tokens: 청크 최소 토큰 수 (기본: 100, 이보다 작으면 이전 청크에 병합)
            include_descriptions: 이미지/테이블 description을 포함할지 여부 (기본: True)
            embed_with_assets: content 필드에도 에셋 설명을 appendix로 추가 (기본: False)
        """
        # Docling 옵션 (config 기본값 사용)
        self.image_scale = image_scale if image_scale is not None else RAGConfig.DOCLING_IMAGE_SCALE
        self.enable_table_structure = enable_table_structure
        self.auto_detect_ocr = auto_detect_ocr
        self.force_ocr = force_ocr
        self.force_no_ocr = force_no_ocr
        self.ocr_engine = ocr_engine.lower()
        self.ocr_threshold = ocr_threshold if ocr_threshold is not None else RAGConfig.DOCLING_OCR_THRESHOLD

        # 고급 모드 옵션
        self.advanced_mode = advanced_mode
        self.enable_image_description = enable_image_description and advanced_mode
        self.enable_table_description = enable_table_description and advanced_mode
        self.filter_junk_images = True if advanced_mode else filter_junk_images

        # LLM/VLM 설정 (config 기본값 사용)
        self.llm_model = llm_model if llm_model is not None else RAGConfig.DOCLING_LLM_MODEL
        self.vision_model = vision_model if vision_model is not None else RAGConfig.DOCLING_VISION_MODEL
        # VLM URL (이미지 description용, Port 8002)
        self.vlm_url = ollama_url if ollama_url is not None else RAGConfig.VLM_URL
        # LLM URL (테이블 description용, Port 8003)
        self.llm_url = RAGConfig.FOLLOW_UP_LLM_URL

        # 프롬프트
        self.image_description_prompt = image_description_prompt
        self.table_description_prompt = table_description_prompt

        # 청킹 옵션 (config 기본값 사용)
        self.max_tokens = max_tokens if max_tokens is not None else RAGConfig.CHUNK_MAX_TOKENS
        self.min_chunk_tokens = max(0, min_chunk_tokens if min_chunk_tokens is not None else RAGConfig.CHUNK_MIN_TOKENS)
        self.include_descriptions = include_descriptions
        self.embed_with_assets = embed_with_assets

        # Progress callback
        self.progress_callback = progress_callback

        self.logger = logging.getLogger(__name__)

        # 고급 모드 설정 로깅
        self.logger.info(
            f"DoclingChunker 설정 완료: advanced_mode={self.advanced_mode}, "
            f"enable_image_description={self.enable_image_description}, "
            f"enable_table_description={self.enable_table_description}, "
            f"vlm_url={self.vlm_url}, llm_url={self.llm_url}"
        )

        # OCR 결과 저장용
        self.last_ocr_result = None

        # 상호 배타적 옵션 검증
        if self.force_ocr and self.force_no_ocr:
            raise ValueError("force_ocr와 force_no_ocr는 동시에 True일 수 없습니다.")

        # 제외할 label 목록
        self.exclude_labels = {
            'page_header',
            'page_footer',
        }

        # 정규식 패턴
        self.ref_re = re.compile(r"^#/(texts|groups|tables|pictures)/(\d+)$", re.I)

    # ===== Docling Complete 메서드 =====

    def _generate_vlm_description(self, image: Image.Image) -> Optional[str]:
        """vLLM VLM을 사용하여 이미지 description 생성 (Port 8002: Qwen VLM)"""
        try:
            self.logger.info(f"VLM description 생성 시작: url={self.vlm_url}, model={self.vision_model}")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            response = requests.post(
                f"{self.vlm_url}/v1/chat/completions",
                json={
                    "model": self.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.image_description_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
                timeout=60
            )

            if response.status_code == 200:
                desc = response.json()["choices"][0]["message"]["content"].strip()
                self.logger.debug(f"VLM description 생성 성공 ({len(desc)} chars)")
                return desc
            else:
                self.logger.warning(f"VLM API error: {response.status_code}")
                return None
        except Exception as e:
            self.logger.warning(f"VLM description 실패: {e}")
            return None

    def _generate_llm_table_description(self, markdown_table: str) -> Optional[str]:
        """LLM을 사용하여 테이블 마크다운 요약 생성"""
        if not markdown_table or len(markdown_table.strip()) == 0:
            return None

        try:
            prompt = f"{self.table_description_prompt}\n\n{markdown_table}"
            return self._call_vllm_text(prompt)

        except Exception as e:
            self.logger.warning(f"LLM table description 실패: {e}")
            return None

    def _call_vllm_text(self, prompt: str) -> Optional[str]:
        """vLLM 텍스트 모델 호출 (Port 8003: GPT-OSS-20B)"""
        try:
            self.logger.info(f"LLM text 생성 시작: url={self.llm_url}, model={self.llm_model}")
            response = requests.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                self.logger.error(f"vLLM API error: {response.status_code}")
                return None

        except Exception as e:  
            self.logger.error(f"vLLM API 호출 실패: {e}")
            return None

    def detect_ocr_requirement(self, pdf_path: Path) -> OCRDetectionResult:
        """PDF 구조를 분석하여 OCR 필요 여부를 판단"""
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        total_chars = 0
        has_images = False
        has_corrupted = False
        sample_text = ""

        pages_with_text_layer = 0

        for page in doc:
            text = page.get_text("text")
            total_chars += len(text.strip())
            sample_text += text[:1000]

            # PDF 구조 분석
            text_dict = page.get_text("dict")
            blocks = text_dict.get("blocks", [])

            has_text_block = False
            has_image_block = False

            for block in blocks:
                block_type = block.get("type", -1)
                if block_type == 0:  # 텍스트 블록
                    lines = block.get("lines", [])
                    for line in lines:
                        spans = line.get("spans", [])
                        for span in spans:
                            span_text = span.get("text", "").strip()
                            if span_text:
                                has_text_block = True
                                break
                elif block_type == 1:  # 이미지 블록
                    has_image_block = True

            if has_text_block:
                pages_with_text_layer += 1
            if has_image_block:
                has_images = True

            # 깨진 문자 감지
            if "GLYPH<" in text:
                has_corrupted = True
            if re.search(r'[\u0300-\u036f]{3,}', text):
                has_corrupted = True
            if re.search(r'[\ufffd]{2,}', text):
                has_corrupted = True

        doc.close()

        avg_chars = total_chars / total_pages if total_pages > 0 else 0
        text_layer_ratio = pages_with_text_layer / total_pages if total_pages > 0 else 0

        # 언어 감지
        recommended_lang = ["en"]
        if re.search(r'[\uac00-\ud7af]', sample_text):  # 한국어
            recommended_lang = ["ko"]
        elif re.search(r'[\u4e00-\u9fff]', sample_text):  # 중국어
            recommended_lang = ["ch"]
        elif re.search(r'[\u3040-\u30ff]', sample_text):  # 일본어
            recommended_lang = ["ja"]

        # OCR 필요 여부 판단
        needs_ocr = False
        reason = "텍스트 추출 정상"

        if has_corrupted:
            needs_ocr = True
            reason = "깨진 문자/GLYPH 태그 감지 - OCR 필요"
        elif text_layer_ratio < self.ocr_threshold:
            needs_ocr = True
            reason = f"텍스트 레이어 부족 ({pages_with_text_layer}/{total_pages} 페이지, {text_layer_ratio:.0%} < {self.ocr_threshold:.0%})"
        else:
            needs_ocr = False
            reason = f"텍스트 레이어 충분 ({pages_with_text_layer}/{total_pages} 페이지, {text_layer_ratio:.0%} >= {self.ocr_threshold:.0%})"

        return OCRDetectionResult(
            needs_ocr=needs_ocr,
            reason=reason,
            avg_chars_per_page=avg_chars,
            has_images=has_images,
            has_corrupted_text=has_corrupted,
            recommended_lang=recommended_lang,
            total_pages=total_pages,
            text_layer_ratio=text_layer_ratio,
        )

    def _create_converter(self, ocr_result: Optional[OCRDetectionResult] = None) -> DocumentConverter:
        """DocumentConverter 인스턴스 생성"""
        pdf_opts = PdfPipelineOptions()
        pdf_opts.images_scale = self.image_scale
        pdf_opts.generate_page_images = True
        pdf_opts.generate_picture_images = True
        pdf_opts.do_table_structure = self.enable_table_structure

        # OCR 활성화 여부 결정
        enable_ocr = False
        if self.force_no_ocr:
            enable_ocr = False
            self.logger.info("OCR 강제 비활성화")
        elif self.force_ocr:
            enable_ocr = True
            self.logger.info("OCR 강제 활성화")
        elif self.auto_detect_ocr and ocr_result:
            enable_ocr = ocr_result.needs_ocr
            self.logger.info(f"OCR 자동 감지: {ocr_result.reason}")

        pdf_opts.do_ocr = enable_ocr

        # OCR 활성화 시 엔진 선택
        if enable_ocr and ocr_result:
            lang = ocr_result.recommended_lang[0] if ocr_result.recommended_lang else "en"

            if self.ocr_engine == "tesseract":
                tesseract_lang_map = {
                    "ko": ["kor", "eng"],
                    "ja": ["jpn", "eng"],
                    "ch": ["chi_sim", "chi_tra", "eng"],
                    "en": ["eng"],
                }
                tesseract_lang = tesseract_lang_map.get(lang, ["eng"])
                ocr_options = TesseractCliOcrOptions(
                    force_full_page_ocr=True,
                    lang=tesseract_lang,
                )
                self.logger.info(f"Tesseract OCR 활성화 (언어: {tesseract_lang})")
            elif self.ocr_engine == "easyocr":
                easyocr_lang_map = {
                    "ko": ["ko", "en"],
                    "ja": ["ja", "en"],
                    "ch": ["ch_sim", "en"],
                    "en": ["en"],
                }
                ocr_options = EasyOcrOptions(
                    force_full_page_ocr=True,
                    lang=easyocr_lang_map.get(lang, ["en"]),
                )
                self.logger.info(f"EasyOCR 활성화 (언어: {easyocr_lang_map.get(lang, ['en'])})")
            else:
                ocr_options = RapidOcrOptions(
                    force_full_page_ocr=True,
                    lang=ocr_result.recommended_lang,
                )
                self.logger.info(f"RapidOCR 활성화 (언어: {ocr_result.recommended_lang})")

            pdf_opts.ocr_options = ocr_options

        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        )

    def convert_to_dict(self, pdf_path: Path, assets_dir: Path) -> dict:
        """
        PDF 파일을 DoclingDocument로 변환 후 딕셔너리 반환

        Args:
            pdf_path: 변환할 PDF 파일 경로
            assets_dir: 에셋 저장 디렉토리

        Returns:
            변환된 딕셔너리
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"PDF 파일이 아닙니다: {pdf_path}")

        self.logger.info(f"PDF 변환 시작: {pdf_path.name}")

        # OCR 감지
        ocr_result = None
        if self.auto_detect_ocr and not self.force_ocr and not self.force_no_ocr:
            self.logger.info("OCR 필요 여부 분석 중...")
            ocr_result = self.detect_ocr_requirement(pdf_path)
            self.logger.info(f"  - 총 페이지: {ocr_result.total_pages}")
            self.logger.info(f"  - 텍스트 레이어 비율: {ocr_result.text_layer_ratio:.1%}")
            self.logger.info(f"  - 평균 문자/페이지: {ocr_result.avg_chars_per_page:.0f}")
            self.logger.info(f"  - 깨진 문자 감지: {ocr_result.has_corrupted_text}")
            self.logger.info(f"  - 권장 언어: {ocr_result.recommended_lang}")
            self.logger.info(f"  - 결정: {ocr_result.reason}")

        # OCR 결과 저장 (metadata에 포함시키기 위해)
        self.last_ocr_result = ocr_result

        # 변환 실행
        converter = self._create_converter(ocr_result)
        result = converter.convert(pdf_path)
        doc = result.document

        self.logger.info(f"PDF 변환 완료: {pdf_path.name}")

        # 첫 번째 패스: PictureItem에서 classification 추출
        classification_map = {}
        img_idx_temp = 0
        for item, _level in doc.iterate_items():
            if isinstance(item, PictureItem):
                for annot in item.annotations:
                    if hasattr(annot, 'predicted_classes') and annot.predicted_classes:
                        best_class = max(annot.predicted_classes, key=lambda x: x.confidence)
                        classification_map[img_idx_temp] = best_class.class_name
                        self.logger.debug(f"이미지 {img_idx_temp} classification: {best_class.class_name} (confidence: {best_class.confidence:.3f})")
                        break
                img_idx_temp += 1

        # DoclingDocument를 딕셔너리로 변환
        doc_dict = doc.export_to_dict()

        # classification을 doc_dict에 추가
        pictures = doc_dict.get("pictures", [])
        for idx, classification in classification_map.items():
            if idx < len(pictures):
                pictures[idx]["classification"] = classification

        if classification_map:
            self.logger.info(f"📝 Classification 추출 완료: {len(classification_map)}개 이미지")

        # 에셋 디렉토리 설정
        images_dir = assets_dir / "images"
        tables_dir = assets_dir / "tables"
        images_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"에셋 저장 디렉토리: {assets_dir}")

        # 이미지 및 테이블 처리
        img_idx = 0
        tbl_idx = 0

        # JUNK 이미지 인덱스 수집
        junk_image_indices: set[int] = set()
        if self.filter_junk_images:
            junk_image_indices = {
                i for i, pic in enumerate(doc_dict.get("pictures", []))
                if isinstance(pic, dict) and pic.get("classification") == "JUNK"
            }
            if junk_image_indices:
                self.logger.info(f"🗑️  JUNK 이미지 {len(junk_image_indices)}개 필터링")

        # 고급 모드 카운터
        total_tables = len(doc_dict.get("tables", []))
        total_images = len(doc_dict.get("pictures", []))

        if self.advanced_mode:
            self.logger.info(f"📊 고급 모드: 테이블 {total_tables}개, 이미지 {total_images - len(junk_image_indices)}개 처리 시작")
            table_desc_count = 0
            image_desc_count = 0
            junk_skipped = 0

        for item, _level in doc.iterate_items():
            # 테이블 처리
            if isinstance(item, TableItem):
                try:
                    # 테이블 이미지 저장
                    img = item.get_image(doc)
                    table_path = tables_dir / f"table-{tbl_idx}.png"
                    img.save(table_path, "PNG")
                    self.logger.debug(f"테이블 저장: {table_path}")

                    # 고급 모드: LLM description 생성
                    if self.enable_table_description and "tables" in doc_dict:
                        if tbl_idx < len(doc_dict["tables"]):
                            table_node = doc_dict["tables"][tbl_idx]

                            # 마크다운 변환
                            markdown_table = _docling_table_to_markdown(table_node)
                            if markdown_table:
                                self.logger.info(f"🔄 테이블 description 생성 중 [{tbl_idx + 1}/{total_tables}]...")

                                # Progress callback 호출
                                if self.progress_callback:
                                    self.progress_callback({
                                        'status': f'Generating table description [{tbl_idx + 1}/{total_tables}]',
                                        'progress': 40 + int((tbl_idx / total_tables) * 20),
                                        'current_table': tbl_idx + 1,
                                        'total_tables': total_tables,
                                    })

                                # LLM 요약 생성
                                desc = self._generate_llm_table_description(markdown_table)
                                if desc:
                                    if "annotations" not in table_node:
                                        table_node["annotations"] = []
                                    table_node["annotations"].append({
                                        "kind": "description",
                                        "label": "llm_table_summary",
                                        "text": desc
                                    })
                                    table_desc_count += 1
                                    self.logger.info(f"✅ 테이블 [{tbl_idx + 1}/{total_tables}] description 완료 (총 {table_desc_count}개 생성)")

                except Exception as e:
                    self.logger.warning(f"❌ 테이블 {tbl_idx} 처리 실패: {e}")

                tbl_idx += 1

            # 이미지 처리
            elif isinstance(item, PictureItem):
                try:
                    # JUNK 이미지 필터링
                    if img_idx in junk_image_indices:
                        self.logger.debug(f"⏭️  이미지 {img_idx} JUNK으로 스킵")
                        if self.advanced_mode:
                            junk_skipped += 1
                        img_idx += 1
                        continue

                    # 이미지 저장
                    img = item.get_image(doc)
                    image_path = images_dir / f"image-{img_idx}.png"
                    img.save(image_path, "PNG")
                    self.logger.debug(f"이미지 저장: {image_path}")

                    # 고급 모드: VLM description 생성
                    if self.enable_image_description and "pictures" in doc_dict:
                        if img_idx < len(doc_dict["pictures"]):
                            picture_node = doc_dict["pictures"][img_idx]

                            self.logger.info(f"🔄 이미지 description 생성 중 [{img_idx + 1 - junk_skipped}/{total_images - len(junk_image_indices)}]...")

                            # Progress callback 호출
                            if self.progress_callback:
                                self.progress_callback({
                                    'status': f'Generating image description [{img_idx + 1 - junk_skipped}/{total_images - len(junk_image_indices)}]',
                                    'progress': 40 + int((img_idx / total_images) * 20),
                                    'current_image': img_idx + 1 - junk_skipped,
                                    'total_images': total_images - len(junk_image_indices),
                                })

                            desc = self._generate_vlm_description(img)
                            if desc:
                                if desc.strip().upper() == "JUNK":
                                    picture_node["classification"] = "JUNK"
                                    self.logger.info(f"🗑️  이미지 [{img_idx + 1 - junk_skipped}] VLM이 JUNK로 분류")
                                else:
                                    if "annotations" not in picture_node:
                                        picture_node["annotations"] = []
                                    picture_node["annotations"].append({
                                        "kind": "description",
                                        "label": "vlm_image_description",
                                        "text": desc
                                    })
                                    image_desc_count += 1
                                    self.logger.info(f"✅ 이미지 [{img_idx + 1 - junk_skipped}/{total_images - len(junk_image_indices)}] description 완료 (총 {image_desc_count}개 생성)")

                except Exception as e:
                    self.logger.warning(f"❌ 이미지 {img_idx} 처리 실패: {e}")

                img_idx += 1

        # 고급 모드 최종 결과
        if self.advanced_mode:
            self.logger.info(f"🎉 고급 모드 처리 완료:")
            if self.enable_table_description:
                self.logger.info(f"   - 테이블 description: {table_desc_count}/{total_tables}개 생성")
            if self.enable_image_description:
                self.logger.info(f"   - 이미지 description: {image_desc_count}/{total_images}개 생성")

        return doc_dict

    # ===== Dual Content 청킹 메서드 =====

    def _generate_chunk_id(self, source_file: str, chunk_index: int, content: str) -> str:
        """Deterministic UUID 생성"""
        content_prefix = content[:100] if content else ""
        unique_str = f"{source_file}|{chunk_index}|{content_prefix}"
        chunk_uuid = uuid5(NAMESPACE_DNS, unique_str)
        return str(chunk_uuid)

    def _generate_section_id(self, source_file: str, section_header: str) -> str:
        """Deterministic Section ID 생성"""
        unique_str = f"{source_file}|section|{section_header}"
        section_uuid = uuid5(NAMESPACE_DNS, unique_str)
        return str(section_uuid)

    def _estimate_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 추정"""
        words = text.split()
        chars = len(text)
        return int(max(len(words) * 1.3, chars * 0.8))

    def _split_into_sentences(self, text: str) -> list[str]:
        """텍스트를 문장 단위로 분할"""
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            self.logger.warning(f"NLTK sent_tokenize 실패, 정규표현식으로 대체: {e}")
            sentences = re.split(r'(?<=[.!?\n])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def _get_item_by_ref(self, ref: str, data: dict) -> dict | None:
        """$ref 문자열로 실제 항목 가져오기"""
        if not ref or not ref.startswith('#/'):
            return None

        parts = ref.strip('#/').split('/')
        if len(parts) != 2:
            return None

        collection, idx = parts
        try:
            return data.get(collection, [])[int(idx)]
        except (IndexError, ValueError):
            return None

    def _build_asset_captions(self, data: dict) -> dict[str, list[dict[str, Any]]]:
        """에셋(테이블/그림)의 캡션을 수집"""
        caps: dict[str, list[dict[str, Any]]] = {}
        for n in data.get("texts", []):
            if (n.get("label") or "").strip().lower() != "caption":
                continue
            parent = n.get("parent") or {}
            cref = parent.get("cref") or parent.get("$ref")
            if not isinstance(cref, str) or not self.ref_re.match(cref):
                continue
            txt = (n.get("text") or n.get("orig") or "").strip()
            if not txt:
                continue
            pg = (n.get("prov") or [{}])[0].get("page_no")
            caps.setdefault(cref, []).append({"text": txt, "page_no": pg})
        return caps

    def _get_page_no(self, node: dict) -> int:
        """노드에서 페이지 번호 추출"""
        prov = node.get("prov") or []
        if prov and isinstance(prov[0], dict):
            p = prov[0].get("page_no")
            if isinstance(p, int):
                return p
        return 1

    def _create_asset_summary(self, tables: list, pictures: list, formulas: list) -> dict[str, Any]:
        """에셋 요약 정보 생성"""
        table_count = len(tables)
        picture_count = len(pictures)
        formula_count = len(formulas)
        total_count = table_count + picture_count + formula_count

        return {
            "total_count": total_count,
            "has_tables": table_count > 0,
            "has_pictures": picture_count > 0,
            "has_formulas": formula_count > 0,
            "table_count": table_count,
            "picture_count": picture_count,
            "formula_count": formula_count,
        }

    def _build_asset_metadata(self, data: dict, captions_by_ref: dict) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        """모든 에셋 메타데이터 수집 + 순서 정보"""
        all_assets: dict[str, dict[str, Any]] = {}
        asset_order: dict[str, int] = {}
        order_counter = 0

        # Tombstone 및 JUNK 이미지 인덱스 수집
        deleted_pidx: set[int] = {
            i for i, pic in enumerate(data.get("pictures", []))
            if isinstance(pic, dict) and pic.get("deleted") is True
        }

        junk_pidx: set[int] = set()
        if self.filter_junk_images:
            junk_pidx = {
                i for i, pic in enumerate(data.get("pictures", []))
                if isinstance(pic, dict) and pic.get("classification") == "JUNK"
            }

        # body children 순회하여 순서 정보 수집
        body = data.get('body', {})
        children = body.get('children', [])

        for child_ref_obj in children:
            ref = child_ref_obj.get('$ref', '')
            if not ref:
                continue

            # 테이블 또는 이미지
            if ref.startswith('#/tables/') or ref.startswith('#/pictures/'):
                # 이미지 필터링 체크
                if ref.startswith('#/pictures/'):
                    idx = int(ref.split('/')[-1])
                    if idx in deleted_pidx or idx in junk_pidx:
                        continue

                asset_order[ref] = order_counter
                order_counter += 1

            # Formula 체크
            elif ref.startswith('#/texts/'):
                item = self._get_item_by_ref(ref, data)
                if item and item.get('label') == 'formula':
                    asset_order[ref] = order_counter
                    order_counter += 1

        # 테이블 메타데이터
        for idx, table_data in enumerate(data.get('tables', [])):
            ref = f"#/tables/{idx}"
            if ref not in asset_order:
                continue

            prov = table_data.get("prov", [])
            page_no = prov[0].get("page_no", 1) if prov and isinstance(prov[0], dict) else 1

            asset_entry = {
                "uid": table_data.get("self_ref") or table_data.get("uid") or table_data.get("id"),
                "ref_norm": ref,
                "page_no": page_no,
                "captions": captions_by_ref.get(ref, []),
                "_type": "tables",
                "_order": asset_order[ref],
            }

            # 테이블 Markdown 변환
            tbl_data = table_data.get("data")
            if tbl_data:
                md_table = _docling_table_to_markdown(tbl_data)
                if md_table:
                    asset_entry["markdown_table"] = md_table

            # 테이블 description
            if self.include_descriptions:
                annotations = table_data.get("annotations", [])
                for annot in annotations:
                    if isinstance(annot, dict) and annot.get("label") == "llm_table_summary":
                        description = annot.get("text", "").strip()
                        if description:
                            asset_entry["description"] = description
                            break

            all_assets[ref] = asset_entry

        # 이미지 메타데이터
        for idx, picture_data in enumerate(data.get('pictures', [])):
            ref = f"#/pictures/{idx}"
            if ref not in asset_order:
                continue

            # Tombstone 및 JUNK 제외
            if idx in deleted_pidx or idx in junk_pidx:
                continue
            if picture_data.get("deleted") is True:
                continue

            prov = picture_data.get("prov", [])
            page_no = prov[0].get("page_no", 1) if prov and isinstance(prov[0], dict) else 1

            asset_entry = {
                "uid": picture_data.get("self_ref") or picture_data.get("uid") or picture_data.get("id"),
                "ref_norm": ref,
                "page_no": page_no,
                "captions": captions_by_ref.get(ref, []),
                "classification": picture_data.get("classification"),
                "_type": "pictures",
                "_order": asset_order[ref],
            }

            # 이미지 description
            if self.include_descriptions:
                annotations = picture_data.get("annotations", [])
                for annot in annotations:
                    if isinstance(annot, dict) and annot.get("label") == "vlm_image_description":
                        description = annot.get("text", "").strip()
                        if description:
                            asset_entry["description"] = description
                            break

            all_assets[ref] = asset_entry

        # Formula 메타데이터
        for idx, text_data in enumerate(data.get('texts', [])):
            if text_data.get('label') != 'formula':
                continue

            ref = f"#/texts/{idx}"
            if ref not in asset_order:
                continue

            formula_text = text_data.get('orig') or text_data.get('text', '')
            if not formula_text:
                continue

            prov = text_data.get("prov", [])
            page_no = prov[0].get("page_no", 1) if prov and isinstance(prov[0], dict) else 1

            asset_entry = {
                "uid": text_data.get("self_ref") or text_data.get("uid") or text_data.get("id"),
                "ref_norm": ref,
                "page_no": page_no,
                "formula": formula_text.strip(),
                "_type": "formulas",
                "_order": asset_order[ref],
            }

            all_assets[ref] = asset_entry

        return all_assets, asset_order

    def _chunk_text_only(self, data: dict, all_assets: dict) -> list[dict]:
        """
        Step 1: 텍스트와 에셋을 inline으로 청킹

        Returns:
            텍스트 + inline 에셋 포함된 청크 리스트
        """
        chunks = []
        current_chunk = {
            "section_header": "",
            "content": "",  # 순수 텍스트만
            "content_with_asset": "",  # 텍스트 + inline 에셋
            "content_token_count": 0,
            "pages": set(),
            "asset_refs": set(),
        }

        section_chunk_indices: dict[str, int] = {}

        # pending_assets: 다음 텍스트 앞에 삽입될 에셋들
        # (ref, page_no, content_text, llm_text): content용 텍스트와 LLM용 텍스트 분리
        pending_assets: list[tuple[str, int, str, str]] = []

        # body children 순회 (groups를 재귀적으로 처리하기 위해 deque 사용)
        from collections import deque

        body = data.get('body', {})
        children = body.get('children', [])
        children_queue = deque(children)

        while children_queue:
            child_ref = children_queue.popleft()
            ref = child_ref.get('$ref', '')
            item = self._get_item_by_ref(ref, data)

            if not item:
                continue

            label = item.get('label', '')
            page_no = self._get_page_no(item)

            # 제외할 항목
            if label in self.exclude_labels:
                continue

            # 테이블 처리
            if ref.startswith('#/tables/'):
                if ref in all_assets:
                    asset = all_assets[ref]
                    table_idx = ref.split('/')[-1]
                    captions = asset.get("captions", [])
                    description = asset.get("description", "")
                    markdown_table = asset.get("markdown_table", "")
                    
                    # content용: Caption 우선, 없으면 Description
                    content_text = ""
                    if captions:
                        content_text = f"[TABLE:table-{table_idx}] {captions[0]['text']}"
                    elif self.include_descriptions and description:
                        content_text = f"[TABLE:table-{table_idx}] {description}"
                    
                    # content_for_llm용: Caption + Markdown Table + Description 모두 포함
                    llm_text_parts = []
                    if captions:
                        llm_text_parts.append(f"[TABLE:table-{table_idx}] {captions[0]['text']}")
                    if markdown_table:
                        llm_text_parts.append(f"```markdown\n{markdown_table}\n```")
                    if self.include_descriptions and description:
                        llm_text_parts.append(f"[TABLE Description: {description}]")
                    llm_text = "\n\n".join(llm_text_parts) if llm_text_parts else ""
                    
                    pending_assets.append((ref, page_no, content_text, llm_text))
                continue

            # 이미지 처리
            if ref.startswith('#/pictures/'):
                if ref in all_assets:
                    asset = all_assets[ref]
                    image_idx = ref.split('/')[-1]
                    captions = asset.get("captions", [])
                    description = asset.get("description", "")
                    
                    # content용: Caption 우선, 없으면 Description
                    content_text = ""
                    if captions:
                        content_text = f"[IMAGE:image-{image_idx}] {captions[0]['text']}"
                    elif self.include_descriptions and description:
                        content_text = f"[IMAGE:image-{image_idx}] {description}"
                    
                    # content_for_llm용: Caption + Description 모두 포함
                    llm_text_parts = []
                    if captions:
                        llm_text_parts.append(f"[IMAGE:image-{image_idx}] {captions[0]['text']}")
                    if self.include_descriptions and description:
                        llm_text_parts.append(f"[IMAGE Description: {description}]")
                    llm_text = "\n".join(llm_text_parts) if llm_text_parts else ""
                    
                    pending_assets.append((ref, page_no, content_text, llm_text))
                continue

            # Formula 처리
            if label == 'formula':
                if ref in all_assets:
                    asset = all_assets[ref]
                    formula_text = asset.get("formula", "")
                    if formula_text:
                        formula_idx = ref.split('/')[-1]
                        asset_text = f"[FORMULA:formula-{formula_idx}] {formula_text}"
                        # Formula는 content와 content_for_llm 동일
                        pending_assets.append((ref, page_no, asset_text, asset_text))
                    else:
                        pending_assets.append((ref, page_no, "", ""))
                continue

            # Groups 처리 (list 등)
            # groups의 children을 재귀적으로 처리 (순서 유지하며 큐 앞에 삽입)
            if ref.startswith('#/groups/'):
                group_children = item.get('children', [])
                # 순서를 유지하며 큐 앞에 삽입 (현재 위치 바로 다음에 처리)
                children_queue.extendleft(reversed(group_children))
                continue

            # 텍스트 노드 처리
            # text: 정규화된 텍스트 (Docling이 처리, 줄바꿈 제거됨)
            # orig: 원본 텍스트 (줄바꿈 포함)
            text = item.get('text', '').strip()  # content용: 정돈된 텍스트
            text_raw_orig = item.get('orig', item.get('text', ''))  # 원본 텍스트 (fallback to text)
            text_for_llm = text_raw_orig.lstrip() if text_raw_orig else text  # content_for_llm용: 원본 줄바꿈 보존

            if not text:
                continue

            # pending_assets를 현재 텍스트 앞에 추가
            for asset_ref, asset_page, content_text, llm_text in pending_assets:
                current_chunk["asset_refs"].add(asset_ref)
                current_chunk["pages"].add(asset_page)

                # content에는 간결한 텍스트만 (caption 우선)
                if content_text:
                    current_chunk['content'] += content_text + "\n\n"
                
                # content_with_asset (LLM용)에는 상세 텍스트 (caption + description)
                if llm_text:
                    current_chunk['content_with_asset'] += llm_text + "\n\n"

            # pending_assets 초기화
            pending_assets = []

            estimated_tokens = self._estimate_tokens(text)

            # section_header: 새로운 섹션 시작
            if label == 'section_header':
                # 이전 청크 저장
                if current_chunk['content'].strip():
                    self._save_text_chunk(chunks, current_chunk, section_chunk_indices)
                    current_chunk = {
                        "section_header": text,
                        "content": "",
                        "content_with_asset": "",
                        "content_token_count": 0,
                        "pages": set(),
                        "asset_refs": set(),
                    }
                else:
                    # 내용이 없으면 헤더 연결
                    if current_chunk['section_header']:
                        current_chunk['section_header'] += " > " + text
                    else:
                        current_chunk['section_header'] = text

            # 일반 텍스트 추가
            else:
                # 단일 텍스트 노드가 max_tokens 초과: 문장 단위 분할
                if estimated_tokens > self.max_tokens:
                    if current_chunk['content'].strip():
                        self._save_text_chunk(chunks, current_chunk, section_chunk_indices)
                        current_chunk = {
                            "section_header": current_chunk['section_header'],
                            "content": "",
                            "content_with_asset": "",
                            "content_token_count": 0,
                            "pages": set(),
                            "asset_refs": set(),
                        }

                    # 문장 단위로 분할
                    # 문장 단위로 분할
                    sentences = self._split_into_sentences(text)
                    for sentence in sentences:
                        sentence_tokens = self._estimate_tokens(sentence)

                        # 문장 하나가 max_tokens 초과하면 그대로 저장
                        if sentence_tokens > self.max_tokens:
                            if current_chunk['content'].strip():
                                self._save_text_chunk(chunks, current_chunk, section_chunk_indices)
                                current_chunk = {
                                    "section_header": current_chunk['section_header'],
                                    "content": "",
                                    "content_with_asset": "",
                                    "content_token_count": 0,
                                    "pages": set(),
                                    "asset_refs": set(),
                                }

                            temp_chunk = {
                                "section_header": current_chunk['section_header'],
                                "content": sentence,  # content: 정돈된 형식
                                "content_with_asset": sentence,  # 문장 분할은 이미 정돈된 텍스트 사용
                                "content_token_count": sentence_tokens,
                                "pages": {page_no},
                                "asset_refs": set(),
                            }
                            self._save_text_chunk(chunks, temp_chunk, section_chunk_indices)
                            continue

                        # 현재 청크에 추가하면 max_tokens 초과
                        if current_chunk['content'].strip() and current_chunk['content_token_count'] + sentence_tokens > self.max_tokens:
                            self._save_text_chunk(chunks, current_chunk, section_chunk_indices)
                            current_chunk = {
                                "section_header": current_chunk['section_header'],
                                "content": "",
                                "content_with_asset": "",
                                "content_token_count": 0,
                                "pages": set(),
                                "asset_refs": set(),
                            }

                        # content: 정돈된 형식 (공백으로 연결)
                        current_chunk['content'] += sentence + " "
                        # content_with_asset: 원본 형식 보존
                        current_chunk['content_with_asset'] += sentence + " "
                        current_chunk['content_token_count'] += sentence_tokens
                        current_chunk["pages"].add(page_no)

                # 현재 청크에 추가하면 max_tokens 초과
                elif current_chunk['content'].strip() and current_chunk['content_token_count'] + estimated_tokens > self.max_tokens:
                    self._save_text_chunk(chunks, current_chunk, section_chunk_indices)
                    current_chunk = {
                        "section_header": current_chunk['section_header'],
                        "content": text + " ",  # content: 정돈된 형식
                        "content_with_asset": text_for_llm if text_for_llm.endswith('\n') else text_for_llm + " ",  # LLM용: 원본 줄바꿈 보존
                        "content_token_count": estimated_tokens,
                        "pages": {page_no},
                        "asset_refs": set(),
                    }

                # 현재 청크에 텍스트 추가
                else:
                    # content: 정돈된 형식 (공백으로 연결)
                    current_chunk['content'] += text + " "
                    # content_with_asset: 원본 형식 보존 (줄바꿈이 있으면 유지, 없으면 공백)
                    current_chunk['content_with_asset'] += text_for_llm if text_for_llm.endswith('\n') else text_for_llm + " "
                    current_chunk['content_token_count'] += estimated_tokens
                    current_chunk["pages"].add(page_no)

        # 남은 pending_assets 처리
        for asset_ref, asset_page, content_text, llm_text in pending_assets:
            current_chunk["asset_refs"].add(asset_ref)
            current_chunk["pages"].add(asset_page)

            if content_text:
                current_chunk['content'] += content_text + "\n\n"
            
            if llm_text:
                current_chunk['content_with_asset'] += llm_text + "\n\n"

        # 마지막 청크 저장
        if current_chunk['content'].strip():
            self._save_text_chunk(chunks, current_chunk, section_chunk_indices)

        return chunks

    def _save_text_chunk(self, chunks: list[dict], current_chunk: dict, section_chunk_indices: dict[str, int]):
        """텍스트 청크 저장 (Step 1)"""
        if not current_chunk['content'].strip():
            return

        # 최소 토큰 수 체크
        token_count = self._estimate_tokens(current_chunk['content'])
        if (chunks and
            token_count < self.min_chunk_tokens and
            chunks[-1]['section_header'] == current_chunk['section_header']):

            # 이전 청크에 병합
            prev_chunk = chunks[-1]
            prev_chunk['content'] = prev_chunk['content'] + " " + current_chunk['content'].strip()
            prev_chunk['content_with_asset'] = prev_chunk['content_with_asset'] + " " + current_chunk['content_with_asset'].strip()
            prev_chunk['pages'] = sorted(list(set(prev_chunk['pages']) | current_chunk['pages']))
            prev_chunk['asset_refs'] = prev_chunk['asset_refs'] | current_chunk['asset_refs']
            return

        # 섹션별 청크 인덱스
        section_header = current_chunk['section_header']
        chunk_index = section_chunk_indices.get(section_header, 0)
        section_chunk_indices[section_header] = chunk_index + 1

        # 청크 저장
        chunks.append({
            "chunk_index": chunk_index,
            "section_header": section_header,
            "content": current_chunk['content'].strip(),
            "content_with_asset": current_chunk['content_with_asset'].strip(),
            "pages": sorted(list(current_chunk['pages'])) if current_chunk['pages'] else [1],
            "asset_refs": current_chunk['asset_refs'],
        })

    def _create_dual_content(self, chunks: list[dict], all_assets: dict[str, dict[str, Any]]):
        """Step 2: Dual Content 완성"""
        for chunk in chunks:
            asset_refs = chunk.pop('asset_refs', set())
            content_with_asset = chunk.pop('content_with_asset', chunk['content'])
            clean_content = chunk['content']

            # content_for_llm으로 rename
            chunk['content_for_llm'] = content_with_asset

            if not asset_refs:
                chunk['assets'] = []
                chunk['asset_summary'] = self._create_asset_summary([], [], [])
                continue

            # 에셋을 순서대로 정렬
            sorted_refs = sorted(asset_refs, key=lambda ref: all_assets.get(ref, {}).get('_order', 999))

            # assets 생성 + appendix 마커 생성
            assets = []
            appendix_markers = []

            for ref in sorted_refs:
                if ref not in all_assets:
                    continue

                asset = all_assets[ref].copy()
                asset_type = asset.pop("_type", "pictures")
                asset.pop("_order", 0)

                # assets에 추가
                asset["type"] = asset_type
                assets.append(asset)

                # embed_with_assets=True일 때 appendix 마커 생성
                if self.embed_with_assets:
                    if asset_type == "tables":
                        table_idx = ref.split('/')[-1]
                        description = asset.get("description", "")
                        captions = asset.get("captions", [])

                        if self.include_descriptions and description:
                            appendix_markers.append(f"[TABLE:table-{table_idx}] {description}")
                        elif captions:
                            appendix_markers.append(f"[TABLE Caption: {captions[0]['text']}]")

                    elif asset_type == "pictures":
                        image_idx = ref.split('/')[-1]
                        description = asset.get("description", "")
                        captions = asset.get("captions", [])

                        if self.include_descriptions and description:
                            appendix_markers.append(f"[IMAGE:image-{image_idx}] {description}")
                        elif captions:
                            appendix_markers.append(f"[IMAGE Caption: {captions[0]['text']}]")

                    elif asset_type == "formulas":
                        formula_idx = ref.split('/')[-1]
                        formula_text = asset.get("formula", "")
                        if formula_text:
                            appendix_markers.append(f"[FORMULA:formula-{formula_idx}] {formula_text}")

            # content: embed_with_assets=True일 때 appendix 추가
            if self.embed_with_assets and appendix_markers:
                chunk['content'] = clean_content + "\n\n" + "\n".join(appendix_markers)
            else:
                chunk['content'] = clean_content

            # assets 저장
            chunk['assets'] = assets

            # asset_summary
            tables = [a for a in assets if a.get('type') == 'tables']
            pictures = [a for a in assets if a.get('type') == 'pictures']
            formulas = [a for a in assets if a.get('type') == 'formulas']
            chunk['asset_summary'] = self._create_asset_summary(tables, pictures, formulas)

    def chunk_docling_dict(self, doc_dict: dict, source_filename: str) -> Tuple[list[dict], str]:
        """
        Docling JSON 딕셔너리를 청킹

        Args:
            doc_dict: Docling JSON 딕셔너리
            source_filename: 원본 파일명

        Returns:
            (청크 리스트, 원본 파일명) 튜플
        """
        self.logger.info(f"Dual Content 청킹 시작: {source_filename}")

        # Description 가용성 체크
        if self.include_descriptions:
            pictures = doc_dict.get("pictures", [])
            tables = doc_dict.get("tables", [])

            image_desc_count = sum(
                1 for pic in pictures
                if isinstance(pic, dict) and any(
                    isinstance(annot, dict) and annot.get("label") == "vlm_image_description"
                    for annot in pic.get("annotations", [])
                )
            )
            table_desc_count = sum(
                1 for tbl in tables
                if isinstance(tbl, dict) and any(
                    isinstance(annot, dict) and annot.get("label") == "llm_table_summary"
                    for annot in tbl.get("annotations", [])
                )
            )

            if (image_desc_count == 0 and len(pictures) > 0) or (table_desc_count == 0 and len(tables) > 0):
                self.logger.warning(
                    f"⚠️  Description이 요청되었으나 JSON에 없습니다. "
                    f"advanced_mode=True로 설정했는지 확인하세요. "
                    f"(이미지: {image_desc_count}/{len(pictures)}, 테이블: {table_desc_count}/{len(tables)})"
                )

        # 캡션 수집
        captions_by_ref = self._build_asset_captions(doc_dict)

        # 에셋 메타데이터 구축
        self.logger.info("Step 0: 에셋 메타데이터 수집 중...")
        all_assets, asset_order = self._build_asset_metadata(doc_dict, captions_by_ref)
        self.logger.info(f"  에셋 총 {len(all_assets)}개 발견")

        # Step 1: 텍스트만으로 청킹
        self.logger.info("Step 1: 텍스트 청킹 중...")
        chunks = self._chunk_text_only(doc_dict, all_assets)
        self.logger.info(f"  {len(chunks)}개 텍스트 청크 생성")

        # Step 2: Dual Content 생성
        self.logger.info("Step 2: Dual Content 생성 중...")
        self._create_dual_content(chunks, all_assets)
        self.logger.info(f"  Dual Content 생성 완료")

        # Step 3: section_header를 content와 content_for_llm에 추가 (검색 최적화)
        self.logger.info("Step 3: section_header를 content와 content_for_llm에 추가 중...")
        for chunk in chunks:
            section_header = chunk.get('section_header', '').strip()
            if section_header:
                # content에 section_header 추가 (토큰 계산은 이미 완료된 상태)
                original_content = chunk.get('content', '')
                chunk['content'] = f"Section: {section_header}\n\n{original_content}"

                # content_for_llm에도 section_header 추가
                original_content_for_llm = chunk.get('content_for_llm', '')
                chunk['content_for_llm'] = f"Section: {section_header}\n\n{original_content_for_llm}"
        self.logger.info(f"  section_header 추가 완료")

        self.logger.info(f"✅ Dual Content 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks, source_filename

    # ===== 통합 처리 메서드 =====

    def process_pdf_to_chunks(
        self,
        pdf_path: Path,
        output_dir: Path,
        original_filename: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        PDF를 한 번에 처리하여 청크 생성 (Docling → Chunks)

        Args:
            pdf_path: PDF 파일 경로
            output_dir: 출력 디렉토리 (assets 저장용)
            original_filename: 원본 파일명 (없으면 pdf_path.name 사용)

        Returns:
            (chunks_list, metadata)
            metadata: {
                "table_count": int,
                "picture_count": int,
                "chunk_count": int,
                "source_file": str
            }
        """
        # 원본 파일명 결정
        if original_filename is None:
            original_filename = pdf_path.name

        # assets 디렉토리명: 원본 파일명 사용 (확장자 제외)
        original_stem = Path(original_filename).stem
        assets_dir = output_dir / f"{original_stem}_assets"

        # Step 1: Docling 처리
        doc_dict = self.convert_to_dict(pdf_path, assets_dir)

        # 원본 파일명 결정 (전달받은 original_filename 우선 사용)
        if original_filename:
            source_filename = original_filename
        else:
            origin = doc_dict.get('origin', {})
            source_filename = origin.get('filename', pdf_path.name)

        # Step 2: 청킹
        chunks, source_filename = self.chunk_docling_dict(doc_dict, source_filename)

        # Step 3: 메타데이터 추가
        for chunk in chunks:
            chunk['source_file'] = source_filename

            # Deterministic Chunk ID 생성
            chunk_id = self._generate_chunk_id(
                source_file=source_filename,
                chunk_index=chunk['chunk_index'],
                content=chunk['content']
            )
            chunk['chunk_id'] = chunk_id

            # Deterministic Section ID 생성
            section_id = self._generate_section_id(
                source_file=source_filename,
                section_header=chunk['section_header']
            )
            chunk['section_id'] = section_id

        # 메타데이터
        # OCR 정보 추출
        if self.last_ocr_result:
            ocr_used = self.last_ocr_result.needs_ocr
            ocr_reason = self.last_ocr_result.reason
            total_pages = self.last_ocr_result.total_pages
            text_layer_ratio = self.last_ocr_result.text_layer_ratio
        else:
            # OCR 감지를 하지 않은 경우 (force_ocr, force_no_ocr 사용 시)
            if self.force_ocr:
                ocr_used = True
                ocr_reason = "OCR 강제 활성화"
            elif self.force_no_ocr:
                ocr_used = False
                ocr_reason = "OCR 강제 비활성화"
            else:
                ocr_used = False
                ocr_reason = "OCR 감지 미수행"
            total_pages = len(doc_dict.get("pages", []))
            text_layer_ratio = 0.0

        metadata = {
            "table_count": len(doc_dict.get("tables", [])),
            "picture_count": len([p for p in doc_dict.get("pictures", []) if not p.get("deleted")]),
            "chunk_count": len(chunks),
            "source_file": source_filename,
            "ocr_used": ocr_used,
            "ocr_reason": ocr_reason,
            "total_pages": total_pages,
            "text_layer_ratio": text_layer_ratio,
        }

        return chunks, metadata


def main():
    """CLI 진입점"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Docling Complete + Chunking 통합 파서",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_pdf", type=Path, help="변환할 PDF 파일 경로")
    parser.add_argument(
        "-o", "--output-dir", type=Path, help="출력 디렉토리 (기본값: PDF와 같은 위치)"
    )

    # Docling 옵션
    parser.add_argument("--image-scale", type=float, default=2.0, help="이미지 해상도 스케일")
    parser.add_argument("--no-table-structure", action="store_true", help="테이블 구조 분석 비활성화")
    parser.add_argument("--force-ocr", action="store_true", help="OCR 강제 활성화")
    parser.add_argument("--no-ocr", action="store_true", help="OCR 강제 비활성화")
    parser.add_argument(
        "--ocr-engine",
        type=str,
        default="tesseract",
        choices=["tesseract", "easyocr", "rapidocr"],
        help="OCR 엔진 선택",
    )
    parser.add_argument("--ocr-threshold", type=float, default=0.4, help="OCR 필요 판단 임계값")

    # 고급 모드
    parser.add_argument("--advanced", action="store_true", help="고급 모드 활성화 (VLM/LLM description 생성)")
    parser.add_argument("--no-image-desc", action="store_true", help="이미지 VLM description 비활성화")
    parser.add_argument("--no-table-desc", action="store_true", help="테이블 LLM description 비활성화")
    parser.add_argument("--include-junk", action="store_true", help="JUNK 이미지도 포함")

    # 청킹 옵션
    parser.add_argument("--max-tokens", type=int, default=400, help="텍스트 청크당 최대 토큰 수")
    parser.add_argument("--min-chunk-tokens", type=int, default=100, help="청크 최소 토큰 수")
    parser.add_argument("--no-include-desc", action="store_false", dest="include_desc", help="description 제외")
    parser.add_argument("--embed-with-assets", action="store_true", help="content 필드에도 에셋 추가")

    parser.add_argument("-v", "--verbose", action="store_true", help="상세 로그 출력")

    args = parser.parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    # 출력 디렉토리 설정
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.input_pdf.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # 청킹 실행
    chunker = DoclingCompleteChunker(
        # Docling 옵션
        image_scale=args.image_scale,
        enable_table_structure=not args.no_table_structure,
        force_ocr=args.force_ocr,
        force_no_ocr=args.no_ocr,
        ocr_engine=args.ocr_engine,
        ocr_threshold=args.ocr_threshold,

        # 고급 모드
        advanced_mode=args.advanced,
        enable_image_description=not args.no_image_desc,
        enable_table_description=not args.no_table_desc,
        filter_junk_images=not args.include_junk,

        # 청킹 옵션
        max_tokens=args.max_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
        include_descriptions=args.include_desc,
        embed_with_assets=args.embed_with_assets,
    )

    try:
        chunks, metadata = chunker.process_pdf_to_chunks(args.input_pdf, output_dir)

        # 청크 JSON 저장
        chunks_json_path = output_dir / f"{args.input_pdf.stem}_chunks.json"
        with open(chunks_json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ 변환 완료: {chunks_json_path}")
        print(f"   청크 수: {metadata['chunk_count']}개")
        print(f"   테이블: {metadata['table_count']}개")
        print(f"   이미지: {metadata['picture_count']}개")

    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


# ===== API 호환 레이어 (parser.py 인터페이스 대체) =====

@dataclass
class IntegratedParserConfig:
    """통합 파서 설정 (고급 모드만 노출)"""
    # 고급 모드: VLM/LLM description 생성
    enable_image_description: bool = False
    enable_table_description: bool = False
    # 임베딩 최적화: content에 asset 설명 appendix 추가
    embed_with_assets: bool = False


def process_pdf_to_chunks(
    file_content: bytes,
    filename: str,
    output_dir: Path,
    source_id: Optional[str] = None,
    config: Optional[IntegratedParserConfig] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    PDF를 한 번에 처리하여 청크 생성 (Docling Complete → Dual Content Chunks)

    이 함수는 parser.py의 process_pdf_to_chunks()와 동일한 인터페이스를 제공하지만,
    내부적으로 DoclingCompleteChunker를 사용합니다.

    Args:
        file_content: PDF 바이너리
        filename: 파일명
        output_dir: 출력 디렉토리 (assets 저장 위치)
        source_id: 소스 UUID (없으면 자동 생성)
        config: 파서 설정

    Returns:
        (chunks_list, metadata)
        chunks_list: 청크 리스트 (각 청크는 dict)
        metadata: {
            "table_count": int,
            "picture_count": int,
            "chunk_count": int,
            "ocr_used": bool,
            "ocr_reason": str,
            "total_pages": int,
            "text_layer_ratio": float
        }
    """
    import uuid

    if config is None:
        config = IntegratedParserConfig()

    if source_id is None:
        source_id = str(uuid.uuid4())

    # DoclingChunker 인스턴스 생성 (모든 옵션은 내부 기본값 사용)
    # 고급 모드만 사용자가 선택 가능
    advanced_mode_enabled = config.enable_image_description or config.enable_table_description
    logging.getLogger(__name__).info(
        f"DoclingChunker 초기화: advanced_mode={advanced_mode_enabled}, "
        f"enable_image_description={config.enable_image_description}, "
        f"enable_table_description={config.enable_table_description}, "
        f"embed_with_assets={config.embed_with_assets}"
    )

    chunker = DoclingChunker(
        advanced_mode=advanced_mode_enabled,
        enable_image_description=config.enable_image_description,
        enable_table_description=config.enable_table_description,
        embed_with_assets=config.embed_with_assets,
        progress_callback=progress_callback,
    )

    # PDF를 임시 파일로 저장 (DoclingCompleteChunker가 Path를 받기 때문)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
        tmp_pdf.write(file_content)
        tmp_pdf_path = Path(tmp_pdf.name)

    try:
        # 청크 생성 (원본 파일명 전달)
        chunks, metadata = chunker.process_pdf_to_chunks(
            tmp_pdf_path,
            output_dir,
            original_filename=filename
        )
        return chunks, metadata

    finally:
        # 임시 PDF 파일 삭제
        if tmp_pdf_path.exists():
            tmp_pdf_path.unlink()


if __name__ == "__main__":
    main()
