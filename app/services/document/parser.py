# -*- coding: utf-8 -*-
"""
통합 PDF 파서 (Docling Auto + General Chunking)

docling_auto.py와 general_chunking.py의 핵심 로직을 통합하여
단일 함수 호출로 PDF → Chunks까지 처리하는 라이브러리 모듈입니다.

Features:
- OCR 자동 감지 및 적용
- 이미지/테이블 description 생성 (VLM)
- 구조화된 청킹 (섹션 기반, 오버랩 지원)
- 메모리 기반 처리 (파일 I/O 최소화)
"""
import os
import logging
import json
import re
import uuid
import unicodedata
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF

import requests
from PIL import Image
import nltk
from nltk.tokenize import sent_tokenize

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


log = logging.getLogger(__name__)


# ===== 상수 =====
DROP_KEYS = {
    "image", "image_ref", "image_uri", "uri", "thumb", "thumbnail",
    "images", "page_image", "picture_image"
}

SECTION_HEADERS = {
    "title", "part", "preamble", "recital", "chapter", "subchapter",
    "section", "subsection", "article", "annex", "annexes", "appendix", "schedule",
    "paragraph", "subparagraph", "point", "subpoint", "item", "clause", "subclause", "heading"
}

SKIP_LABELS = {"page_header", "page_footer", "running_header", "running_footer", "header", "footer"}

REF_RE = re.compile(r"^#?/pictures/(\d+)(?:$|/|\b)", re.I)
ASSET_REF_RE = re.compile(r"^#/(texts|groups|tables|pictures)/(\d+)$", re.I)
_CTRL = re.compile(r"[\u0000-\u001F\u007F]")


# ===== 설정 클래스 =====
@dataclass
class IntegratedParserConfig:
    """통합 파서 설정"""
    # Docling 옵션
    image_resolution_scale: float = 3.0
    pixel_area_threshold: int = 0
    enable_table_structure: bool = True

    # OCR 옵션
    auto_detect_ocr: bool = True
    force_ocr: bool = False
    force_no_ocr: bool = False
    min_chars_per_page: int = 50
    ocr_engine: str = "tesseract"  # tesseract, rapidocr

    # 이미지 description (VLM)
    enable_image_description: bool = False
    ollama_url: str = "http://localhost:11434"
    vision_model: str = "qwen3-vl:latest"
    image_description_prompt: str = "Analyze this image. If it's JUNK (decorative element, logo, icon, or not meaningful content), respond with only 'JUNK'. Otherwise, describe this technical image in english concisely. Focus on key components. Limit to 3~5 sentences."
    skip_junk_images: bool = True

    # 테이블 description (VLM)
    enable_table_description: bool = False
    table_description_prompt: str = "Analyze this table and provide a concise description in english. Explain what data it contains and any key insights. Limit to 3~5 sentences."

    # 청킹 옵션
    max_chars: int = 1500
    min_section_chars: int = 120
    overlap_sentences: int = 2
    collect_assets: bool = True

    # 디버깅 옵션
    save_debug_json: bool = True      # True면 output_dir에 JSON 저장


@dataclass
class OCRDetectionResult:
    """OCR 감지 결과"""
    needs_ocr: bool
    reason: str
    avg_chars_per_page: float
    has_images: bool
    total_pages: int


# ===== OCR 감지 =====
def detect_ocr_requirement(
    pdf_content: bytes,
    min_chars_per_page: int = 50
) -> OCRDetectionResult:
    """
    PDF 바이너리를 분석하여 OCR 필요 여부를 판단

    Args:
        pdf_content: PDF 파일 바이너리
        min_chars_per_page: 페이지당 최소 글자 수 기준

    Returns:
        OCRDetectionResult
    """
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        total_pages = len(doc)
        total_chars = 0
        has_images = False

        for page in doc:
            text = page.get_text()
            total_chars += len(text.strip())

            # 이미지 확인
            if page.get_images():
                has_images = True

        avg_chars = total_chars / total_pages if total_pages > 0 else 0
        needs_ocr = avg_chars < min_chars_per_page

        if needs_ocr:
            reason = f"Average {avg_chars:.1f} chars/page < {min_chars_per_page} threshold"
        else:
            reason = f"Sufficient text layer detected ({avg_chars:.1f} chars/page)"

        doc.close()

        return OCRDetectionResult(
            needs_ocr=needs_ocr,
            reason=reason,
            avg_chars_per_page=avg_chars,
            has_images=has_images,
            total_pages=total_pages
        )
    except Exception as e:
        log.warning(f"OCR detection failed: {e}")
        return OCRDetectionResult(
            needs_ocr=False,
            reason=f"Detection failed: {e}",
            avg_chars_per_page=0,
            has_images=False,
            total_pages=0
        )


# ===== VLM Description 생성 =====
def generate_vlm_description(
    image: Image.Image,
    prompt: str,
    ollama_url: str,
    model: str = "qwen3-vl:latest"
) -> Optional[str]:
    """
    Ollama VLM을 사용하여 이미지 description 생성
    """
    try:
        import io
        import base64

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            log.warning(f"VLM API error: {response.status_code}")
            return None
    except Exception as e:
        log.warning(f"VLM description failed: {e}")
        return None


# ===== 헬퍼 함수 =====
def _strip_ctrl(s: str) -> str:
    return _CTRL.sub("", s or "")


def _clean_text(s: str) -> str:
    """텍스트 정제 (Unicode 정규화만 수행)"""
    if not s:
        return ""

    # Unicode 정규화
    s = unicodedata.normalize('NFKC', s)
    
    # 깨진 문자 제거
    s = s.replace("\ufffd", "")

    return s


def _collapse_ws_keep_newlines(s: str) -> str:
    s = _clean_text(s)
    s = _strip_ctrl(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in s.split("\n")]
    return "\n".join(lines)


def _embed_text(s: str) -> str:
    s = _clean_text(s)
    s = _strip_ctrl(s)
    s = re.sub(r"[ \t\f\v]+", " ", s.replace("\r\n", "\n").replace("\r", "\n"))
    s = re.sub(r"\s*\n\s*", " ", s).strip()
    return s


def _prune_images(obj):
    if isinstance(obj, dict):
        return {k: _prune_images(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [_prune_images(v) for v in obj]
    return obj


def _docling_table_to_markdown(table_data: Dict[str, Any]) -> str:
    """테이블을 Markdown으로 변환"""
    cells = table_data.get("table_cells")
    if not cells:
        return "Table data not available."

    data_list = []
    max_row, max_col = 0, 0

    for cell in cells:
        r_start = cell.get("start_row_offset_idx", 0)
        c_start = cell.get("start_col_offset_idx", 0)
        text = (_collapse_ws_keep_newlines(cell.get("text")) or "").replace('\n', ' ').strip()
        
        # 파이프 문자 이스케이프 (마크다운 테이블 구분자와 충돌 방지)
        text = text.replace("|", "\\|")

        if text:
            data_list.append({"row": r_start, "col": c_start, "text": text})
            max_row = max(max_row, r_start)
            max_col = max(max_col, c_start)

    if not data_list:
        return "Table data not available."

    table_array = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for item in data_list:
        r, c = item['row'], item['col']
        if 0 <= r <= max_row and 0 <= c <= max_col:
            table_array[r][c] = item['text']

    markdown_lines = []
    header = table_array[0]
    markdown_lines.append("| " + " | ".join(header) + " |")

    if max_row >= 1:
        markdown_lines.append("|" + "---|" * len(header))
        for row in table_array[1:]:
            markdown_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown_lines)


# ===== 후처리 함수 (general_docling_ocr.py에서 가져옴) =====
REF_RE = re.compile(r"^#?/pictures/(\d+)(?:$|/|\b)", re.I)


def _extract_vlm_description(
    picture_node: Dict[str, Any],
    picture_ref: str | None = None,
) -> str:
    """이미지 노드에서 VLM description 추출"""
    ref_tag = ""
    if picture_ref:
        m = REF_RE.match(picture_ref)
        if m:
            ref_tag = f" pictures/{m.group(1)}"
        else:
            ref_tag = f" {picture_ref}"

    if isinstance(picture_node.get("annotations"), list):
        # 우선순위 1: label이 'picture_description'인 것 (우리가 생성한 VLM description)
        for ann in picture_node["annotations"]:
            if (
                isinstance(ann, dict)
                and ann.get("label") == "picture_description"
                and ann.get("text")
            ):
                return f"[IMAGE CONTEXT{ref_tag}: {ann['text']}]"
        
        # 우선순위 2: kind가 'description'인 것 (Docling 기본 VLM)
        for ann in picture_node["annotations"]:
            if (
                isinstance(ann, dict)
                and ann.get("kind") == "description"
                and ann.get("text")
            ):
                return f"[IMAGE CONTEXT{ref_tag}: {ann['text']}]"

    # Captions에서도 찾기 (Docling이 item.caption으로 저장한 경우)
    if isinstance(picture_node.get("captions"), list):
        for cap in picture_node["captions"]:
            if isinstance(cap, dict) and cap.get("text"):
                return f"[IMAGE CONTEXT{ref_tag}: {cap['text']}]"

    return f"[IMAGE CONTEXT{ref_tag}: (VLM Description Not Found)]"


def _build_table_captions(doc_dict: Dict[str, Any]) -> Dict[str, List[str]]:
    """테이블 캡션 수집"""
    caps: Dict[str, List[str]] = {}
    for node in doc_dict.get("texts") or []:
        if (node.get("label") or "").strip().lower() != "caption":
            continue
        parent = node.get("parent") or {}
        cref = parent.get("cref") or parent.get("$ref")
        if not isinstance(cref, str):
            continue
        if "/tables/" not in cref:
            continue

        txt = (node.get("text") or node.get("orig") or "").strip()
        if not txt:
            continue
        caps.setdefault(cref, []).append(txt)
    return caps


def merge_image_annotations(doc_dict: Dict[str, Any]) -> Dict[str, Any]:
    """이미지를 앞 텍스트 노드에 병합"""
    log.info("Starting post-processing: Merging VLM annotations into preceding text nodes.")

    text_map = {t["self_ref"]: t for t in doc_dict.get("texts", [])}
    picture_map = {p["self_ref"]: p for p in doc_dict.get("pictures", [])}

    containers = []

    if "body" in doc_dict and isinstance(doc_dict["body"].get("children"), list):
        containers.append(doc_dict["body"]["children"])
    for item in doc_dict.get("sections", []):
        if isinstance(item.get("children"), list):
            containers.append(item["children"])
    for item in doc_dict.get("groups", []):
        if isinstance(item.get("children"), list):
            containers.append(item["children"])

    merge_count = 0

    for children_array in containers:
        for i in range(len(children_array) - 1, 0, -1):
            current_ref_obj = children_array[i]

            if current_ref_obj.get("$ref", "").startswith("#/pictures/"):
                picture_ref = current_ref_obj["$ref"]

                prev_ref_obj = children_array[i - 1]
                if prev_ref_obj.get("$ref", "").startswith("#/texts/"):
                    prev_text_ref = prev_ref_obj["$ref"]

                    if picture_ref in picture_map and prev_text_ref in text_map:
                        picture_node = picture_map[picture_ref]
                        vlm_description_text = _extract_vlm_description(picture_node, picture_ref)

                        text_node = text_map[prev_text_ref]
                        text_node["text"] += vlm_description_text
                        if "orig" in text_node:
                            text_node["orig"] += vlm_description_text

                        children_array.pop(i)
                        merge_count += 1
                        log.debug(f"Merged {picture_ref} into {prev_text_ref}. Picture reference removed.")

    log.info(f"Finished merging annotations. Total {merge_count} merges performed.")
    return doc_dict


def merge_table_annotations(doc_dict: Dict[str, Any]) -> Dict[str, Any]:
    """테이블을 앞 텍스트 노드에 병합"""
    log.info("Starting post-processing: Merging TABLE CONTEXT into preceding text nodes.")

    text_map = {t["self_ref"]: t for t in doc_dict.get("texts") or []}
    table_map = {f"#/tables/{i}": t for i, t in enumerate(doc_dict.get("tables") or []) if isinstance(t, dict)}

    table_captions = _build_table_captions(doc_dict)

    containers: List[List[Dict[str, Any]]] = []
    if "body" in doc_dict and isinstance(doc_dict["body"].get("children"), list):
        containers.append(doc_dict["body"]["children"])
    for item in doc_dict.get("sections") or []:
        if isinstance(item.get("children"), list):
            containers.append(item["children"])
    for item in doc_dict.get("groups") or []:
        if isinstance(item.get("children"), list):
            containers.append(item["children"])

    merge_count = 0

    for children_array in containers:
        for i in range(len(children_array) - 1, 0, -1):
            current_ref_obj = children_array[i]
            ref = current_ref_obj.get("$ref")
            if not isinstance(ref, str):
                continue
            if not ref.startswith("#/tables/"):
                continue

            table_ref = ref

            prev_ref_obj = children_array[i - 1]
            prev_ref = prev_ref_obj.get("$ref")
            if not isinstance(prev_ref, str) or not prev_ref.startswith("#/texts/"):
                continue

            if prev_ref not in text_map:
                continue

            text_node = text_map[prev_ref]

            ref_tag = table_ref.lstrip("#/")

            cap_list = table_captions.get(table_ref) or []
            if cap_list:
                cap_text = cap_list[0].strip()
                context_str = f"[TABLE CONTEXT {ref_tag}: {cap_text}]"
            else:
                context_str = f"[TABLE CONTEXT {ref_tag}]"

            if "text" in text_node and isinstance(text_node["text"], str):
                if text_node["text"] and not text_node["text"].endswith(" "):
                    text_node["text"] += " "
                text_node["text"] += context_str

            if "orig" in text_node and isinstance(text_node["orig"], str):
                if text_node["orig"] and not text_node["orig"].endswith(" "):
                    text_node["orig"] += " "
                text_node["orig"] += context_str

            children_array.pop(i)
            merge_count += 1
            log.debug(f"Merged TABLE {table_ref} into {prev_ref} and removed table ref from children.")

    log.info(f"Finished merging table annotations. Total {merge_count} merges performed.")
    return doc_dict


def _split_by_sentences(text: str, max_chars: int = float('inf')) -> List[str]:
    """문장 단위 분할"""
    if not text or len(text) <= max_chars:
        return [text] if text else []

    try:
        parts = sent_tokenize(text.strip())
    except Exception:
        return [text.strip()]

    return [p.strip() for p in parts if p.strip()]


# ===== 메인 처리 함수 =====
def process_pdf_integrated(
    file_content: bytes,
    filename: str,
    output_dir: Path,
    config: Optional[IntegratedParserConfig] = None
) -> Tuple[Dict[str, Any], int, int, OCRDetectionResult]:
    """
    PDF를 Docling으로 처리하여 JSON 반환 (OCR 자동 감지 포함)

    Args:
        file_content: PDF 파일 바이너리
        filename: 파일명
        output_dir: 출력 디렉토리
        config: 파서 설정

    Returns:
        (doc_dict, table_count, picture_count, ocr_result)
    """
    if config is None:
        config = IntegratedParserConfig()

    # 1. OCR 감지
    ocr_result = None
    enable_ocr = False

    if config.force_ocr:
        enable_ocr = True
        log.info("OCR forced enabled")
    elif config.force_no_ocr:
        enable_ocr = False
        log.info("OCR forced disabled")
    elif config.auto_detect_ocr:
        ocr_result = detect_ocr_requirement(file_content, config.min_chars_per_page)
        enable_ocr = ocr_result.needs_ocr
        log.info(f"OCR auto-detect: {ocr_result.reason}")

    # 2. Docling 파이프라인 설정
    pdf_opts = PdfPipelineOptions()
    pdf_opts.images_scale = config.image_resolution_scale
    pdf_opts.generate_page_images = True
    pdf_opts.generate_picture_images = True
    pdf_opts.do_ocr = enable_ocr
    pdf_opts.do_table_structure = config.enable_table_structure
    pdf_opts.do_picture_description = False  # Docling 기본 VLM 비활성화 (Ollama VLM만 사용)

    # OCR 엔진 설정
    if enable_ocr:
        if config.ocr_engine == "rapidocr":
            pdf_opts.ocr_options = RapidOcrOptions()
        else:  # tesseract
            pdf_opts.ocr_options = TesseractCliOcrOptions()

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )

    # 3. 임시 파일로 저장하여 처리
    temp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(temp_dir, filename)

    try:
        with open(tmp_path, 'wb') as tmp_file:
            tmp_file.write(file_content)

        conv = converter.convert(tmp_path)
        doc = conv.document
        stem = Path(filename).stem
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

    # 4. 이미지/테이블 추출
    tables_dir = output_dir / "tables"
    images_dir = output_dir / "images"
    tables_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    t_idx = 0
    p_idx = 0
    keep_pidx: Set[int] = set()
    
    # VLM descriptions 저장 (나중에 JSON에 추가)
    table_descriptions = {}  # {table_idx: description}
    picture_descriptions = {}  # {picture_idx: description}

    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            try:
                img = item.get_image(doc)
                img.save(tables_dir / f"{stem}-table-{t_idx}.png", "PNG")

                # 테이블 description 생성
                if config.enable_table_description:
                    desc = generate_vlm_description(
                        img,
                        config.table_description_prompt,
                        config.ollama_url,
                        config.vision_model
                    )
                    if desc:
                        table_descriptions[t_idx] = desc
            except Exception as e:
                log.warning(f"Table {t_idx} image skipped: {e}")
            t_idx += 1

        elif isinstance(item, PictureItem):
            try:
                img = item.get_image(doc)
                w, h = img.size

                if w * h >= config.pixel_area_threshold:
                    # 이미지 description 생성 및 JUNK 판별 (저장 전)
                    should_save = True
                    desc = None
                    
                    if config.enable_image_description:
                        desc = generate_vlm_description(
                            img,
                            config.image_description_prompt,
                            config.ollama_url,
                            config.vision_model
                        )
                        
                        # JUNK 판별
                        if desc and config.skip_junk_images and desc.strip().upper().startswith("JUNK"):
                            log.info(f"Picture {p_idx} filtered as JUNK (not saved)")
                            should_save = False
                    
                    # JUNK가 아닌 경우만 저장
                    if should_save:
                        img.save(images_dir / f"{stem}-image-{p_idx}.png", "PNG")
                        
                        # Description 저장
                        if desc:
                            picture_descriptions[p_idx] = desc
                        
                        keep_pidx.add(p_idx)
                else:
                    log.info(f"Picture {p_idx} skipped (too small: {w}x{h})")
            except Exception as e:
                log.warning(f"Picture {p_idx} image skipped: {e}")
            p_idx += 1

    # 5. JSON 변환
    doc_dict = doc.export_to_dict()
    
    # 5.1 VLM descriptions를 annotations에 추가
    if table_descriptions:
        for idx, desc in table_descriptions.items():
            if "tables" in doc_dict and idx < len(doc_dict["tables"]):
                table_node = doc_dict["tables"][idx]
                if "annotations" not in table_node:
                    table_node["annotations"] = []
                table_node["annotations"].append({
                    "kind": "description",
                    "label": "table_description",
                    "text": desc
                })
    
    if picture_descriptions:
        for idx, desc in picture_descriptions.items():
            if "pictures" in doc_dict and idx < len(doc_dict["pictures"]):
                picture_node = doc_dict["pictures"][idx]
                if "annotations" not in picture_node:
                    picture_node["annotations"] = []
                picture_node["annotations"].append({
                    "kind": "description",
                    "label": "picture_description",
                    "text": desc
                })

    # 6. 후처리: 이미지/테이블을 텍스트 노드에 병합
    doc_dict = merge_image_annotations(doc_dict)
    doc_dict = merge_table_annotations(doc_dict)

    # 7. 이미지 바이너리 제거
    clean = _prune_images(doc_dict)

    # 8. 디버깅: JSON 저장 (옵션)
    if config.save_debug_json:
        json_path = output_dir / f"{stem}_debug.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(clean, f, ensure_ascii=False, indent=2)
            log.info(f"Debug JSON saved: {json_path}")
        except Exception as e:
            log.warning(f"Failed to save debug JSON: {e}")

    return clean, t_idx, len(keep_pidx), ocr_result


def chunk_integrated_json(
    doc_dict: Dict[str, Any],
    source_name: str,
    source_id: str,
    config: Optional[IntegratedParserConfig] = None
) -> List[Dict[str, Any]]:
    """
    Docling JSON을 청킹 (general_chunking.py 로직 통합)

    Args:
        doc_dict: Docling JSON dictionary
        source_name: 소스 파일명
        source_id: 소스 UUID
        config: 파서 설정

    Returns:
        List of chunk dicts with page_content and metadata
    """
    if config is None:
        config = IntegratedParserConfig()

    J = doc_dict

    # 섹션 헤더 찾기
    def _is_section_header_label(n: dict) -> Tuple[bool, str]:
        if (n.get("label") or "").strip().lower() == "section_header":
            t = (n.get("section_header") or "").strip().lower()
            if t in SECTION_HEADERS:
                return True, t
            return True, (t or "section")
        return False, ""

    def _page_no(n: dict) -> int:
        prov = n.get("prov") or []
        if prov and isinstance(prov[0], dict):
            p = prov[0].get("page_no")
            if isinstance(p, int):
                return p
        return 10**9

    # 선형화
    items = []
    for coll in ("texts", "tables", "pictures"):
        arr = J.get(coll) or []
        for i, node in enumerate(arr):
            node["_kind"] = coll
            node["_id"] = f"{coll}/{i}"
            pn = _page_no(node)
            items.append((pn, coll, i, node, f"#/{coll}/{i}"))

    items.sort(key=lambda t: (t[0], t[1], t[2]))
    flat = [(coll, i, node, ref, pn) for pn, coll, i, node, ref in items]

    # 섹션 헤더 인덱스
    heads = [i for i, (c, _, n, _, _) in enumerate(flat) if c == "texts" and _is_section_header_label(n)[0]]
    if not heads:
        heads = [0]

    # Caption 수집
    captions_by_ref: Dict[str, List[Dict[str, Any]]] = {}
    for n in J.get("texts") or []:
        if (n.get("label") or "").strip().lower() != "caption":
            continue
        parent = n.get("parent") or {}
        cref = parent.get("cref") or parent.get("$ref")
        if not isinstance(cref, str) or not ASSET_REF_RE.match(cref):
            continue
        txt = _collapse_ws_keep_newlines(n.get("text") or n.get("orig") or "")
        if not txt:
            continue
        pg = (n.get("prov") or [{}])[0].get("page_no")
        captions_by_ref.setdefault(cref, []).append({"text": txt, "page_no": pg})

    # 청킹
    chunks: List[Dict[str, Any]] = []
    seq = 0
    path_stack: List[str] = []

    for k, h in enumerate(heads):
        coll, _, node, norm_ref, header_page = flat[h]
        is_head, head_type = _is_section_header_label(node)

        # 섹션 제목
        title_txt = ""
        if is_head:
            for key in ("title", "header", "name", "text", "caption"):
                v = node.get(key)
                if isinstance(v, str) and v.strip():
                    title_txt = _collapse_ws_keep_newlines(v)
                    break
            if not title_txt:
                title_txt = (head_type or "SECTION").upper()
        else:
            title_txt = "DOCUMENT"

        level = int(node.get("level") or 1) if is_head else 1
        if level <= 0:
            level = 1

        # 경로 스택 업데이트
        if len(path_stack) >= level:
            path_stack = path_stack[:level - 1]
        while len(path_stack) < level - 1:
            path_stack.append("")
        if len(path_stack) == level - 1:
            path_stack.append(title_txt)
        else:
            path_stack[level - 1] = title_txt

        section_path = [p for p in path_stack if p]
        section_title = " > ".join(section_path) if section_path else title_txt
        section_id = str(uuid.uuid5(uuid.UUID(source_id), f"section:{norm_ref}"))

        # 섹션 범위
        lo = h + (1 if is_head else 0)
        hi = (heads[k + 1] - 1) if (k + 1 < len(heads)) else (len(flat) - 1)

        # 컨텐츠 수집
        content_nodes: List[Tuple[str, int]] = []
        section_assets: List[Tuple[str, Dict[str, Any]]] = []

        for j in range(lo, hi + 1):
            c2, _, n2, norm2, _pn = flat[j]
            current_page = _pn if isinstance(_pn, int) else header_page

            if c2 == "texts":
                lab = (n2.get("label") or "").strip().lower()
                if lab in SKIP_LABELS:
                    continue
                if lab not in {"text", "paragraph", "list_item", "formula", "footnote"}:
                    continue

                raw_t = (n2.get("text") or n2.get("orig") or "").strip()
                if lab == "list_item":
                    mk = n2.get("marker") or ""
                    raw_t = f"{mk} {raw_t}".strip()

                cleaned = _collapse_ws_keep_newlines(raw_t)
                if cleaned:
                    content_nodes.append((cleaned, current_page))

            elif c2 in {"tables", "pictures"} and config.collect_assets:
                pg = (n2.get("prov") or [{}])[0].get("page_no")
                asset_entry = {
                    "uid": n2.get("self_ref") or n2.get("uid") or n2.get("id"),
                    "ref_norm": norm2,
                    "page_no": pg,
                    "captions": captions_by_ref.get(norm2, []),
                }

                if c2 == "tables":
                    table_data = n2.get("data")
                    if table_data:
                        md = _docling_table_to_markdown(table_data)
                        if md and md != "Table data not available.":
                            asset_entry["markdown_table"] = md

                section_assets.append((c2, asset_entry))

        # 병합 로직 (짧은 섹션)
        all_text = " ".join([text for text, _ in content_nodes]).strip()
        is_short = len(_embed_text(all_text)) < config.min_section_chars

        if is_short and chunks and not section_assets:
            last = chunks[-1]
            merged = (last["metadata"]["display_content"] + " " + all_text).strip()
            last["metadata"]["display_content"] = merged
            last["page_content"] = _embed_text(merged)

            new_pages = {page for _, page in content_nodes if isinstance(page, int)}
            last["metadata"]["pages"] = sorted(set(last["metadata"]["pages"]) | new_pages)
            continue

        # 청킹
        sentence_page_map: List[Tuple[str, int]] = []
        for text, page in content_nodes:
            sentences = _split_by_sentences(text)
            for s in sentences:
                if s:
                    sentence_page_map.append((s, page))

        chunk_parts: List[Tuple[str, Set[int]]] = []
        current_text = ""
        current_pages: Set[int] = set()

        for sentence, page in sentence_page_map:
            add_len = len(sentence) + (1 if current_text else 0)

            if current_text and (len(current_text) + add_len > config.max_chars):
                chunk_parts.append((current_text.strip(), set(current_pages)))
                current_text = sentence
                current_pages = {page}
            else:
                if current_text:
                    current_text += " " + sentence
                else:
                    current_text = sentence
                current_pages.add(page)

        if current_text:
            chunk_parts.append((current_text.strip(), set(current_pages)))

        if not chunk_parts:
            p_num = header_page if isinstance(header_page, int) else 1
            chunk_parts.append((section_title, {p_num}))

        # 에셋 맵핑 (페이지 기준)
        tables_by_page = {}
        pictures_by_page = {}
        for kind, asset in section_assets:
            page_no = asset.get("page_no")
            if page_no is None:
                continue

            if kind == "tables":
                tables_by_page.setdefault(page_no, []).append(asset)
            elif kind == "pictures":
                pictures_by_page.setdefault(page_no, []).append(asset)

        # 청크 생성 (각 청크의 페이지와 일치하는 assets만 포함)
        for i, (part, pages) in enumerate(chunk_parts):
            # 현재 청크의 페이지에 해당하는 assets 수집
            chunk_tables = []
            chunk_pictures = []

            for page in pages:
                if page in tables_by_page:
                    chunk_tables.extend(tables_by_page[page])
                if page in pictures_by_page:
                    chunk_pictures.extend(pictures_by_page[page])

            # 텍스트 내 [IMAGE CONTEXT ...], [TABLE CONTEXT ...] 패턴을 간단한 형식으로 변환
            enhanced_text = part

            # [IMAGE CONTEXT pictures/2: description] -> [Image 2: description]
            enhanced_text = re.sub(
                r'\[IMAGE CONTEXT pictures/(\d+): ([^\]]+)\]',
                r'[Image \1: \2]',
                enhanced_text
            )

            # [TABLE CONTEXT tables/3: description] -> [Table 3: description]
            enhanced_text = re.sub(
                r'\[TABLE CONTEXT tables/(\d+): ([^\]]+)\]',
                r'[Table \1: \2]',
                enhanced_text
            )

            # VLM description이 없는 경우: [IMAGE CONTEXT pictures/2: (VLM Description Not Found)] -> [Image 2]
            enhanced_text = re.sub(
                r'\[IMAGE CONTEXT pictures/(\d+): \(VLM Description Not Found\)\]',
                r'[Image \1]',
                enhanced_text
            )

            # 캡션 없는 테이블: [TABLE CONTEXT tables/3] -> [Table 3]
            enhanced_text = re.sub(
                r'\[TABLE CONTEXT tables/(\d+)\]',
                r'[Table \1]',
                enhanced_text
            )
            
            # 텍스트에서 참조된 이미지/테이블 인덱스 추출하여 assets에 추가
            # [Image X:...] 또는 [Table X:...] 패턴에서 인덱스 추출
            image_indices = set(re.findall(r'\[Image (\d+)[:\]]', enhanced_text))
            table_indices = set(re.findall(r'\[Table (\d+)[:\]]', enhanced_text))
            
            # JSON의 pictures/tables 배열에서 직접 가져오기
            for idx_str in image_indices:
                idx = int(idx_str)
                if "pictures" in doc_dict and idx < len(doc_dict["pictures"]):
                    picture_node = doc_dict["pictures"][idx]
                    pg = (picture_node.get("prov") or [{}])[0].get("page_no")
                    picture_asset = {
                        "uid": picture_node.get("self_ref") or picture_node.get("uid"),
                        "ref_norm": f"#/pictures/{idx}",
                        "page_no": pg,
                        "captions": [],  # 이미 텍스트에 병합되어 있음
                    }
                    # 현재 페이지와 일치하는 경우만 추가
                    if pg in pages:
                        chunk_pictures.append(picture_asset)
            
            for idx_str in table_indices:
                idx = int(idx_str)
                if "tables" in doc_dict and idx < len(doc_dict["tables"]):
                    table_node = doc_dict["tables"][idx]
                    pg = (table_node.get("prov") or [{}])[0].get("page_no")
                    table_asset = {
                        "uid": table_node.get("self_ref") or table_node.get("uid"),
                        "ref_norm": f"#/tables/{idx}",
                        "page_no": pg,
                        "captions": [],
                    }
                    # 마크다운 테이블 추가
                    table_data = table_node.get("data")
                    if table_data:
                        md = _docling_table_to_markdown(table_data)
                        if md and md != "Table data not available.":
                            table_asset["markdown_table"] = md
                    
                    # 현재 페이지와 일치하는 경우만 추가
                    if pg in pages:
                        chunk_tables.append(table_asset)
            
            # 페이지 기반으로 수집된 assets 중 텍스트에 없는 항목을 텍스트에 추가
            # (병합되지 않은 standalone 이미지/테이블 처리)
            standalone_descriptions = []
            
            for pic_asset in chunk_pictures:
                # ref_norm에서 인덱스 추출
                match = re.search(r'/pictures/(\d+)', pic_asset.get("ref_norm", ""))
                if match:
                    pic_idx = match.group(1)
                    # 이미 텍스트에 포함되어 있는지 확인
                    if pic_idx not in image_indices:
                        # 텍스트에 없으면 추가
                        idx = int(pic_idx)
                        if "pictures" in doc_dict and idx < len(doc_dict["pictures"]):
                            picture_node = doc_dict["pictures"][idx]
                            desc = _extract_vlm_description(picture_node, f"#/pictures/{idx}")
                            standalone_descriptions.append(desc)
            
            for tbl_asset in chunk_tables:
                # ref_norm에서 인덱스 추출
                match = re.search(r'/tables/(\d+)', tbl_asset.get("ref_norm", ""))
                if match:
                    tbl_idx = match.group(1)
                    # 이미 텍스트에 포함되어 있는지 확인
                    if tbl_idx not in table_indices:
                        # 텍스트에 없으면 description 추출하여 추가
                        idx = int(tbl_idx)
                        if "tables" in doc_dict and idx < len(doc_dict["tables"]):
                            table_node = doc_dict["tables"][idx]
                            # annotations에서 table_description 찾기
                            table_desc = None
                            if isinstance(table_node.get("annotations"), list):
                                for ann in table_node["annotations"]:
                                    if (isinstance(ann, dict) and 
                                        ann.get("label") == "table_description" and 
                                        ann.get("text")):
                                        table_desc = ann["text"]
                                        break
                            
                            if table_desc:
                                standalone_descriptions.append(f"[Table {tbl_idx}: {table_desc}]")
                            else:
                                standalone_descriptions.append(f"[Table {tbl_idx}]")

            
            # Standalone descriptions를 텍스트에 추가
            display_text = part  # display용 텍스트
            if standalone_descriptions:
                enhanced_text = enhanced_text + " " + " ".join(standalone_descriptions)
                display_text = display_text + " " + " ".join(standalone_descriptions)

            chunk = {
                "page_content": _embed_text(enhanced_text),  # Description 포함된 텍스트로 임베딩
                "metadata": {
                    "source": source_name,
                    "source_name": source_name,
                    "source_id": source_id,
                    "section_id": section_id,
                    "section_title": section_title,
                    "section_path": section_path,
                    "level": len(section_path),
                    "pages": sorted(list(pages)),
                    "labels": ["section"],
                    "part_index": i,
                    "display_content": display_text,  # Standalone descriptions 포함
                    "seq": seq + i,
                    "assets": {
                        "tables": chunk_tables,
                        "pictures": chunk_pictures,
                    },
                },
            }
            chunks.append(chunk)

        seq += len(chunk_parts)

    return chunks


# ===== 원스텝 처리 =====
def process_pdf_to_chunks(
    file_content: bytes,
    filename: str,
    output_dir: Path,
    source_id: Optional[str] = None,
    config: Optional[IntegratedParserConfig] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    PDF를 한 번에 처리하여 청크 생성 (Docling → Chunks)

    Args:
        file_content: PDF 바이너리
        filename: 파일명
        output_dir: 출력 디렉토리
        source_id: 소스 UUID (없으면 자동 생성)
        config: 파서 설정

    Returns:
        (chunks_list, metadata)
        metadata: {
            "table_count": int,
            "picture_count": int,
            "chunk_count": int,
            "ocr_used": bool,
            "ocr_reason": str
        }
    """
    if config is None:
        config = IntegratedParserConfig()

    if source_id is None:
        source_id = str(uuid.uuid4())

    # Step 1: Docling 처리
    doc_dict, table_count, picture_count, ocr_result = process_pdf_integrated(
        file_content=file_content,
        filename=filename,
        output_dir=output_dir,
        config=config
    )

    # Step 2: 청킹
    chunks = chunk_integrated_json(
        doc_dict=doc_dict,
        source_name=filename,
        source_id=source_id,
        config=config
    )

    # Step 3: 청크 JSON 저장 (디버깅용)
    if config.save_debug_json:
        stem = Path(filename).stem
        chunks_json_path = output_dir / f"{stem}_chunks.json"
        try:
            with open(chunks_json_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            log.info(f"Chunks JSON saved: {chunks_json_path}")
        except Exception as e:
            log.warning(f"Failed to save chunks JSON: {e}")

    # 메타데이터
    metadata = {
        "table_count": table_count,
        "picture_count": picture_count,
        "chunk_count": len(chunks),
        "ocr_used": ocr_result.needs_ocr if ocr_result else False,
        "ocr_reason": ocr_result.reason if ocr_result else "N/A",
    }

    return chunks, metadata
