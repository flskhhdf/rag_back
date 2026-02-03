# -*- coding: utf-8 -*-
"""
Docling JSON Dual Content 청킹 스크립트

Dual Content Strategy:
1. content: 검색용 깨끗한 텍스트만 (embedding 최적화)
2. content_for_llm: LLM 생성용 인라인 asset 참조 포함
3. assets: 순서대로 정렬된 asset 리스트 (각 asset에 type 포함)

이 방식은 검색 정확도와 LLM 생성 품질을 모두 최적화합니다.

청크 구조:
{
  "chunk_index": 0,
  "section_header": "섹션 제목",
  "content": "깨끗한 텍스트",
  "content_for_llm": "텍스트\n\n[IMAGE:image-0] 설명\n[TABLE:table-1] 캡션",
  "pages": [1, 2],
  "assets": [
    {
      "type": "pictures",
      "ref_norm": "#/pictures/0",
      "page_no": 1,
      "captions": [...],
      "description": "..."
    }
  ],
  "asset_summary": {
    "total_count": 1,
    "has_pictures": true,
    ...
  },
  "source_file": "document.pdf"
}
"""
import json
import logging
import re
from pathlib import Path
from typing import Any
import hashlib
from uuid import uuid5, NAMESPACE_DNS

import nltk
from nltk.tokenize import sent_tokenize

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


class DoclingDualChunker:
    """Docling JSON Dual Content 청킹 클래스"""

    def __init__(self, max_tokens: int = 400, include_descriptions: bool = True, filter_junk_images: bool = True, embed_with_assets: bool = False, min_chunk_tokens: int = 100):
        """
        Args:
            max_tokens: 텍스트 청크당 최대 토큰 수 (기본: 400, asset 추가 전)
            include_descriptions: 이미지/테이블 description을 포함할지 여부 (기본: True)
            filter_junk_images: JUNK으로 분류된 이미지를 필터링할지 여부
            embed_with_assets: content 필드에도 에셋 설명을 appendix로 추가 (기본: False)
            min_chunk_tokens: 청크 최소 토큰 수 (기본: 100, 이보다 작으면 이전 청크에 병합)
        """
        self.max_tokens = max_tokens
        self.include_descriptions = include_descriptions
        self.filter_junk_images = filter_junk_images
        self.embed_with_assets = embed_with_assets
        self.min_chunk_tokens = max(0, min_chunk_tokens)
        self.logger = logging.getLogger(__name__)

        # 제외할 label 목록
        self.exclude_labels = {
            'page_header',
            'page_footer',
        }

        # 정규식 패턴
        self.ref_re = re.compile(r"^#/(texts|groups|tables|pictures)/(\d+)$", re.I)

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

    def _generate_chunk_id(self, source_file: str, chunk_index: int, content: str) -> str:
        """
        Deterministic UUID 생성
        
        같은 source_file, chunk_index, content를 가진 청크는 항상 같은 UUID를 생성합니다.
        이를 통해:
        - 재처리 시 같은 청크는 같은 ID로 인식
        - 중복 upsert 방지 (Qdrant에서 자동으로 같은 ID는 업데이트)
        
        Args:
            source_file: 원본 파일명
            chunk_index: 청크 인덱스
            content: 청크 내용 (처음 100자만 사용)
            
        Returns:
            UUID 문자열
        """
        # content의 처음 100자만 사용 (동일성 체크용)
        content_prefix = content[:100] if content else ""
        
        # 고유 식별자 조합
        unique_str = f"{source_file}|{chunk_index}|{content_prefix}"
        
        # UUID5 생성 (Deterministic)
        chunk_uuid = uuid5(NAMESPACE_DNS, unique_str)
        
        return str(chunk_uuid)

    def _generate_section_id(self, source_file: str, section_header: str) -> str:
        """
        Deterministic Section ID 생성
        
        같은 source_file, section_header를 가진 섹션은 항상 같은 UUID를 생성합니다.
        이를 통해:
        - 같은 섹션의 모든 청크를 쉽게 그룹화
        - 섹션 단위 검색/필터링 가능
        
        Args:
            source_file: 원본 파일명
            section_header: 섹션 제목
            
        Returns:
            Section UUID 문자열
        """
        # 고유 식별자 조합 (source_file + section_header)
        unique_str = f"{source_file}|section|{section_header}"
        
        # UUID5 생성 (Deterministic)
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

    def _docling_table_to_markdown(self, table_data: dict[str, Any]) -> str:
        """Docling table_cells 메타데이터를 Markdown 테이블로 변환"""
        cells = table_data.get("table_cells")
        if not cells:
            return ""

        data_list: list[dict[str, Any]] = []
        max_row, max_col = 0, 0

        for cell in cells:
            r_start = cell.get("start_row_offset_idx", 0)
            c_start = cell.get("start_col_offset_idx", 0)

            text = (cell.get("text") or "").replace('\n', ' ').strip()
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

        markdown_lines = []
        header = table_array[0]
        markdown_lines.append("| " + " | ".join(header) + " |")

        if max_row >= 1:
            markdown_lines.append("|" + "---|" * len(header))
            for row in table_array[1:]:
                markdown_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(markdown_lines)

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
        """
        모든 에셋 메타데이터 수집 + 순서 정보

        Returns:
            (ref별 에셋 메타데이터, ref별 순서) 튜플
        """
        all_assets: dict[str, dict[str, Any]] = {}
        asset_order: dict[str, int] = {}  # ref -> 순서 인덱스
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

            # Formula 체크 (texts에서)
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
                md_table = self._docling_table_to_markdown(tbl_data)
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
        pending_assets: list[tuple[str, int, str]] = []  # (ref, page_no, asset_text)

        # body children 순회
        body = data.get('body', {})
        children = body.get('children', [])

        for child_ref in children:
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
                    asset_text_parts = []
                    
                    # Description 추가
                    if self.include_descriptions and asset.get("description"):
                        asset_text_parts.append(f"[TABLE:table-{ref.split('/')[-1]}] {asset['description']}")
                    
                    # Caption 추가
                    for cap in asset.get("captions", []):
                        asset_text_parts.append(f"[TABLE Caption: {cap['text']}]")
                    
                    if asset_text_parts:
                        asset_text = "\n".join(asset_text_parts)
                        pending_assets.append((ref, page_no, asset_text))
                    else:
                        # 텍스트 없어도 ref는 추가
                        pending_assets.append((ref, page_no, ""))
                continue

            # 이미지 처리
            if ref.startswith('#/pictures/'):
                if ref in all_assets:
                    asset = all_assets[ref]
                    asset_text_parts = []
                    
                    # Description 추가
                    if self.include_descriptions and asset.get("description"):
                        asset_text_parts.append(f"[IMAGE:image-{ref.split('/')[-1]}] {asset['description']}")
                    
                    # Caption 추가
                    for cap in asset.get("captions", []):
                        asset_text_parts.append(f"[IMAGE Caption: {cap['text']}]")
                    
                    if asset_text_parts:
                        asset_text = "\n".join(asset_text_parts)
                        pending_assets.append((ref, page_no, asset_text))
                    else:
                        pending_assets.append((ref, page_no, ""))
                continue

            # Formula 처리
            if label == 'formula':
                if ref in all_assets:
                    asset = all_assets[ref]
                    formula_text = asset.get("formula", "")
                    if formula_text:
                        asset_text = f"[FORMULA:formula-{ref.split('/')[-1]}] {formula_text}"
                        pending_assets.append((ref, page_no, asset_text))
                    else:
                        pending_assets.append((ref, page_no, ""))
                continue

            # 텍스트 노드 처리
            text = item.get('text', '').strip()
            if not text:
                continue

            # pending_assets를 현재 텍스트 앞에 추가
            for asset_ref, asset_page, asset_text in pending_assets:
                current_chunk["asset_refs"].add(asset_ref)
                current_chunk["pages"].add(asset_page)
                
                if asset_text:
                    # content_with_asset에만 추가
                    current_chunk['content_with_asset'] += asset_text + "\n\n"
            
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
                                "content": sentence,
                                "content_with_asset": sentence,
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

                        current_chunk['content'] += sentence + " "
                        current_chunk['content_with_asset'] += sentence + " "
                        current_chunk['content_token_count'] += sentence_tokens
                        current_chunk["pages"].add(page_no)

                # 현재 청크에 추가하면 max_tokens 초과
                elif current_chunk['content'].strip() and current_chunk['content_token_count'] + estimated_tokens > self.max_tokens:
                    self._save_text_chunk(chunks, current_chunk, section_chunk_indices)
                    current_chunk = {
                        "section_header": current_chunk['section_header'],
                        "content": text + " ",
                        "content_with_asset": text + " ",
                        "content_token_count": estimated_tokens,
                        "pages": {page_no},
                        "asset_refs": set(),
                    }

                # 현재 청크에 텍스트 추가
                else:
                    current_chunk['content'] += text + " "
                    current_chunk['content_with_asset'] += text + " "
                    current_chunk['content_token_count'] += estimated_tokens
                    current_chunk["pages"].add(page_no)

        # 남은 pending_assets 처리
        for asset_ref, asset_page, asset_text in pending_assets:
            current_chunk["asset_refs"].add(asset_ref)
            current_chunk["pages"].add(asset_page)
            
            if asset_text:
                current_chunk['content_with_asset'] += asset_text + "\n\n"

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
            "asset_refs": current_chunk['asset_refs'],  # Set 그대로 저장 (Step 2에서 사용)
        })

    def _create_dual_content(self, chunks: list[dict], all_assets: dict[str, dict[str, Any]]):
        """
        Step 2: Dual Content 완성

        1. content: 깨끗한 텍스트만 (검색 최적화) - 이미 완료
           - embed_with_assets=True이면 appendix로 에셋 추가
        2. content_for_llm: inline 에셋 포함 (LLM 생성 최적화) - content_with_asset을 rename
        3. assets: 에셋 메타데이터 리스트 생성
        """
        for chunk in chunks:
            asset_refs = chunk.pop('asset_refs', set())
            content_with_asset = chunk.pop('content_with_asset', chunk['content'])
            clean_content = chunk['content']  # 순수 텍스트만

            # content_for_llm으로 rename
            chunk['content_for_llm'] = content_with_asset

            if not asset_refs:
                # 에셋 없음
                chunk['assets'] = []
                chunk['asset_summary'] = self._create_asset_summary([], [], [])
                continue

            # 에셋을 순서대로 정렬
            sorted_refs = sorted(asset_refs, key=lambda ref: all_assets.get(ref, {}).get('_order', 999))

            # assets 생성 + appendix 마커 생성
            assets = []
            appendix_markers = []  # content에 추가할 appendix 마커

            for ref in sorted_refs:
                if ref not in all_assets:
                    continue

                asset = all_assets[ref].copy()
                asset_type = asset.pop("_type", "pictures")
                asset.pop("_order", 0)

                # assets에 추가 (type 포함)
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

            # asset_summary: assets 리스트에서 타입별 카운트
            tables = [a for a in assets if a.get('type') == 'tables']
            pictures = [a for a in assets if a.get('type') == 'pictures']
            formulas = [a for a in assets if a.get('type') == 'formulas']
            chunk['asset_summary'] = self._create_asset_summary(tables, pictures, formulas)

    def chunk(self, json_path: Path) -> tuple[list[dict], str]:
        """
        Dual Content 청킹 실행

        Args:
            json_path: Docling JSON 파일 경로

        Returns:
            (청크 리스트, 원본 파일명) 튜플
        """
        self.logger.info(f"Dual Content 청킹 시작: {json_path.name}")

        # JSON 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 원본 파일명 추출
        origin = data.get('origin', {})
        source_filename = origin.get('filename', json_path.name)

        # Description 가용성 체크
        if self.include_descriptions:
            pictures = data.get("pictures", [])
            tables = data.get("tables", [])

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
                    f"docling 변환 시 --advanced 옵션을 사용했는지 확인하세요. "
                    f"(이미지: {image_desc_count}/{len(pictures)}, 테이블: {table_desc_count}/{len(tables)})"
                )

        # 캡션 수집
        captions_by_ref = self._build_asset_captions(data)

        # 에셋 메타데이터 구축
        self.logger.info("Step 0: 에셋 메타데이터 수집 중...")
        all_assets, asset_order = self._build_asset_metadata(data, captions_by_ref)
        self.logger.info(f"  에셋 총 {len(all_assets)}개 발견")

        # Step 1: 텍스트만으로 청킹
        self.logger.info("Step 1: 텍스트 청킹 중...")
        chunks = self._chunk_text_only(data, all_assets)
        self.logger.info(f"  {len(chunks)}개 텍스트 청크 생성")

        # Step 2: Dual Content 생성
        self.logger.info("Step 2: Dual Content 생성 중...")
        self._create_dual_content(chunks, all_assets)
        self.logger.info(f"  Dual Content 생성 완료")

        self.logger.info(f"✅ Dual Content 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks, source_filename

    def chunk_and_save(
        self,
        json_path: Path,
        output_path: Path | None = None,
        indent: int = 2,
    ) -> Path:
        """
        JSON 파일을 청킹하고 결과 저장

        Args:
            json_path: 입력 Docling JSON 파일 경로
            output_path: 출력 파일 경로 (기본: 입력파일명_chunk.json)
            indent: JSON 들여쓰기

        Returns:
            생성된 청크 JSON 파일 경로
        """
        chunks, source_filename = self.chunk(json_path)

        # source_file + chunk_id + section_id 메타데이터 추가
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



        if output_path is None:
            output_path = json_path.parent / f"{json_path.stem}_chunk.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=indent)

        self.logger.info(f"청크 저장 완료: {output_path}")
        return output_path

    @staticmethod
    def to_qdrant_format(chunks: list[dict]) -> list[dict]:
        """
        청크를 Qdrant upsert 형식으로 변환

        검색용으로는 content를 사용하고,
        content_for_llm과 assets는 metadata에 저장됩니다.
        """
        qdrant_docs = []

        for chunk in chunks:
            # 검색용: 깨끗한 content 사용
            page_content = chunk.get('content', '')

            metadata = {
                'chunk_index': chunk.get('chunk_index'),
                'section_header': chunk.get('section_header'),
                'pages': chunk.get('pages', []),
                'asset_summary': chunk.get('asset_summary', {}),
                'assets': chunk.get('assets', []),
                'source_file': chunk.get('source_file'),
                'content_for_llm': chunk.get('content_for_llm', ''),
            }

            qdrant_docs.append({
                'page_content': page_content,
                'metadata': metadata,
            })

        return qdrant_docs


def main():
    """사용 예제"""
    import argparse

    parser = argparse.ArgumentParser(description="Docling JSON Dual Content 청킹")
    parser.add_argument("input_json", type=Path, help="입력 Docling JSON 파일 경로")
    parser.add_argument(
        "-o", "--output", type=Path, help="출력 파일 경로 (기본: 입력파일명_chunk.json)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="텍스트 청크당 최대 토큰 수 (기본: 400, asset 추가 전)",
    )
    parser.add_argument(
        "--no-include-desc",
        action="store_false",
        dest="include_desc",
        help="이미지/테이블 description을 청크에서 제외",
    )
    parser.add_argument(
        "--embed-with-assets",
        action="store_true",
        help="content 필드에도 에셋 설명을 appendix로 추가 (검색 임베딩에 에셋 포함)",
    )
    parser.add_argument(
        "--include-junk",
        action="store_true",
        help="JUNK으로 분류된 이미지도 포함",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=100,
        help="청크 최소 토큰 수 (기본: 100, 이보다 작으면 이전 청크에 병합)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 로그 출력")

    args = parser.parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # 청킹 실행
    chunker = DoclingDualChunker(
        max_tokens=args.max_tokens,
        include_descriptions=args.include_desc,
        filter_junk_images=not args.include_junk,
        embed_with_assets=args.embed_with_assets,
        min_chunk_tokens=args.min_chunk_tokens,
    )

    try:
        output_path = chunker.chunk_and_save(args.input_json, args.output)
        print(f"✅ Dual Content 청킹 완료: {output_path}")

        # 통계 출력
        with open(output_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"   총 청크 수: {len(chunks)}개")

        # 에셋 통계
        total_assets = sum(len(c.get('assets', [])) for c in chunks)
        total_tables = sum(len([a for a in c.get('assets', []) if a.get('type') == 'tables']) for c in chunks)
        total_pictures = sum(len([a for a in c.get('assets', []) if a.get('type') == 'pictures']) for c in chunks)
        total_formulas = sum(len([a for a in c.get('assets', []) if a.get('type') == 'formulas']) for c in chunks)

        if total_assets > 0:
            print(f"   에셋: 총 {total_assets}개 (테이블 {total_tables}개, 이미지 {total_pictures}개, 수식 {total_formulas}개)")

    except Exception as e:
        print(f"❌ 청킹 실패: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
