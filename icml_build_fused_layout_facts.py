"""Build evidence-oriented layout facts from a PDF and MinerU output.

PyMuPDF owns numeric PDF facts (coordinates, fonts, baselines, drawings),
while MinerU supplies semantic roles (headings, captions, tables, references).
The output keeps both sources and records which source won each fused field.

Examples:
    python build_fused_layout_facts.py \
        --pdf li22n.pdf \
        --mineru-json li22n_minerU_result.json \
        --output li22n.fused_facts.json

    # Without --mineru-json the script calls MinerU. The token is read only
    # from MINERU_API_KEY or MINERU_TOKEN; it is never persisted.
    $env:MINERU_API_KEY = "..."
    python build_fused_layout_facts.py --pdf paper.pdf --output paper.fused.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "fused_layout_facts_v1"
DEFAULT_MINERU_BASE_URL = "https://mineru.net/api/v4"
KNOWN_MINERU_TYPES = {
    "title",
    "text",
    "ref_text",
    "image",
    "chart",
    "table",
    "image_caption",
    "chart_caption",
    "table_caption",
    "interline_equation",
    "equation",
    "code",
    "page_footnote",
    "header",
    "page_number",
}
SUBSET_FONT_PREFIX = re.compile(r"^[A-Z]{6}\+")
APPENDIX_HEADING = re.compile(r"^(?:Appendix\b|[A-Z]\.\s+[A-Z])", re.IGNORECASE)
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,2})\.?\s+\D")
FIGURE_CAPTION = re.compile(r"^(?:Figure|Fig\.)\s*(\d+)\s*[.:]", re.IGNORECASE)
TABLE_CAPTION = re.compile(r"^Table\s*(\d+)\s*[.:]", re.IGNORECASE)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
        return None
    x0, y0, x1, y1 = (float(item) for item in value)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _bbox_union(values: Iterable[list[float] | None]) -> list[float] | None:
    boxes = [item for item in values if _bbox(item) is not None]
    if not boxes:
        return None
    return [
        min(item[0] for item in boxes),
        min(item[1] for item in boxes),
        max(item[2] for item in boxes),
        max(item[3] for item in boxes),
    ]


def _intersection_ratio(inner: list[float] | None, outer: list[float] | None) -> float:
    if _bbox(inner) is None or _bbox(outer) is None:
        return 0.0
    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return intersection / area if area else 0.0


def _bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if _bbox(left) is None or _bbox(right) is None:
        return 0.0
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_inside_page(
    bbox: list[float] | None, page: dict[str, Any] | None, tolerance: float = 2.0
) -> bool:
    if bbox is None or page is None:
        return False
    return (
        bbox[0] >= -tolerance
        and bbox[1] >= -tolerance
        and bbox[2] <= float(page["width_pt"]) + tolerance
        and bbox[3] <= float(page["height_pt"]) + tolerance
    )


def _bbox_area(bbox: list[float] | None) -> float:
    if _bbox(bbox) is None:
        return 0.0
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _horizontal_overlap(left: list[float], right: list[float]) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    minimum = min(left[2] - left[0], right[2] - right[0])
    return overlap / minimum if minimum > 0 else 0.0


def _expand_bbox(value: list[float], margin: float) -> list[float]:
    return [value[0] - margin, value[1] - margin, value[2] + margin, value[3] + margin]


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _join_visual_line(parts: list[tuple[str, list[float]]]) -> str:
    """Restore spaces omitted between separately positioned PDF spans."""

    result = ""
    previous_bbox: list[float] | None = None
    for text, bbox in parts:
        if not text:
            continue
        gap = bbox[0] - previous_bbox[2] if previous_bbox is not None else 0.0
        if result and not result[-1].isspace() and not text[0].isspace() and gap >= 1.0:
            result += " "
        result += text
        previous_bbox = bbox
    return result.strip()


def _text_key(value: Any) -> str:
    return re.sub(r"[^\w]+", "", _normalize_text(value).casefold(), flags=re.UNICODE)


def _text_similarity(left: str, right: str) -> float:
    a, b = _text_key(left), _text_key(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return SequenceMatcher(None, a[:4000], b[:4000]).ratio()


def _weighted_median(values: list[tuple[float, int]]) -> float | None:
    values = sorted((value, max(1, weight)) for value, weight in values)
    if not values:
        return None
    half = sum(weight for _, weight in values) / 2
    total = 0
    for value, weight in values:
        total += weight
        if total >= half:
            return float(value)
    return float(values[-1][0])


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def extract_pymupdf(pdf_path: Path) -> dict[str, Any]:
    """Extract full native facts; no semantic guesses are made here."""

    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("PyMuPDF is required: pip install pymupdf") from exc

    document = fitz.open(pdf_path)
    if document.needs_pass:
        document.close()
        raise RuntimeError("The PDF is encrypted and cannot be parsed")

    pages: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    text_blocks: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    drawings: list[dict[str, Any]] = []

    try:
        for page_number, page in enumerate(document, start=1):
            page_rect = page.rect
            page_width, page_height = float(page_rect.width), float(page_rect.height)
            pages.append(
                {
                    "page_number": page_number,
                    "width_pt": page_width,
                    "height_pt": page_height,
                    "rotation": int(page.rotation or 0),
                }
            )
            raw = page.get_text("dict")
            for native_block_index, block in enumerate(raw.get("blocks", [])):
                if int(block.get("type", 0)) != 0:
                    continue
                block_line_ids: list[str] = []
                block_text: list[str] = []
                for native_line_index, line in enumerate(block.get("lines", [])):
                    line_id = f"p{page_number}-l{native_block_index}-{native_line_index}"
                    line_span_ids: list[str] = []
                    line_parts: list[tuple[str, list[float]]] = []
                    baselines: list[float] = []
                    for native_span_index, span in enumerate(line.get("spans", [])):
                        text = str(span.get("text") or "")
                        span_bbox = _bbox(span.get("bbox"))
                        if not text.strip() or span_bbox is None:
                            continue
                        span_id = (
                            f"p{page_number}-s{native_block_index}-{native_line_index}-"
                            f"{native_span_index}"
                        )
                        raw_font = str(span.get("font") or "") or None
                        flags = int(span.get("flags") or 0)
                        origin = span.get("origin")
                        baseline = (
                            float(origin[1])
                            if isinstance(origin, (list, tuple)) and len(origin) == 2
                            else None
                        )
                        if baseline is not None:
                            baselines.append(baseline)
                        spans.append(
                            {
                                "span_id": span_id,
                                "line_id": line_id,
                                "page_number": page_number,
                                "text": text,
                                "bbox": span_bbox,
                                "baseline_y": baseline,
                                "raw_font_name": raw_font,
                                "font_name": SUBSET_FONT_PREFIX.sub("", raw_font or "") or None,
                                "font_size_pt": float(span.get("size") or 0.0),
                                "font_flags": flags,
                                "is_bold": "bold" in (raw_font or "").lower() or bool(flags & 16),
                                "is_italic": bool(flags & 2) or any(
                                    token in (raw_font or "").lower()
                                    for token in ("italic", "oblique", "slant")
                                ),
                                "color": int(span.get("color") or 0),
                                "source": "pymupdf_text_layer",
                            }
                        )
                        line_span_ids.append(span_id)
                        line_parts.append((text, span_bbox))
                    if not line_span_ids:
                        continue
                    line_bbox = _bbox(line.get("bbox")) or _bbox_union(
                        span["bbox"] for span in spans if span["span_id"] in line_span_ids
                    )
                    line_text = _join_visual_line(line_parts)
                    lines.append(
                        {
                            "line_id": line_id,
                            "page_number": page_number,
                            "text": line_text,
                            "bbox": line_bbox,
                            "baseline_y": statistics.median(baselines) if baselines else None,
                            "direction": list(line.get("dir") or (1.0, 0.0)),
                            "span_ids": line_span_ids,
                        }
                    )
                    block_line_ids.append(line_id)
                    block_text.append(line_text)
                if block_line_ids:
                    text_blocks.append(
                        {
                            "native_block_id": f"p{page_number}-b{native_block_index}",
                            "page_number": page_number,
                            "text": "\n".join(block_text),
                            "bbox": _bbox(block.get("bbox")),
                            "line_ids": block_line_ids,
                        }
                    )

            for image_index, image in enumerate(page.get_images(full=True), start=1):
                xref = int(image[0])
                for rect_index, rect in enumerate(page.get_image_rects(xref), start=1):
                    image_bbox = _bbox((rect.x0, rect.y0, rect.x1, rect.y1))
                    if image_bbox is None:
                        continue
                    objects.append(
                        {
                            "object_id": f"p{page_number}-image-{xref}-{image_index}-{rect_index}",
                            "object_type": "image",
                            "page_number": page_number,
                            "bbox": image_bbox,
                            "width_px": int(image[2]),
                            "height_px": int(image[3]),
                            "source": "pymupdf_image_object",
                        }
                    )

            for drawing_index, drawing in enumerate(page.get_drawings(), start=1):
                drawing_id = f"p{page_number}-drawing-{drawing_index}"
                drawing_bbox = None
                if drawing.get("rect") is not None:
                    rect = drawing["rect"]
                    drawing_bbox = _bbox((rect.x0, rect.y0, rect.x1, rect.y1))
                drawing_record = {
                    "drawing_id": drawing_id,
                    "page_number": page_number,
                    "bbox": drawing_bbox,
                    "width_pt": float(drawing.get("width") or 0.0),
                    "stroke_color": list(drawing.get("color")) if drawing.get("color") else None,
                    "fill_color": list(drawing.get("fill")) if drawing.get("fill") else None,
                    "items": [],
                    "source": "pymupdf_vector_drawing",
                }
                for item_index, item in enumerate(drawing.get("items", []), start=1):
                    if not item:
                        continue
                    item_type = str(item[0])
                    record: dict[str, Any] = {
                        "item_id": f"{drawing_id}-item-{item_index}",
                        "item_type": item_type,
                    }
                    if item_type == "l" and len(item) >= 3:
                        start, end = item[1], item[2]
                        record.update(
                            {
                                "start": [float(start.x), float(start.y)],
                                "end": [float(end.x), float(end.y)],
                                "width_pt": float(drawing.get("width") or 0.0),
                            }
                        )
                    elif item_type == "re" and len(item) >= 2:
                        rect = item[1]
                        record["bbox"] = _bbox((rect.x0, rect.y0, rect.x1, rect.y1))
                    drawing_record["items"].append(record)
                drawings.append(drawing_record)
                if drawing_bbox is not None:
                    objects.append(
                        {
                            "object_id": drawing_id,
                            "object_type": "vector_drawing",
                            "page_number": page_number,
                            "bbox": drawing_bbox,
                            "width_pt": float(drawing.get("width") or 0.0),
                            "source": "pymupdf_vector_drawing",
                        }
                    )
    finally:
        document.close()

    return {
        "pages": pages,
        "spans": spans,
        "lines": lines,
        "text_blocks": text_blocks,
        "objects": objects,
        "drawings": drawings,
    }


def _node_text(node: dict[str, Any]) -> str:
    direct = node.get("text")
    if isinstance(direct, str) and direct.strip():
        return _normalize_text(direct)
    parts: list[str] = []
    for line in node.get("lines", []) if isinstance(node.get("lines"), list) else []:
        if not isinstance(line, dict):
            continue
        for span in line.get("spans", []) if isinstance(line.get("spans"), list) else []:
            if isinstance(span, dict) and isinstance(span.get("content"), str):
                parts.append(str(span["content"]))
    return _normalize_text(" ".join(parts))


def _semantic_role(item_type: str, text: str, level: int | None, page: int) -> str:
    if FIGURE_CAPTION.match(text):
        return "figure_caption"
    if TABLE_CAPTION.match(text):
        return "table_caption"
    if item_type in {"image_caption", "chart_caption"}:
        return "figure_caption"
    if item_type == "table_caption":
        return "table_caption"
    if item_type == "title":
        return "document_title" if page == 1 and level == 1 else "section_heading"
    if item_type == "ref_text":
        return "reference_entry"
    if item_type in {"image", "chart"}:
        return "figure_object"
    if item_type == "table":
        return "table_object"
    if item_type in {"equation", "interline_equation"}:
        return "display_formula"
    if item_type in {"page_number", "header", "page_footnote"}:
        return item_type
    return "paragraph"


def parse_mineru(payload: Any, pymupdf_pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten MinerU layout/content-list variants into semantic blocks."""

    page_dimensions = {int(page["page_number"]): page for page in pymupdf_pages}
    blocks: list[dict[str, Any]] = []
    source_pages: dict[int, tuple[float, float]] = {}

    def emit(node: dict[str, Any], page: int, path: str, parent_id: str | None = None) -> None:
        item_type = str(node.get("type") or "unknown")
        if item_type not in KNOWN_MINERU_TYPES:
            return
        text = _node_text(node)
        level_value = node.get("level", node.get("text_level"))
        level = int(level_value) if isinstance(level_value, (int, float)) else None
        raw_bbox = _bbox(node.get("bbox"))
        source_size = source_pages.get(page)
        target = page_dimensions.get(page)
        normalized_bbox = raw_bbox
        scale = [1.0, 1.0]
        if raw_bbox and source_size and target and source_size[0] > 0 and source_size[1] > 0:
            scale = [float(target["width_pt"]) / source_size[0], float(target["height_pt"]) / source_size[1]]
            normalized_bbox = [
                raw_bbox[0] * scale[0],
                raw_bbox[1] * scale[1],
                raw_bbox[2] * scale[0],
                raw_bbox[3] * scale[1],
            ]
        role = _semantic_role(item_type, text, level, page)
        block_id = f"mineru-p{page}-{path.replace('.', '-')}-{item_type}"
        blocks.append(
            {
                "mineru_block_id": block_id,
                "parent_mineru_block_id": parent_id,
                "page_number": page,
                "mineru_type": item_type,
                "semantic_role": role,
                "heading_level": level,
                "text": text,
                "bbox": normalized_bbox,
                "raw_bbox": raw_bbox,
                "coordinate_scale": scale,
                "source": "mineru",
            }
        )
        children = node.get("blocks")
        if isinstance(children, list):
            for index, child in enumerate(children):
                if isinstance(child, dict):
                    emit(child, page, f"{path}.{index}", block_id)

    if isinstance(payload, dict) and isinstance(payload.get("pdf_info"), list):
        for page_data in payload["pdf_info"]:
            if not isinstance(page_data, dict):
                continue
            page = int(page_data.get("page_idx", 0)) + 1
            size = page_data.get("page_size")
            if isinstance(size, (list, tuple)) and len(size) == 2:
                source_pages[page] = (float(size[0]), float(size[1]))
        for page_data in payload["pdf_info"]:
            if not isinstance(page_data, dict):
                continue
            page = int(page_data.get("page_idx", 0)) + 1
            raw_blocks = page_data.get("para_blocks") or page_data.get("preproc_blocks") or []
            for index, block in enumerate(raw_blocks):
                if isinstance(block, dict):
                    emit(block, page, str(index))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            page = int(item.get("page_idx", 0)) + 1
            target = page_dimensions.get(page)
            if target:
                source_pages.setdefault(page, (float(target["width_pt"]), float(target["height_pt"])))
            emit(item, page, str(index))
    else:
        raise ValueError("Unsupported MinerU JSON: expected pdf_info or a content-list array")

    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for block in blocks:
        box = block.get("bbox") or []
        key = (
            block["page_number"],
            block["mineru_type"],
            tuple(round(float(value), 2) for value in box),
            _text_key(block.get("text")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(block)
    return {"blocks": unique, "source_page_sizes": source_pages}


def _line_style(line: dict[str, Any], spans_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = [spans_by_id[item] for item in line.get("span_ids", []) if item in spans_by_id]
    sizes = [(float(span["font_size_pt"]), len(str(span["text"]))) for span in selected]
    font_weights: Counter[str] = Counter()
    total_chars = 0
    bold_chars = 0
    italic_chars = 0
    for span in selected:
        weight = max(1, len(str(span.get("text") or "")))
        total_chars += weight
        if span.get("font_name"):
            font_weights[str(span["font_name"])] += weight
        if span.get("is_bold"):
            bold_chars += weight
        if span.get("is_italic"):
            italic_chars += weight
    return {
        "font_size_median_pt": _weighted_median(sizes),
        "dominant_font": font_weights.most_common(1)[0][0] if font_weights else None,
        "bold_character_ratio": bold_chars / total_chars if total_chars else None,
        "italic_character_ratio": italic_chars / total_chars if total_chars else None,
    }


def _aggregate_style(
    selected_lines: list[dict[str, Any]], spans_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    span_ids = [item for line in selected_lines for item in line.get("span_ids", [])]
    selected = [spans_by_id[item] for item in span_ids if item in spans_by_id]
    sizes = [(float(span["font_size_pt"]), len(str(span["text"]))) for span in selected]
    font_weights: Counter[str] = Counter()
    total_chars = bold_chars = italic_chars = 0
    for span in selected:
        weight = max(1, len(str(span.get("text") or "")))
        total_chars += weight
        if span.get("font_name"):
            font_weights[str(span["font_name"])] += weight
        bold_chars += weight if span.get("is_bold") else 0
        italic_chars += weight if span.get("is_italic") else 0
    baselines = sorted(
        float(line["baseline_y"])
        for line in selected_lines
        if isinstance(line.get("baseline_y"), (int, float))
    )
    gaps = [right - left for left, right in zip(baselines, baselines[1:]) if 5 <= right - left <= 30]
    return {
        "dominant_font": font_weights.most_common(1)[0][0] if font_weights else None,
        "font_size_median_pt": _weighted_median(sizes),
        "bold_character_ratio": bold_chars / total_chars if total_chars else None,
        "italic_character_ratio": italic_chars / total_chars if total_chars else None,
        "baseline_gap_median_pt": statistics.median(gaps) if gaps else None,
    }


def fuse_blocks(native: dict[str, Any], mineru: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lines_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in native["lines"]:
        lines_by_page[int(line["page_number"])].append(line)
    native_blocks_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for block in native["text_blocks"]:
        native_blocks_by_page[int(block["page_number"])].append(block)
    spans_by_id = {str(span["span_id"]): span for span in native["spans"]}
    pages_by_number = {int(page["page_number"]): page for page in native["pages"]}
    objects_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for obj in native["objects"]:
        objects_by_page[int(obj["page_number"])].append(obj)

    fused: list[dict[str, Any]] = []
    matched = low_confidence = unmatched = 0
    bbox_rejected = bbox_recovered = bbox_unresolved = 0
    for index, semantic in enumerate(mineru["blocks"], start=1):
        page = int(semantic["page_number"])
        semantic_bbox = _bbox(semantic.get("bbox"))
        semantic_text = str(semantic.get("text") or "")
        semantic_role = str(semantic.get("semantic_role") or "")
        is_object_role = semantic_role in {"figure_object", "table_object"}
        spatial_lines: list[dict[str, Any]] = []
        spatial_score = 0.0
        spatial_text_score = 0.0
        geometry_valid = _bbox_inside_page(semantic_bbox, pages_by_number.get(page))
        rejection_reasons: list[str] = []
        if semantic_bbox is None:
            rejection_reasons.append("missing_or_invalid_bbox")
        elif not geometry_valid:
            rejection_reasons.append("bbox_outside_page")

        native_object_bbox = None
        if is_object_role and semantic_bbox and geometry_valid:
            object_candidates = []
            for obj in objects_by_page.get(page, []):
                object_bbox = _bbox(obj.get("bbox"))
                if object_bbox is None:
                    continue
                object_coverage = _intersection_ratio(object_bbox, semantic_bbox)
                mineru_coverage = _intersection_ratio(semantic_bbox, object_bbox)
                if max(object_coverage, mineru_coverage) >= 0.35:
                    object_candidates.append(
                        (max(object_coverage, mineru_coverage), _bbox_area(object_bbox), object_bbox)
                    )
            if object_candidates:
                native_object_bbox = max(object_candidates, key=lambda item: (item[0], item[1]))[2]

        if semantic_bbox and geometry_valid:
            expanded = _expand_bbox(semantic_bbox, 3.0)
            spatial_lines = [
                line
                for line in lines_by_page.get(page, [])
                if _intersection_ratio(line.get("bbox"), expanded) >= 0.45
            ]
            spatial_lines.sort(
                key=lambda item: (
                    float((item.get("bbox") or [0, 0, 0, 0])[1]),
                    float((item.get("bbox") or [0, 0, 0, 0])[0]),
                )
            )
            if spatial_lines:
                spatial_score = sum(
                    _intersection_ratio(line.get("bbox"), expanded) for line in spatial_lines
                ) / len(spatial_lines)
                spatial_text = " ".join(
                    str(line.get("text") or "") for line in spatial_lines
                )
                spatial_text_score = (
                    _text_similarity(semantic_text, spatial_text) if semantic_text else 0.0
                )

        text_candidate_lines: list[dict[str, Any]] = []
        recovered_text_score = 0.0
        if semantic_text:
            candidates = sorted(
                (
                    (_text_similarity(semantic_text, str(block.get("text") or "")), block)
                    for block in native_blocks_by_page.get(page, [])
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if candidates:
                recovered_text_score, candidate = candidates[0]
                wanted = set(candidate.get("line_ids", []))
                text_candidate_lines = [
                    line for line in lines_by_page.get(page, []) if line["line_id"] in wanted
                ]
        text_candidate_bbox = _bbox_union(line.get("bbox") for line in text_candidate_lines)
        recovery_is_clear_conflict = (
            bool(text_candidate_lines)
            and recovered_text_score >= 0.65
            and recovered_text_score - spatial_text_score >= 0.15
            and _bbox_iou(semantic_bbox, text_candidate_bbox) < 0.1
        )

        selected_lines: list[dict[str, Any]] = []
        location_method = "none"
        bbox_status = "unverifiable"
        if spatial_lines:
            selected_lines = spatial_lines
            location_method = "mineru_bbox_cross_checked_by_pymupdf"
            if not semantic_text:
                bbox_status = (
                    "accepted_cross_checked_native_object"
                    if native_object_bbox
                    else "content_envelope_no_text_cross_check"
                )
            elif recovery_is_clear_conflict:
                selected_lines = text_candidate_lines
                location_method = "pymupdf_text_recovery"
                bbox_status = "rejected_used_pymupdf"
                rejection_reasons.append("mineru_bbox_text_conflict")
            elif spatial_text_score >= 0.45:
                bbox_status = "accepted_cross_checked"
            elif (
                text_candidate_lines
                and recovered_text_score >= 0.55
                and recovered_text_score - spatial_text_score >= 0.15
            ):
                selected_lines = text_candidate_lines
                location_method = "pymupdf_text_recovery"
                bbox_status = "rejected_used_pymupdf"
                rejection_reasons.append("mineru_bbox_text_conflict")
            else:
                selected_lines = []
                location_method = "none"
                bbox_status = "unresolved_text_mismatch"
                rejection_reasons.append("text_in_bbox_does_not_match")
        elif text_candidate_lines and recovered_text_score >= 0.55:
            selected_lines = text_candidate_lines
            location_method = "pymupdf_text_recovery"
            bbox_status = "rejected_used_pymupdf"
            rejection_reasons.append("no_matching_text_at_mineru_bbox")
        elif semantic_text:
            bbox_status = "unresolved_no_pymupdf_match"
            rejection_reasons.append("no_reliable_pymupdf_text_match")

        if is_object_role and native_object_bbox and not semantic_text:
            bbox_status = "accepted_cross_checked_native_object"
            location_method = "pymupdf_native_object_cross_check"

        if bbox_status == "rejected_used_pymupdf":
            bbox_rejected += 1
            bbox_recovered += 1
        elif bbox_status.startswith("unresolved"):
            bbox_rejected += 1
            bbox_unresolved += 1

        text_native_bbox = _bbox_union(line.get("bbox") for line in selected_lines)
        native_bbox = native_object_bbox or text_native_bbox
        native_text = " ".join(str(line.get("text") or "") for line in selected_lines).strip()
        text_score = (
            recovered_text_score if location_method == "pymupdf_text_recovery" else spatial_text_score
        )
        confidence = min(1.0, 0.55 * text_score + 0.45 * spatial_score)
        if location_method == "pymupdf_text_recovery":
            confidence = recovered_text_score
        if selected_lines and not semantic_text:
            confidence = max(confidence, 0.75 * spatial_score)
        if selected_lines:
            matched += 1
            if confidence < 0.65:
                low_confidence += 1
        else:
            unmatched += 1
        final_bbox = native_bbox or semantic_bbox
        final_text = native_text or semantic_text
        style = _aggregate_style(selected_lines, spans_by_id) if selected_lines else {}
        if native_object_bbox:
            selected_bbox_source = "pymupdf_native_object"
        elif text_native_bbox:
            selected_bbox_source = "pymupdf_text_geometry"
        else:
            selected_bbox_source = "mineru"
        fused.append(
            {
                "fused_block_id": f"fused-{index:05d}",
                "page_number": page,
                "semantic_role": semantic["semantic_role"],
                "heading_level": semantic.get("heading_level"),
                "text": final_text,
                "bbox": final_bbox,
                "style": style,
                "match": {
                    "confidence": confidence,
                    "text_similarity": text_score,
                    "spatial_score": spatial_score,
                    "location_method": location_method,
                    "status": (
                        "matched"
                        if selected_lines and confidence >= 0.65
                        else "low_confidence"
                        if selected_lines
                        else "unmatched"
                    ),
                },
                "bbox_validation": {
                    "mineru_bbox_status": bbox_status,
                    "geometry_valid": geometry_valid,
                    "rejection_reasons": rejection_reasons,
                    "spatial_text_similarity": spatial_text_score,
                    "pymupdf_recovery_text_similarity": recovered_text_score,
                    "mineru_pymupdf_iou": _bbox_iou(semantic_bbox, native_bbox),
                    "selected_bbox_source": selected_bbox_source,
                },
                "source_ids": {
                    "mineru_block_ids": [semantic["mineru_block_id"]],
                    "pymupdf_line_ids": [line["line_id"] for line in selected_lines],
                    "pymupdf_span_ids": [
                        span_id for line in selected_lines for span_id in line.get("span_ids", [])
                    ],
                },
                "field_provenance": {
                    "text": "pymupdf" if native_text else "mineru",
                    "semantic_role": "mineru",
                    "bbox": selected_bbox_source,
                    "style": "pymupdf" if selected_lines else None,
                },
                "alternatives": {
                    "mineru_type": semantic["mineru_type"],
                    "mineru_bbox": semantic_bbox,
                    "pymupdf_bbox": native_bbox,
                    "pymupdf_text_bbox": text_native_bbox,
                    "pymupdf_native_object_bbox": native_object_bbox,
                    "pymupdf_text": native_text or None,
                },
            }
        )
    return fused, {
        "mineru_block_count": len(mineru["blocks"]),
        "matched_block_count": matched,
        "low_confidence_block_count": low_confidence,
        "unmatched_block_count": unmatched,
        "mineru_bbox_rejected_count": bbox_rejected,
        "mineru_bbox_recovered_with_pymupdf_count": bbox_recovered,
        "mineru_bbox_unresolved_count": bbox_unresolved,
    }


def _coordinate_mode(values: list[float], bin_width: float = 2.0) -> float | None:
    if not values:
        return None
    buckets: dict[int, list[float]] = defaultdict(list)
    for value in values:
        buckets[round(value / bin_width)].append(value)
    winning = max(buckets.values(), key=lambda items: (len(items), -statistics.pstdev(items)))
    return float(statistics.median(winning))


def derive_structure(fused: list[dict[str, Any]], pages: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        (item for item in fused if item.get("bbox")),
        key=lambda item: (item["page_number"], item["bbox"][1], item["bbox"][0]),
    )
    references = next(
        (
            item
            for item in ordered
            if str(item.get("text") or "").strip().casefold() in {"references", "bibliography"}
            and item.get("semantic_role") == "section_heading"
        ),
        None,
    )
    appendix = next(
        (
            item
            for item in ordered
            if APPENDIX_HEADING.match(str(item.get("text") or "").strip())
            and item.get("semantic_role") == "section_heading"
        ),
        None,
    )
    last_page = max((int(page["page_number"]) for page in pages), default=0)
    body_end = last_page
    boundary = references or appendix
    if boundary:
        boundary_page = int(boundary["page_number"])
        boundary_y = float(boundary["bbox"][1])
        prior_on_page = any(
            item["page_number"] == boundary_page
            and float(item["bbox"][1]) < boundary_y
            and item.get("semantic_role") not in {"header", "page_number", "reference_entry"}
            for item in ordered
        )
        body_end = boundary_page if prior_on_page else max(0, boundary_page - 1)
    return {
        "page_count": last_page,
        "references_start": (
            {
                "page_number": references["page_number"],
                "bbox": references["bbox"],
                "evidence_id": references["fused_block_id"],
            }
            if references
            else None
        ),
        "appendix_start": (
            {
                "page_number": appendix["page_number"],
                "bbox": appendix["bbox"],
                "evidence_id": appendix["fused_block_id"],
            }
            if appendix
            else None
        ),
        "main_body_page_count": body_end,
    }


def derive_column_geometry(native: dict[str, Any], body_end_page: int) -> dict[str, Any] | None:
    pages_by_id = {int(page["page_number"]): page for page in native["pages"]}
    spans_by_id = {str(span["span_id"]): span for span in native["spans"]}
    candidates: list[dict[str, Any]] = []
    size_weights: Counter[float] = Counter()
    for line in native["lines"]:
        page = int(line["page_number"])
        page_meta = pages_by_id.get(page)
        box = _bbox(line.get("bbox"))
        text = _normalize_text(line.get("text"))
        if not page_meta or box is None or page > max(1, body_end_page):
            continue
        if len(_text_key(text)) < 20:
            continue
        if box[0] < page_meta["width_pt"] * 0.07 or box[2] > page_meta["width_pt"] * 0.96:
            continue
        if box[1] < page_meta["height_pt"] * 0.06 or box[3] > page_meta["height_pt"] * 0.93:
            continue
        style = _line_style(line, spans_by_id)
        size = style.get("font_size_median_pt")
        if not isinstance(size, (int, float)) or not 7.5 <= size <= 12.5:
            continue
        size_weights[round(float(size) * 2) / 2] += len(text)
        candidates.append({**line, "style": style})
    if not candidates or not size_weights:
        return None
    dominant_size = size_weights.most_common(1)[0][0]
    body_lines = [
        item
        for item in candidates
        if abs(float(item["style"]["font_size_median_pt"]) - dominant_size) <= 0.55
    ]
    if not body_lines:
        return None
    page_width = statistics.median(
        float(pages_by_id[int(item["page_number"])]["width_pt"]) for item in body_lines
    )
    left_lines = [item for item in body_lines if item["bbox"][0] < page_width * 0.45]
    right_lines = [item for item in body_lines if item["bbox"][0] > page_width * 0.50]
    if len(left_lines) < 5 or len(right_lines) < 5:
        return {
            "column_count": 1,
            "body_font_size_mode_pt": dominant_size,
            "sample_line_count": len(body_lines),
            "confidence": 0.7,
        }
    left_start = _coordinate_mode([item["bbox"][0] for item in left_lines])
    right_start = _coordinate_mode([item["bbox"][0] for item in right_lines])
    left_end = _coordinate_mode([item["bbox"][2] for item in left_lines])
    right_end = _coordinate_mode([item["bbox"][2] for item in right_lines])
    if None in {left_start, right_start, left_end, right_end}:
        return None
    text_width = right_end - left_start
    pitch = right_start - left_start
    inferred_column_width = text_width - pitch
    inferred_gutter = 2 * pitch - text_width
    return {
        "column_count": 2,
        "text_left_pt": left_start,
        "text_right_pt": right_end,
        "text_width_pt": text_width,
        "left_column_start_pt": left_start,
        "right_column_start_pt": right_start,
        "left_ink_end_pt": left_end,
        "right_ink_end_pt": right_end,
        "observed_ink_gap_pt": right_start - left_end,
        "column_width_pt": inferred_column_width,
        "column_width_in": inferred_column_width / 72,
        "gutter_pt": inferred_gutter,
        "gutter_in": inferred_gutter / 72,
        "body_font_size_mode_pt": dominant_size,
        "sample_line_count": len(body_lines),
        "sample_pages": sorted({int(item["page_number"]) for item in body_lines}),
        "derivation": "repeated native text-line boundary modes with equal-column constraint",
        "confidence": min(0.99, 0.75 + len(body_lines) / 1000),
    }


def _alignment(bbox: list[float] | None, center_x: float | None) -> str | None:
    if bbox is None or center_x is None:
        return None
    return "center" if abs((bbox[0] + bbox[2]) / 2 - center_x) <= 8 else "not_centered"


def derive_front_matter(
    fused: list[dict[str, Any]], native: dict[str, Any], columns: dict[str, Any] | None
) -> dict[str, Any]:
    first_page = [item for item in fused if int(item["page_number"]) == 1 and item.get("bbox")]
    titles = [item for item in first_page if item.get("semantic_role") == "document_title"]
    if not titles:
        titles = sorted(first_page, key=lambda item: item["bbox"][1])[:1]
    title_bbox = _bbox_union(item.get("bbox") for item in titles)
    title_text = " ".join(str(item.get("text") or "") for item in titles).strip()
    text_center = None
    if columns and columns.get("text_left_pt") is not None and columns.get("text_right_pt") is not None:
        text_center = (float(columns["text_left_pt"]) + float(columns["text_right_pt"])) / 2
    elif native["pages"]:
        text_center = float(native["pages"][0]["width_pt"]) / 2
    title_style = {}
    if titles:
        title_style = titles[0].get("style") or {}
    horizontal_rules: list[dict[str, Any]] = []
    if title_bbox:
        for drawing in native["drawings"]:
            if int(drawing["page_number"]) != 1:
                continue
            for item in drawing.get("items", []):
                if item.get("item_type") != "l":
                    continue
                start, end = item.get("start"), item.get("end")
                if not start or not end:
                    continue
                if abs(start[1] - end[1]) <= 0.5 and abs(end[0] - start[0]) >= 200:
                    if title_bbox[1] - 50 <= start[1] <= title_bbox[3] + 50:
                        horizontal_rules.append(
                            {
                                "start": start,
                                "end": end,
                                "width_pt": item.get("width_pt"),
                                "drawing_id": drawing["drawing_id"],
                            }
                        )
    abstract_heading = next(
        (
            item
            for item in first_page
            if str(item.get("text") or "").strip().casefold() == "abstract"
        ),
        None,
    )
    author_candidates = []
    if title_bbox and abstract_heading:
        author_candidates = [
            item
            for item in first_page
            if title_bbox[3] < item["bbox"][1] < abstract_heading["bbox"][1]
            and item.get("semantic_role") not in {"page_number", "header"}
            and str(item.get("text") or "").strip()
        ]
    return {
        "title": {
            "text": title_text,
            "bbox": title_bbox,
            "style": title_style,
            "alignment": _alignment(title_bbox, text_center),
            "horizontal_rules": sorted(horizontal_rules, key=lambda item: item["start"][1]),
            "evidence_ids": [item["fused_block_id"] for item in titles],
        },
        "author_blocks": [
            {
                "text": item["text"],
                "bbox": item["bbox"],
                "style": item.get("style"),
                "alignment": _alignment(item["bbox"], text_center),
                "evidence_id": item["fused_block_id"],
            }
            for item in author_candidates
        ],
    }


def derive_abstract(
    fused: list[dict[str, Any]], columns: dict[str, Any] | None
) -> dict[str, Any] | None:
    heading = next(
        (
            item
            for item in fused
            if str(item.get("text") or "").strip().casefold() == "abstract"
            and item.get("bbox")
        ),
        None,
    )
    if not heading:
        return None
    page = int(heading["page_number"])
    right_column_start = float(columns["right_column_start_pt"]) if columns else None
    heading_is_left = (
        right_column_start is not None
        and (float(heading["bbox"][0]) + float(heading["bbox"][2])) / 2 < right_column_start
    )

    def same_column(item: dict[str, Any]) -> bool:
        if right_column_start is None:
            return True
        if heading_is_left:
            return float(item["bbox"][0]) < right_column_start - 8
        return float(item["bbox"][2]) > right_column_start + 8

    following = sorted(
        (
            item
            for item in fused
            if int(item["page_number"]) == page
            and item.get("bbox")
            and item["bbox"][1] >= heading["bbox"][3]
            and item.get("semantic_role") == "paragraph"
            and same_column(item)
        ),
        key=lambda item: item["bbox"][1],
    )
    body = following[0] if following else None
    if not body:
        return {
            "heading_evidence_id": heading["fused_block_id"],
            "body_evidence_id": None,
        }
    later = sorted(
        (
            item
            for item in fused
            if int(item["page_number"]) == page
            and item.get("bbox")
            and item["bbox"][1] >= body["bbox"][3]
            and item["fused_block_id"] != body["fused_block_id"]
            and same_column(item)
        ),
        key=lambda item: item["bbox"][1],
    )
    next_block = later[0] if later else None
    sentences = re.findall(r"(?<!\b[A-Z])(?:[.!?](?=\s|$)|(?<=\w)\.(?=[A-Z]))", str(body["text"]))
    left_indent = None
    right_indent = None
    column_center = None
    if columns and columns.get("left_column_start_pt") is not None:
        column_width = float(columns.get("column_width_pt") or 0)
        column_left = float(columns["left_column_start_pt"])
        if not heading_is_left and columns.get("right_column_start_pt") is not None:
            column_left = float(columns["right_column_start_pt"])
        column_right = column_left + column_width
        if column_width > 0:
            column_center = (column_left + column_right) / 2
        left_indent = float(body["bbox"][0]) - column_left
        right_indent = column_right - float(body["bbox"][2])
    heading_center = (float(heading["bbox"][0]) + float(heading["bbox"][2])) / 2
    return {
        "heading": {
            "bbox": heading["bbox"],
            "style": heading.get("style"),
            "column_center_x_pt": column_center,
            "center_offset_from_column_pt": (
                abs(heading_center - column_center) if column_center is not None else None
            ),
            "evidence_id": heading["fused_block_id"],
        },
        "body": {
            "bbox": body["bbox"],
            "style": body.get("style"),
            "paragraph_count": 1,
            "sentence_count": len(sentences),
            "left_extra_indent_pt": left_indent,
            "right_extra_indent_pt": right_indent,
            "gap_after_pt": (
                float(next_block["bbox"][1]) - float(body["bbox"][3]) if next_block else None
            ),
            "evidence_id": body["fused_block_id"],
        },
    }


def derive_captions(fused: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind, role, pattern, object_role in (
        ("figures", "figure_caption", FIGURE_CAPTION, "figure_object"),
        ("tables", "table_caption", TABLE_CAPTION, "table_object"),
    ):
        captions = [
            item
            for item in fused
            if item.get("semantic_role") == role
            and item.get("bbox")
            and pattern.match(str(item.get("text") or ""))
        ]
        objects = [item for item in fused if item.get("semantic_role") == object_role and item.get("bbox")]
        records = []
        for caption in captions:
            match = pattern.match(str(caption.get("text") or ""))
            number = int(match.group(1))
            candidates: list[tuple[float, float, dict[str, Any]]] = []
            for obj in objects:
                if int(obj["page_number"]) != int(caption["page_number"]):
                    continue
                overlap = _horizontal_overlap(caption["bbox"], obj["bbox"])
                if overlap < 0.25:
                    continue
                if obj["bbox"][3] <= caption["bbox"][1] + 3:
                    gap = caption["bbox"][1] - obj["bbox"][3]
                elif caption["bbox"][3] <= obj["bbox"][1] + 3:
                    gap = obj["bbox"][1] - caption["bbox"][3]
                elif obj["bbox"][1] < caption["bbox"][3] and caption["bbox"][1] < obj["bbox"][3]:
                    gap = -min(
                        caption["bbox"][3] - obj["bbox"][1],
                        obj["bbox"][3] - caption["bbox"][1],
                    )
                else:
                    continue
                candidates.append((abs(gap) - overlap, gap, obj))
            paired = min(candidates, key=lambda item: item[0]) if candidates else None
            gap = None
            if paired:
                gap = paired[1]
            records.append(
                {
                    "number": number,
                    "page_number": caption["page_number"],
                    "caption_text": caption["text"],
                    "caption_bbox": caption["bbox"],
                    "caption_style": caption.get("style"),
                    "paired_object_bbox": paired[2].get("bbox") if paired else None,
                    "gap_pt": gap,
                    "caption_evidence_id": caption["fused_block_id"],
                    "object_evidence_id": paired[2].get("fused_block_id") if paired else None,
                }
            )
        numbers = sorted({item["number"] for item in records if item["number"] is not None})
        result[kind] = {
            "count": len(records),
            "numbers": numbers,
            "numbering_continuous": numbers == list(range(1, max(numbers) + 1)) if numbers else None,
            "items": records,
        }
    return result


def derive_references(
    fused: list[dict[str, Any]], native: dict[str, Any], structure: dict[str, Any]
) -> dict[str, Any] | None:
    start = structure.get("references_start")
    if not start:
        return None
    appendix = structure.get("appendix_start")
    start_page = int(start["page_number"])
    end_page = int(appendix["page_number"]) if appendix else int(structure["page_count"]) + 1
    entries = [
        item
        for item in fused
        if item.get("semantic_role") == "reference_entry"
        and start_page <= int(item["page_number"]) < end_page
    ]
    x_values_by_side: dict[str, list[float]] = defaultdict(list)
    page_widths = {int(page["page_number"]): float(page["width_pt"]) for page in native["pages"]}
    for line in native["lines"]:
        page = int(line["page_number"])
        if not start_page <= page < end_page or not line.get("bbox"):
            continue
        width = page_widths.get(page, 612.0)
        side = "left" if line["bbox"][0] < width / 2 else "right"
        if len(_text_key(line.get("text"))) >= 8:
            x_values_by_side[side].append(float(line["bbox"][0]))
    indent_samples: list[float] = []
    modes: dict[str, list[float]] = {}
    for side, values in x_values_by_side.items():
        buckets: dict[int, list[float]] = defaultdict(list)
        for value in values:
            buckets[round(value / 2)].append(value)
        ranked = sorted(buckets.values(), key=len, reverse=True)
        centers = sorted(statistics.median(group) for group in ranked[:4])
        close_pairs = [b - a for a, b in zip(centers, centers[1:]) if 6 <= b - a <= 14]
        if close_pairs:
            indent_samples.append(min(close_pairs, key=lambda value: abs(value - 10)))
        modes[side] = centers
    return {
        "heading": start,
        "entry_count": len(entries),
        "entry_evidence_ids": [item["fused_block_id"] for item in entries],
        "hanging_indent_pt": statistics.median(indent_samples) if indent_samples else None,
        "x_start_modes": modes,
    }


def build_rule_evidence_index(derived: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = [
        ("page_and_body", ["page_geometry", "column_geometry"]),
        ("title", ["front_matter.title"]),
        ("headings", ["fused_blocks:section_heading"]),
        ("figures", ["captions.figures"]),
        ("tables", ["captions.tables"]),
        ("references", ["references"]),
        ("appendix", ["document_structure.appendix_start", "column_geometry"]),
        ("initial_anonymity", ["front_matter.author_blocks"]),
        ("initial_page_limit", ["document_structure.main_body_page_count"]),
        ("camera_ready_letter", ["page_geometry"]),
        ("camera_ready_authors", ["front_matter.author_blocks", "front_matter.title.horizontal_rules"]),
        ("camera_ready_abstract", ["abstract"]),
        ("camera_ready_page_limit", ["document_structure.main_body_page_count"]),
    ]
    return [
        {"rule_key": rule_key, "evidence_paths": paths, "assessment": "evidence_available"}
        for rule_key, paths in mapping
    ]


def _acquire_mineru_json(pdf_path: Path, args: argparse.Namespace) -> tuple[Any, str]:
    if args.mineru_json:
        return _read_json(args.mineru_json.resolve()), str(args.mineru_json.resolve())

    token = os.environ.get("MINERU_API_KEY") or os.environ.get("MINERU_TOKEN")
    if not token:
        raise RuntimeError(
            "Set MINERU_API_KEY/MINERU_TOKEN or pass --mineru-json with an existing result"
        )
    script_dir = Path(__file__).resolve().parent
    user_paper_dir = script_dir / "format_test" / "user_paper"
    if not user_paper_dir.is_dir():
        user_paper_dir = script_dir / "user_paper"
    if not user_paper_dir.is_dir():
        raise RuntimeError("Cannot locate user_paper/mineru_clean.py beside the script")
    sys.path.insert(0, str(user_paper_dir))
    from mineru_clean import MineruClient  # type: ignore  # noqa: PLC0415

    cache_dir = args.mineru_cache_dir.resolve() if args.mineru_cache_dir else None
    temporary = tempfile.TemporaryDirectory(prefix="mineru-fusion-") if cache_dir is None else None
    work_dir = cache_dir or Path(temporary.name)
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        client = MineruClient(token, args.mineru_base_url, timeout=args.timeout)
        data_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:32]
        batch_id, upload_url = client.create_upload(pdf_path, data_id)
        client.upload_file(upload_url, pdf_path)
        result_url = client.wait_result(
            batch_id,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
        )
        export_dir = client.download_and_extract(result_url, work_dir)
        layout_path = export_dir / "layout.json"
        if layout_path.exists():
            return _read_json(layout_path), "mineru_api:layout.json"
        content_path = next(iter(sorted(export_dir.glob("*_content_list.json"))), None)
        if content_path is None:
            raise RuntimeError("MinerU completed but returned no layout/content-list JSON")
        return _read_json(content_path), "mineru_api:content_list.json"
    finally:
        if temporary is not None:
            temporary.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--mineru-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mineru-cache-dir", type=Path)
    parser.add_argument("--mineru-base-url", default=DEFAULT_MINERU_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-wait-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf.resolve()
    output_path = args.output.resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")
    if args.mineru_json and not args.mineru_json.is_file():
        raise SystemExit(f"MinerU JSON not found: {args.mineru_json}")

    started = time.perf_counter()
    native = extract_pymupdf(pdf_path)
    mineru_payload, mineru_source = _acquire_mineru_json(pdf_path, args)
    mineru = parse_mineru(mineru_payload, native["pages"])
    fused_blocks, quality = fuse_blocks(native, mineru)
    structure = derive_structure(fused_blocks, native["pages"])
    columns = derive_column_geometry(native, int(structure["main_body_page_count"]))
    page_geometry = {
        "page_count": len(native["pages"]),
        "pages": native["pages"],
        "all_pages_same_size": len(
            {(round(page["width_pt"], 2), round(page["height_pt"], 2)) for page in native["pages"]}
        )
        == 1,
    }
    derived = {
        "page_geometry": page_geometry,
        "document_structure": structure,
        "column_geometry": columns,
        "front_matter": derive_front_matter(fused_blocks, native, columns),
        "abstract": derive_abstract(fused_blocks, columns),
        "captions": derive_captions(fused_blocks),
        "references": derive_references(fused_blocks, native, structure),
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "pdf": str(pdf_path),
            "pdf_sha256": _sha256(pdf_path),
            "mineru": mineru_source,
            "precedence": {
                "geometry_and_style": "pymupdf",
                "semantic_roles": "mineru",
                "native_text": "pymupdf_when_available",
            },
        },
        "quality": {
            **quality,
            "pymupdf_span_count": len(native["spans"]),
            "pymupdf_line_count": len(native["lines"]),
            "pymupdf_object_count": len(native["objects"]),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "derived_facts": derived,
        "rule_evidence_index": build_rule_evidence_index(derived),
        "fused_blocks": fused_blocks,
        "native_facts": {
            "pages": native["pages"],
            "spans": native["spans"],
            "lines": native["lines"],
            "objects": native["objects"],
            "drawings": native["drawings"],
        },
    }
    _write_json(output_path, _round_floats(output))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "fused_blocks": len(fused_blocks),
                "native_spans": len(native["spans"]),
                "matched_blocks": quality["matched_block_count"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
