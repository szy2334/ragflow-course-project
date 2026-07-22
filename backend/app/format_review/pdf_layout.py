"""Native PDF layout facts for the format-review workflow.

This adapter is intentionally independent of paper-reading ingestion. It can
be run lazily for a format-review task and keeps raw PDF style facts separate
from semantic chunks used by reading and retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")


@dataclass(frozen=True, slots=True)
class NativePdfLayout:
    spans: list[dict[str, Any]]
    objects: list[dict[str, Any]]
    page_count: int
    available: bool
    reason: str | None = None


def extract_native_pdf_layout(file_path: str) -> NativePdfLayout:
    """Read span-level geometry and style without guessing unavailable facts."""

    try:
        import fitz
    except ImportError:
        return NativePdfLayout([], [], 0, False, "PyMuPDF is not installed.")
    path = Path(file_path)
    if not path.is_file():
        return NativePdfLayout([], [], 0, False, "论文原始 PDF 文件不存在。")
    try:
        document = fitz.open(path)
    except (fitz.FileDataError, RuntimeError, OSError) as exc:
        return NativePdfLayout([], [], 0, False, f"PDF 无法解析：{type(exc).__name__}")
    try:
        if document.needs_pass:
            return NativePdfLayout(
                [], [], document.page_count, False, "PDF 已加密，无法提取原生样式。"
            )
        spans: list[dict[str, Any]] = []
        objects: list[dict[str, Any]] = []
        for page_index, page in enumerate(document, start=1):
            page_rect = page.rect
            rotation = int(page.rotation or 0)
            raw = page.get_text("rawdict")
            for block in raw.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = "".join(
                            str(character.get("c") or "")
                            for character in span.get("chars", [])
                            if isinstance(character, dict)
                        ).strip()
                        bbox = _bbox(span.get("bbox"))
                        if not text or bbox is None:
                            continue
                        raw_font = str(span.get("font") or "") or None
                        spans.append(
                            {
                                "page_number": page_index,
                                "bbox": bbox,
                                "page_width_pt": float(page_rect.width),
                                "page_height_pt": float(page_rect.height),
                                "page_rotation": rotation,
                                "text": text,
                                "raw_font_name": raw_font,
                                "font_name": normalize_font_name(raw_font),
                                "font_size_pt": _number(span.get("size")),
                                "font_flags": _integer(span.get("flags")),
                                "color": _integer(span.get("color")),
                                "extraction_source": "native_pdf",
                            }
                        )
            objects.extend(_page_objects(page, page_index, page_rect))
        return NativePdfLayout(spans, objects, document.page_count, True)
    finally:
        document.close()


def normalize_font_name(value: str | None) -> str | None:
    if not value:
        return None
    return _SUBSET_PREFIX.sub("", value)


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    width, height = max(0.0, x1 - x0), max(0.0, y1 - y0)
    intersection = width * height
    if not intersection:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    x0, y0, x1, y1 = (float(item) for item in value)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _page_objects(page: Any, page_number: int, page_rect: Any) -> list[dict[str, Any]]:
    """Expose native image/vector geometry without guessing visual content."""

    objects: list[dict[str, Any]] = []
    for image_index, image in enumerate(page.get_images(full=True), start=1):
        xref = int(image[0])
        for rect_index, rect in enumerate(page.get_image_rects(xref), start=1):
            bbox = _bbox((rect.x0, rect.y0, rect.x1, rect.y1))
            if bbox is None:
                continue
            objects.append(
                {
                    "object_id": f"image-{xref}-{image_index}-{rect_index}",
                    "object_type": "image",
                    "page_number": page_number,
                    "bbox": bbox,
                    "page_width_pt": float(page_rect.width),
                    "page_height_pt": float(page_rect.height),
                    "vertical_rule_count": 0,
                    "horizontal_rule_count": 0,
                    "extraction_source": "native_pdf_image_object",
                }
            )

    drawings = page.get_drawings()
    drawing_bboxes = [
        _bbox((item["rect"].x0, item["rect"].y0, item["rect"].x1, item["rect"].y1))
        for item in drawings
        if item.get("rect")
    ]
    valid_bboxes = [bbox for bbox in drawing_bboxes if bbox is not None]
    vertical_rules = 0
    horizontal_rules = 0
    for drawing in drawings:
        for item in drawing.get("items", []):
            if not item or item[0] != "l" or len(item) < 3:
                continue
            start, end = item[1], item[2]
            if abs(float(start.x) - float(end.x)) < 0.5 and abs(float(start.y) - float(end.y)) > 3:
                vertical_rules += 1
            if abs(float(start.y) - float(end.y)) < 0.5 and abs(float(start.x) - float(end.x)) > 3:
                horizontal_rules += 1
    if valid_bboxes:
        objects.append(
            {
                "object_id": f"vector-layout-{page_number}",
                "object_type": "vector_graphic",
                "page_number": page_number,
                "bbox": [
                    min(item[0] for item in valid_bboxes),
                    min(item[1] for item in valid_bboxes),
                    max(item[2] for item in valid_bboxes),
                    max(item[3] for item in valid_bboxes),
                ],
                "page_width_pt": float(page_rect.width),
                "page_height_pt": float(page_rect.height),
                "vertical_rule_count": vertical_rules,
                "horizontal_rule_count": horizontal_rules,
                "extraction_source": "native_pdf_vector_geometry",
            }
        )
    return objects
