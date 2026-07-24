"""Offline MinerU + PyMuPDF geometry validation pipeline.

This tool deliberately does not import or change the format-review workflow.
It creates three auditable artifacts for a supplied PDF and MinerU layout:

1. ``native_pdf.json``: native PyMuPDF text, images, and drawing geometry.
2. ``enriched_layout.json``: a copy of the MinerU layout where successfully
   matched text-line/block ``bbox`` values are replaced by native PDF geometry.
3. ``derived_geometry.json``: column and alignment facts derived only from
   native-backed ``bbox`` values in the enriched layout.

The original MinerU layout is never written to or changed.  MinerU remains the
semantic authority; PyMuPDF is the geometry authority whenever a match exists.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import fitz


SCHEMA_VERSION = "geometry-validation-v1"
TEXT_BLOCK_TYPES = {"text", "title", "ref_text"}
SPACE = re.compile(r"\s+")
MINERU_EQUATION_MARKUP = re.compile(r"\^\{([^}]*)\}")
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,2})\.?\s+")
CAPTION = re.compile(r"^(figure|fig\.?|table)\s*(\d+)\b", re.IGNORECASE)
MIN_TEXT_MATCH_SCORE = 0.78


def rounded_bbox(values: Iterable[float]) -> list[float]:
    return [round(float(value), 3) for value in values]


def union_bbox(bboxes: Iterable[list[float]]) -> list[float] | None:
    values = [bbox for bbox in bboxes if is_bbox(bbox)]
    if not values:
        return None
    return rounded_bbox((
        min(item[0] for item in values),
        min(item[1] for item in values),
        max(item[2] for item in values),
        max(item[3] for item in values),
    ))


def bbox_intersection_area(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def bbox_iou(left: list[float], right: list[float]) -> float:
    intersection = bbox_intersection_area(left, right)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def bbox_center_inside(inner: list[float], outer: list[float], padding: float = 12.0) -> bool:
    center_x = (inner[0] + inner[2]) / 2
    center_y = (inner[1] + inner[3]) / 2
    return (
        outer[0] - padding <= center_x <= outer[2] + padding
        and outer[1] - padding <= center_y <= outer[3] + padding
    )


def is_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and value[2] > value[0]
        and value[3] > value[1]
    )


def normalize_text(value: str) -> str:
    """Normalise textual noise without using geometry as a match condition."""

    value = MINERU_EQUATION_MARKUP.sub(r"\1", value or "")
    return SPACE.sub("", value).replace("‐", "-").replace("–", "-").lower()


def line_text(line: dict[str, Any]) -> str:
    return "".join(
        str(span.get("content") or "")
        for span in line.get("spans") or []
        if isinstance(span, dict)
    )


def native_pdf(pdf_path: Path) -> dict[str, Any]:
    """Export raw PDF facts with individual lines, characters, images, and paths."""

    pages: list[dict[str, Any]] = []
    document = fitz.open(pdf_path)
    try:
        for page_number, page in enumerate(document, start=1):
            page_lines: list[dict[str, Any]] = []
            for block_index, raw_block in enumerate(page.get_text("rawdict").get("blocks", [])):
                for line_index, raw_line in enumerate(raw_block.get("lines", [])):
                    spans: list[dict[str, Any]] = []
                    for span_index, raw_span in enumerate(raw_line.get("spans", [])):
                        chars = [
                            {
                                "text": str(char.get("c") or ""),
                                "bbox": rounded_bbox(char["bbox"]),
                                "origin": rounded_bbox([*char.get("origin", (0, 0)), 0, 0])[:2],
                            }
                            for char in raw_span.get("chars", [])
                            if isinstance(char, dict) and isinstance(char.get("bbox"), (list, tuple))
                        ]
                        text = "".join(item["text"] for item in chars)
                        if not text.strip() or not isinstance(raw_span.get("bbox"), (list, tuple)):
                            continue
                        spans.append({
                            "span_index": span_index,
                            "text": text,
                            "normalized_text": normalize_text(text),
                            "bbox": rounded_bbox(raw_span["bbox"]),
                            "origin": rounded_bbox([*raw_span.get("origin", (0, 0)), 0, 0])[:2],
                            "font": str(raw_span.get("font") or ""),
                            "font_size_pt": round(float(raw_span.get("size") or 0), 3) or None,
                            "font_flags": int(raw_span.get("flags") or 0),
                            "color": int(raw_span.get("color") or 0),
                            "ascender": round(float(raw_span.get("ascender") or 0), 5),
                            "descender": round(float(raw_span.get("descender") or 0), 5),
                            "chars": chars,
                        })
                    bbox = union_bbox([span["bbox"] for span in spans])
                    if not spans or bbox is None:
                        continue
                    text = "".join(span["text"] for span in spans)
                    page_lines.append({
                        "line_id": f"p{page_number}-b{block_index}-l{line_index}",
                        "text": text,
                        "normalized_text": normalize_text(text),
                        "bbox": bbox,
                        "origin": rounded_bbox([min(span["origin"][0] for span in spans), min(span["origin"][1] for span in spans), 0, 0])[:2],
                        "spans": spans,
                    })

            images: list[dict[str, Any]] = []
            for image_index, image in enumerate(page.get_images(full=True), start=1):
                xref = int(image[0])
                for rect_index, rect in enumerate(page.get_image_rects(xref), start=1):
                    images.append({
                        "object_id": f"image-{xref}-{image_index}-{rect_index}",
                        "xref": xref,
                        "bbox": rounded_bbox((rect.x0, rect.y0, rect.x1, rect.y1)),
                    })

            drawings: list[dict[str, Any]] = []
            for drawing_index, drawing in enumerate(page.get_drawings(), start=1):
                rect = drawing.get("rect")
                if rect is None:
                    continue
                segments: list[dict[str, Any]] = []
                for item in drawing.get("items", []):
                    if not item or item[0] != "l" or len(item) < 3:
                        continue
                    start, end = item[1], item[2]
                    x0, y0, x1, y1 = float(start.x), float(start.y), float(end.x), float(end.y)
                    orientation = (
                        "horizontal" if abs(y0 - y1) <= 0.25 and abs(x1 - x0) > 1
                        else "vertical" if abs(x0 - x1) <= 0.25 and abs(y1 - y0) > 1
                        else "other"
                    )
                    segments.append({
                        "orientation": orientation,
                        "start": [round(x0, 3), round(y0, 3)],
                        "end": [round(x1, 3), round(y1, 3)],
                        "length_pt": round(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 3),
                    })
                drawings.append({
                    "drawing_id": f"drawing-{page_number}-{drawing_index}",
                    "bbox": rounded_bbox((rect.x0, rect.y0, rect.x1, rect.y1)),
                    "stroke_width_pt": round(float(drawing.get("width") or 0), 3),
                    "stroke_color": drawing.get("color"),
                    "fill_color": drawing.get("fill"),
                    "segments": segments,
                })

            pages.append({
                "page_number": page_number,
                "page_size_pt": [round(float(page.rect.width), 3), round(float(page.rect.height), 3)],
                "rotation": int(page.rotation or 0),
                "text_lines": page_lines,
                "images": images,
                "drawings": drawings,
            })
    finally:
        document.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "source_pdf": str(pdf_path.resolve()),
        "page_count": len(pages),
        "pages": pages,
    }


def best_native_line(
    mineru_line: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float]:
    """Match a MinerU line to one contiguous native-PDF line sequence.

    MinerU often joins PDF lines into one semantic line (notably paragraphs
    and multi-line titles).  The returned sequence is therefore the native
    coordinate source, never a guessed rectangle based on MinerU geometry.
    """

    target = normalize_text(line_text(mineru_line))
    if len(target) < 2:
        return [], 0.0
    best: tuple[float, list[dict[str, Any]]] | None = None
    prefix = target[: min(10, len(target))]
    start_positions = [
        index
        for index, candidate in enumerate(candidates)
        if (
            prefix in candidate["normalized_text"]
            or candidate["normalized_text"][: min(10, len(candidate["normalized_text"]))] in target
        )
    ]
    if not start_positions:
        return [], 0.0
    # A paragraph never needs the next page, but can span many native lines.
    # Limit the window to avoid accidentally collecting a following paragraph.
    for start in start_positions:
        joined = ""
        for end in range(start, min(start + 45, len(candidates))):
            joined += candidates[end]["normalized_text"]
            if len(joined) < max(2, len(target) // 6):
                continue
            if len(joined) > len(target) * 1.18 + 24:
                break
            score = SequenceMatcher(None, target, joined, autojunk=False).ratio()
            if target in joined or joined in target:
                score = max(score, min(len(target), len(joined)) / max(len(target), len(joined)))
            if best is None or score > best[0]:
                best = (score, candidates[start : end + 1])
    if best is None or best[0] < MIN_TEXT_MATCH_SCORE:
        return [], 0.0
    return best[1], round(best[0], 4)


def native_style(native_lines: list[dict[str, Any]], score: float) -> dict[str, Any]:
    spans = [span for line in native_lines for span in line["spans"]]
    span_count = Counter(span["font"] for span in spans if span["font"])
    size_count = Counter(span["font_size_pt"] for span in spans if span["font_size_pt"])
    flags_count = Counter(span["font_flags"] for span in spans)
    bbox = union_bbox([line["bbox"] for line in native_lines])
    return {
        "source": "pymupdf_text_line_sequence",
        "match_method": "same_page_normalized_text_contiguous_sequence",
        "match_score": score,
        "native_line_ids": [line["line_id"] for line in native_lines],
        "native_bbox": bbox,
        "origin": native_lines[0]["origin"],
        "font": span_count.most_common(1)[0][0] if span_count else None,
        "font_size_pt": size_count.most_common(1)[0][0] if size_count else None,
        "font_flags": flags_count.most_common(1)[0][0] if flags_count else None,
    }


def native_object_matches(
    mineru_bbox: list[float] | None, native_page: dict[str, Any]
) -> list[dict[str, Any]]:
    """Select native visual components using MinerU only as a semantic anchor.

    The output rectangle is always a union of native PDF objects.  MinerU's
    rectangle is used only to associate a semantic figure/table/chart with the
    native elements that make it up.
    """

    if not is_bbox(mineru_bbox):
        return []
    candidates = [
        {"kind": "image", "object_id": item["object_id"], "bbox": item["bbox"]}
        for item in native_page.get("images") or []
    ] + [
        {"kind": "drawing", "object_id": item["drawing_id"], "bbox": item["bbox"]}
        for item in native_page.get("drawings") or []
    ] + [
        # Vector figures and tables commonly contain native text (axis labels,
        # table cells, legends).  It is part of the object geometry even though
        # PyMuPDF exposes it as ordinary text rather than a figure/table node.
        {"kind": "text", "object_id": item["line_id"], "bbox": item["bbox"]}
        for item in native_page.get("text_lines") or []
    ]
    return [
        candidate
        for candidate in candidates
        if is_bbox(candidate["bbox"])
        and (
            bbox_iou(mineru_bbox, candidate["bbox"]) >= 0.03
            or bbox_center_inside(candidate["bbox"], mineru_bbox)
        )
    ]


def enrich_layout(layout: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    """Copy MinerU hierarchy, replacing only successful text geometry."""

    enriched = copy.deepcopy(layout)
    native_by_page = {page["page_number"]: page["text_lines"] for page in native["pages"]}
    native_pages_by_number = {page["page_number"]: page for page in native["pages"]}
    counts = Counter()
    for page in enriched.get("pdf_info") or []:
        if not isinstance(page, dict):
            continue
        page_number = int(page.get("page_idx") or 0) + 1
        native_page = next((item for item in native["pages"] if item["page_number"] == page_number), None)
        if native_page is None:
            continue
        page["mineru_page_size"] = copy.deepcopy(page.get("page_size"))
        page["page_size"] = native_page["page_size_pt"]
        page["geometry"] = {"source": "pymupdf_page_rect", "rotation": native_page["rotation"]}
        for collection in ("para_blocks", "discarded_blocks"):
            for block in page.get(collection) or []:
                if not isinstance(block, dict):
                    continue
                block_bbox_before = copy.deepcopy(block.get("bbox"))
                matched_line_bboxes: list[list[float]] = []
                for line in block.get("lines") or []:
                    if not isinstance(line, dict):
                        continue
                    matches, score = best_native_line(line, native_by_page[page_number])
                    if not matches:
                        counts["unmatched_lines"] += 1
                        line["geometry"] = {"source": "mineru_layout"}
                        continue
                    line["mineru_bbox"] = copy.deepcopy(line.get("bbox"))
                    line["bbox"] = union_bbox([match["bbox"] for match in matches])
                    line["geometry"] = native_style(matches, score)
                    matched_line_bboxes.append(line["bbox"])
                    counts["native_backed_lines"] += 1

                if matched_line_bboxes:
                    block["mineru_bbox"] = block_bbox_before
                    # Do not include unmatched MinerU lines: their bbox cannot be allowed
                    # to move a strict physical calculation.  Their original values remain
                    # preserved on the line itself.
                    block["bbox"] = union_bbox(matched_line_bboxes)
                    block["geometry"] = {
                        "source": "pymupdf_text_line_union",
                        "native_backed_line_count": len(matched_line_bboxes),
                        "total_line_count": len(block.get("lines") or []),
                    }
                    counts["native_backed_blocks"] += 1
                else:
                    block["geometry"] = {"source": "mineru_layout"}
                    counts["mineru_only_blocks"] += 1

                # Visual objects have no stable semantic identity in PyMuPDF.
                # For them MinerU keeps the object type while native PDF images
                # and drawings, once associated, supply the effective geometry.
                if str(block.get("type") or "") in {"image", "chart", "table"}:
                    original_bbox = block.get("mineru_bbox") or block.get("bbox")
                    visual_matches = native_object_matches(original_bbox, native_pages_by_number[page_number])
                    visual_bbox = union_bbox([item["bbox"] for item in visual_matches])
                    if visual_bbox is not None:
                        if "mineru_bbox" not in block:
                            block["mineru_bbox"] = copy.deepcopy(block.get("bbox"))
                        block["bbox"] = visual_bbox
                        block["geometry"] = {
                            "source": "pymupdf_visual_object_union",
                            "native_object_ids": [item["object_id"] for item in visual_matches],
                            "native_object_kinds": sorted({item["kind"] for item in visual_matches}),
                        }
                        counts["native_backed_visual_blocks"] += 1

    enriched["geometry_fusion"] = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_policy": "bbox is the effective geometry; mineru_bbox preserves original coordinates",
        "semantic_authority": "mineru",
        "geometry_authority_when_matched": "pymupdf",
        "statistics": dict(counts),
    }
    return enriched


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    position = (len(ordered) - 1) * fraction
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def native_backed_body_lines(page: dict[str, Any]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for block in page.get("para_blocks") or []:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        for line in block.get("lines") or []:
            geometry = line.get("geometry") if isinstance(line, dict) else None
            if (
                isinstance(geometry, dict)
                and geometry.get("source") == "pymupdf_text_line_sequence"
                and is_bbox(line.get("bbox"))
            ):
                lines.append(line)
    return lines


def detect_columns(page: dict[str, Any]) -> dict[str, Any]:
    lines = native_backed_body_lines(page)
    page_size = page.get("page_size") or [0, 0]
    if len(lines) < 8 or not isinstance(page_size, list) or not page_size[0]:
        return {"page_number": int(page.get("page_idx") or 0) + 1, "column_count": None}
    width = float(page_size[0])
    candidates = [line for line in lines if 45 <= line["bbox"][2] - line["bbox"][0] <= width * 0.49]
    starts = sorted(float(line["geometry"]["origin"][0]) for line in candidates)
    if len(starts) < 8:
        return {"page_number": int(page.get("page_idx") or 0) + 1, "column_count": 1, "reason": "too_few_column_candidates"}
    left = [line for line in candidates if line["geometry"]["origin"][0] < width / 2]
    right = [line for line in candidates if line["geometry"]["origin"][0] >= width / 2]
    if len(left) < 4 or len(right) < 4:
        return {"page_number": int(page.get("page_idx") or 0) + 1, "column_count": 1, "reason": "single_column_native_origins"}
    left_x0 = percentile([line["geometry"]["origin"][0] for line in left], 0.15)
    left_x1 = percentile([line["bbox"][2] for line in left], 0.85)
    right_x0 = percentile([line["geometry"]["origin"][0] for line in right], 0.15)
    right_x1 = percentile([line["bbox"][2] for line in right], 0.85)
    if right_x0 <= left_x1:
        return {"page_number": int(page.get("page_idx") or 0) + 1, "column_count": 1, "reason": "no_native_gutter"}
    return {
        "page_number": int(page.get("page_idx") or 0) + 1,
        "column_count": 2,
        "source": "enriched_layout.native_text_line_bbox_and_origin",
        "columns": [
            {"index": 1, "x0": round(left_x0, 3), "x1": round(left_x1, 3)},
            {"index": 2, "x0": round(right_x0, 3), "x1": round(right_x1, 3)},
        ],
        "gutter_width_pt": round(right_x0 - left_x1, 3),
        "text_extent_width_pt": round(right_x1 - left_x0, 3),
        "candidate_line_count": len(candidates),
    }


def template_rules(native_page: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for drawing in native_page.get("drawings") or []:
        for segment in drawing.get("segments") or []:
            if segment.get("orientation") != "horizontal" or segment.get("length_pt", 0) < 200:
                continue
            rules.append({
                "drawing_id": drawing["drawing_id"],
                "x0": min(segment["start"][0], segment["end"][0]),
                "x1": max(segment["start"][0], segment["end"][0]),
                "y": round((segment["start"][1] + segment["end"][1]) / 2, 3),
                "width_pt": segment["length_pt"],
                "stroke_width_pt": drawing["stroke_width_pt"],
            })
    return sorted(rules, key=lambda item: (-item["width_pt"], item["y"]))


def template_backed_column_model(
    enriched: dict[str, Any],
    double_column_pages: list[dict[str, Any]],
    template_rules_on_first_page: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Solve exact symmetric-column geometry from native PDF constraints.

    Visible glyph boxes are deliberately not used as a column right boundary.
    The horizontal template rule supplies total width ``T``; repeated native
    text origins supply left/right starts ``L`` and ``R``.  For a symmetric
    two-column layout, ``T = 2C + G`` and ``R - L = C + G``.
    """

    if len(double_column_pages) < 2 or not template_rules_on_first_page:
        return None
    template = template_rules_on_first_page[0]
    total_width = float(template["width_pt"])
    left_start = median(page["columns"][0]["x0"] for page in double_column_pages)
    right_start = median(page["columns"][1]["x0"] for page in double_column_pages)
    start_delta = right_start - left_start
    column_width = total_width - start_delta
    gutter_width = 2 * start_delta - total_width
    if column_width <= 0 or gutter_width < 0:
        return None
    return {
        "column_count": 2,
        "source": "native_horizontal_template_rule + repeated_native_text_origins",
        "template_rule": template,
        "left_column_x0_pt": round(left_start, 3),
        "left_column_x1_pt": round(left_start + column_width, 3),
        "right_column_x0_pt": round(right_start, 3),
        "right_column_x1_pt": round(right_start + column_width, 3),
        "column_width_pt": round(column_width, 3),
        "gutter_width_pt": round(gutter_width, 3),
        "body_total_width_pt": round(total_width, 3),
        "evidence_pages": [page["page_number"] for page in double_column_pages],
    }


def block_text(block: dict[str, Any]) -> str:
    return "".join(line_text(line) for line in block.get("lines") or [] if isinstance(line, dict)).strip()


def block_records(enriched: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten MinerU semantic blocks while retaining their effective geometry."""

    records: list[dict[str, Any]] = []
    for page in enriched.get("pdf_info") or []:
        if not isinstance(page, dict):
            continue
        page_number = int(page.get("page_idx") or 0) + 1
        for collection in ("para_blocks", "discarded_blocks"):
            for index, block in enumerate(page.get(collection) or []):
                if not isinstance(block, dict) or not is_bbox(block.get("bbox")):
                    continue
                records.append({
                    "record_id": f"p{page_number}-{collection}-{index}",
                    "page_number": page_number,
                    "collection": collection,
                    "type": str(block.get("type") or ""),
                    "text": block_text(block),
                    "bbox": block["bbox"],
                    "block": block,
                })
    return records


def native_line_map(native: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        line["line_id"]: line
        for page in native.get("pages") or []
        for line in page.get("text_lines") or []
    }


def record_style(record: dict[str, Any], lines_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    spans: list[dict[str, Any]] = []
    origins: list[list[float]] = []
    for line in record["block"].get("lines") or []:
        geometry = line.get("geometry") if isinstance(line, dict) else None
        if not isinstance(geometry, dict) or geometry.get("source") != "pymupdf_text_line_sequence":
            continue
        for line_id in geometry.get("native_line_ids") or []:
            native_line = lines_by_id.get(line_id)
            if native_line is None:
                continue
            origins.append(native_line["origin"])
            spans.extend(native_line.get("spans") or [])
    if not spans:
        return None
    fonts = Counter(span.get("font") for span in spans if span.get("font"))
    sizes = Counter(span.get("font_size_pt") for span in spans if span.get("font_size_pt"))
    flags = Counter(span.get("font_flags") for span in spans if span.get("font_flags") is not None)
    return {
        "font": fonts.most_common(1)[0][0] if fonts else None,
        "font_size_pt": sizes.most_common(1)[0][0] if sizes else None,
        "font_flags": flags.most_common(1)[0][0] if flags else None,
        "is_bold": bool((flags.most_common(1)[0][0] if flags else 0) & 16),
        "origins": sorted({(round(origin[0], 3), round(origin[1], 3)) for origin in origins}, key=lambda item: (item[1], item[0])),
    }


def container_for_bbox(bbox: list[float], page_width: float, column_model: dict[str, Any] | None) -> dict[str, Any]:
    center = (bbox[0] + bbox[2]) / 2
    if column_model and column_model.get("column_count") == 1:
        x0, x1 = float(column_model["body_x0_pt"]), float(column_model["body_x1_pt"])
        return {"kind": "single_column_body", "bbox": [x0, 0, x1, 0], "center_x_pt": round((x0 + x1) / 2, 3)}
    if column_model and column_model.get("column_count") == 2:
        left, right = column_model["left_column_x0_pt"], column_model["right_column_x1_pt"]
        left_end, right_start = column_model["left_column_x1_pt"], column_model["right_column_x0_pt"]
        if bbox[0] >= left - 8 and bbox[2] <= left_end + 8:
            return {"kind": "left_column", "bbox": [left, 0, left_end, 0], "center_x_pt": round((left + left_end) / 2, 3)}
        if bbox[0] >= right_start - 8 and bbox[2] <= right + 8:
            return {"kind": "right_column", "bbox": [right_start, 0, right, 0], "center_x_pt": round((right_start + right) / 2, 3)}
        return {"kind": "cross_column", "bbox": [left, 0, right, 0], "center_x_pt": round((left + right) / 2, 3)}
    return {"kind": "page_body", "bbox": [0, 0, page_width, 0], "center_x_pt": round(page_width / 2, 3)}


def alignment_fact(bbox: list[float], container: dict[str, Any], tolerance: float = 3.0) -> dict[str, Any]:
    center = (bbox[0] + bbox[2]) / 2
    target = float(container["center_x_pt"])
    return {
        "container_kind": container["kind"],
        "container_center_x_pt": target,
        "object_center_x_pt": round(center, 3),
        "delta_x_pt": round(center - target, 3),
        "is_centered": abs(center - target) <= tolerance,
        "left_x_pt": round(bbox[0], 3),
    }


def text_baseline_samples(native: dict[str, Any], size_range: tuple[float, float] = (8.0, 12.0)) -> list[float]:
    gaps: list[float] = []
    for page in native.get("pages") or []:
        groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for line in page.get("text_lines") or []:
            spans = line.get("spans") or []
            if not spans:
                continue
            size = Counter(span.get("font_size_pt") for span in spans if span.get("font_size_pt")).most_common(1)
            if not size or not (size_range[0] <= float(size[0][0]) <= size_range[1]):
                continue
            groups[round(line["origin"][0], 1)].append(line)
        for lines in groups.values():
            lines.sort(key=lambda item: item["origin"][1])
            for left, right in zip(lines, lines[1:]):
                gap = right["origin"][1] - left["origin"][1]
                if 7.0 <= gap <= 15.0:
                    gaps.append(round(gap, 3))
    return gaps


def derive_object_facts(
    enriched: dict[str, Any], native: dict[str, Any], column_model: dict[str, Any] | None, first_page_rules: list[dict[str, Any]]
) -> dict[str, Any]:
    records = block_records(enriched)
    styles = native_line_map(native)
    page_widths = {page["page_number"]: float(page["page_size_pt"][0]) for page in native.get("pages") or []}
    titled = [record for record in records if record["type"] == "title"]
    headings: list[dict[str, Any]] = []
    for record in titled:
        match = NUMBERED_HEADING.match(record["text"])
        if not match:
            continue
        level = match.group(1).count(".") + 1
        style = record_style(record, styles)
        container = container_for_bbox(record["bbox"], page_widths[record["page_number"]], column_model)
        headings.append({
            "record_id": record["record_id"], "page_number": record["page_number"], "text": record["text"],
            "number": match.group(1), "level": level, "bbox": record["bbox"], "style": style,
            "alignment": alignment_fact(record["bbox"], container),
        })

    body_records = [record for record in records if record["collection"] == "para_blocks" and record["type"] == "text" and record["page_number"] > 1]
    body_styles = [record_style(record, styles) for record in body_records]
    body_styles = [style for style in body_styles if style and style.get("font_size_pt")]
    fonts = Counter(style["font"] for style in body_styles if style.get("font"))
    sizes = Counter(style["font_size_pt"] for style in body_styles if style.get("font_size_pt"))
    leading = text_baseline_samples(native)

    title_facts: list[dict[str, Any]] = []
    for record in titled:
        if record["page_number"] != 1 or NUMBERED_HEADING.match(record["text"]):
            continue
        style = record_style(record, styles)
        container = container_for_bbox(record["bbox"], page_widths[1], column_model)
        surrounding_rules = [rule for rule in first_page_rules if rule["y"] < record["bbox"][1] or rule["y"] > record["bbox"][3]]
        title_facts.append({
            "record_id": record["record_id"], "text": record["text"], "bbox": record["bbox"],
            "style": style, "alignment": alignment_fact(record["bbox"], container),
            "horizontal_rules_outside_title": surrounding_rules,
        })

    abstract_title = next((record for record in titled if record["text"].strip().lower() == "abstract"), None)
    abstract_fact: dict[str, Any] | None = None
    if abstract_title:
        following = [record for record in records if record["page_number"] == abstract_title["page_number"] and record["collection"] == "para_blocks" and record["type"] == "text" and record["bbox"][1] >= abstract_title["bbox"][3]]
        following.sort(key=lambda item: item["bbox"][1])
        body = following[0] if following else None
        container = container_for_bbox(abstract_title["bbox"], page_widths[abstract_title["page_number"]], column_model)
        abstract_fact = {
            "title": {
                "record_id": abstract_title["record_id"], "bbox": abstract_title["bbox"],
                "style": record_style(abstract_title, styles),
                "alignment": alignment_fact(abstract_title["bbox"], container),
            },
            "body": ({
                "record_id": body["record_id"], "bbox": body["bbox"], "style": record_style(body, styles),
                "line_count": len(body["block"].get("lines") or []),
                "left_inset_from_container_pt": round(body["bbox"][0] - container["bbox"][0], 3),
                "right_inset_from_container_pt": round(container["bbox"][2] - body["bbox"][2], 3),
            } if body else None),
        }

    visual_records = [record for record in records if record["collection"] == "para_blocks" and record["type"] in {"image", "chart", "table"}]
    caption_candidates: list[dict[str, Any]] = []
    for record in records:
        match = CAPTION.match(record["text"])
        if match:
            caption_candidates.append({**record, "caption_kind": "table" if match.group(1).lower() == "table" else "figure", "number": int(match.group(2))})
    relations: list[dict[str, Any]] = []
    for caption in caption_candidates:
        expected = "table" if caption["caption_kind"] == "table" else None
        candidates = [record for record in visual_records if record["page_number"] == caption["page_number"] and (expected is None or record["type"] == expected)]
        if not candidates:
            continue
        visual = min(candidates, key=lambda item: abs((item["bbox"][1] + item["bbox"][3]) / 2 - (caption["bbox"][1] + caption["bbox"][3]) / 2))
        container = container_for_bbox(visual["bbox"], page_widths[visual["page_number"]], column_model)
        relations.append({
            "kind": caption["caption_kind"], "number": caption["number"], "page_number": caption["page_number"],
            "object_record_id": visual["record_id"], "object_bbox": visual["bbox"],
            "caption_record_id": caption["record_id"], "caption_bbox": caption["bbox"],
            "caption_position": "below" if caption["bbox"][1] >= visual["bbox"][3] else "above" if caption["bbox"][3] <= visual["bbox"][1] else "overlaps",
            "object_alignment": alignment_fact(visual["bbox"], container),
            "caption_style": record_style(caption, styles),
            "caption_line_count": len(caption["block"].get("lines") or []),
        })

    references = next((record for record in titled if record["text"].strip().lower() == "references"), None)
    reference_fact: dict[str, Any] | None = None
    if references:
        entries = [record for record in records if record["type"] == "ref_text" and (record["page_number"] > references["page_number"] or (record["page_number"] == references["page_number"] and record["bbox"][1] > references["bbox"][3]))]
        indent_samples: list[dict[str, Any]] = []
        for entry in entries:
            style = record_style(entry, styles)
            origins = style.get("origins") if style else []
            if len(origins) >= 2:
                indent_samples.append({
                    "record_id": entry["record_id"], "first_line_x_pt": origins[0][0],
                    "following_line_x_pt": min(origin[0] for origin in origins[1:]),
                    "hanging_indent_pt": round(min(origin[0] for origin in origins[1:]) - origins[0][0], 3),
                })
        reference_fact = {
            "heading": {"record_id": references["record_id"], "bbox": references["bbox"], "style": record_style(references, styles)},
            "entry_count": len(entries), "hanging_indent_samples": indent_samples[:25],
        }

    appendix = [record for record in titled if record["text"].strip().lower().startswith("appendix")]
    return {
        "page_geometry": {
            "page_sizes_pt": sorted({tuple(page.get("page_size") or []) for page in enriched.get("pdf_info") or []}),
            "all_pages_same_size": len({tuple(page.get("page_size") or []) for page in enriched.get("pdf_info") or []}) == 1,
        },
        "body_typography": {
            "sampled_block_count": len(body_styles),
            "dominant_font": fonts.most_common(1)[0][0] if fonts else None,
            "dominant_font_size_pt": sizes.most_common(1)[0][0] if sizes else None,
            "baseline_spacing_pt_samples": leading[:200],
            "median_baseline_spacing_pt": round(median(leading), 3) if leading else None,
        },
        "title_geometry": title_facts,
        "heading_geometry": headings,
        "abstract_geometry": abstract_fact,
        "figure_caption_relations": [item for item in relations if item["kind"] == "figure"],
        "table_caption_relations": [item for item in relations if item["kind"] == "table"],
        "references_section": reference_fact,
        "appendix_sections": [{"record_id": record["record_id"], "page_number": record["page_number"], "bbox": record["bbox"]} for record in appendix],
    }


def derive_geometry(enriched: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    pages = [detect_columns(page) for page in enriched.get("pdf_info") or [] if isinstance(page, dict)]
    double_column = [page for page in pages if page.get("column_count") == 2]
    native_pages = native.get("pages") or []
    first_page_rules = template_rules(native_pages[0]) if native_pages else []
    document_model: dict[str, Any] | None = template_backed_column_model(
        enriched, double_column, first_page_rules
    )
    if len(double_column) >= 2:
        text_estimate = {
            "column_count": 2,
            "left_column_x0_pt": round(median(page["columns"][0]["x0"] for page in double_column), 3),
            "right_column_x1_pt": round(median(page["columns"][1]["x1"] for page in double_column), 3),
            "native_text_extent_width_pt": round(median(page["text_extent_width_pt"] for page in double_column), 3),
            "native_text_gutter_estimate_pt": round(median(page["gutter_width_pt"] for page in double_column), 3),
            "evidence_pages": [page["page_number"] for page in double_column],
        }
        if document_model is None:
            document_model = {
                **text_estimate,
                "source": "native_text_origins_and_visible_text_boxes",
            }
        else:
            document_model["visible_text_estimate"] = text_estimate
    if document_model is None and first_page_rules:
        # A single full-width native rule can define an exact single-column
        # body container (as in the NeurIPS 2020 template sample).
        rule = first_page_rules[0]
        document_model = {
            "column_count": 1,
            "source": "native_horizontal_template_rule",
            "template_rule": rule,
            "body_x0_pt": rule["x0"],
            "body_x1_pt": rule["x1"],
            "body_total_width_pt": rule["width_pt"],
        }
    object_facts = derive_object_facts(enriched, native, document_model, first_page_rules)
    return {
        "schema_version": SCHEMA_VERSION,
        "input_coordinate_policy": "only enriched-layout bbox values whose line geometry.source is pymupdf_text_line_sequence",
        "page_count": len(pages),
        "page_models": pages,
        "document_column_model": document_model,
        "first_page_horizontal_template_rules": first_page_rules,
        **object_facts,
        "interpretation": {
            "native_text_extent": "visible text extent from native lines; not treated as a template width",
            "template_rule": "native horizontal drawing; may provide an exact template width when it is a title/body rule",
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("layout", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")
    if not args.layout.is_file():
        raise SystemExit(f"MinerU layout not found: {args.layout}")

    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    native = native_pdf(args.pdf.resolve())
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    enriched = enrich_layout(layout, native)
    derived = derive_geometry(enriched, native)
    write_json(destination / "native_pdf.json", native)
    write_json(destination / "enriched_layout.json", enriched)
    write_json(destination / "derived_geometry.json", derived)
    print(json.dumps({
        "native_pdf": str(destination / "native_pdf.json"),
        "enriched_layout": str(destination / "enriched_layout.json"),
        "derived_geometry": str(destination / "derived_geometry.json"),
        "fusion_statistics": enriched["geometry_fusion"]["statistics"],
        "double_column_pages": [item["page_number"] for item in derived["page_models"] if item.get("column_count") == 2],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
