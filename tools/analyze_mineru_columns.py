"""Infer page columns from MinerU layout.json text-line geometry.

This tool treats MinerU as the authority for page geometry.  It does not open
the source PDF or use PyMuPDF.  The output is intended as auditable input for
format-review alignment checks, not as a replacement for the raw layout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


TEXT_BLOCK_TYPES = {"text", "title", "ref_text"}


@dataclass(frozen=True)
class TextLine:
    bbox: tuple[float, float, float, float]
    block_type: str

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


def is_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and value[2] > value[0]
        and value[3] > value[1]
    )


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    remainder = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * remainder


def page_text_lines(page: dict[str, Any]) -> list[TextLine]:
    """Use paragraph-stage lines; preproc_blocks duplicate these blocks."""

    lines: list[TextLine] = []
    for block in page.get("para_blocks") or []:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type not in TEXT_BLOCK_TYPES:
            continue
        for line in block.get("lines") or []:
            if not isinstance(line, dict) or not is_bbox(line.get("bbox")):
                continue
            bbox = tuple(float(value) for value in line["bbox"])
            lines.append(TextLine(bbox=bbox, block_type=block_type))
    return lines


def column_bounds(lines: list[TextLine]) -> tuple[float, float]:
    """Estimate stable column edges while ignoring ragged last lines."""

    return (
        percentile((line.bbox[0] for line in lines), 0.10),
        percentile((line.bbox[2] for line in lines), 0.90),
    )


def analyze_page(page: dict[str, Any]) -> dict[str, Any]:
    page_idx = int(page.get("page_idx") or 0)
    page_size = page.get("page_size")
    if not isinstance(page_size, list) or len(page_size) != 2:
        return {"page_idx": page_idx, "column_count": None, "reason": "page_size_missing"}
    page_width, page_height = (float(value) for value in page_size)
    if page_width <= 0 or page_height <= 0:
        return {"page_idx": page_idx, "column_count": None, "reason": "page_size_invalid"}

    all_lines = page_text_lines(page)
    # Full-width titles, abstracts, and wide table text would blur the gutter.
    candidates = [
        line
        for line in all_lines
        if page_width * 0.08 <= line.width <= page_width * 0.48
    ]
    left_lines = [line for line in candidates if line.center_x < page_width / 2]
    right_lines = [line for line in candidates if line.center_x >= page_width / 2]
    result: dict[str, Any] = {
        "page_idx": page_idx,
        "page_size": [round(page_width, 3), round(page_height, 3)],
        "text_line_count": len(all_lines),
        "candidate_line_count": len(candidates),
        "left_candidate_count": len(left_lines),
        "right_candidate_count": len(right_lines),
    }
    if len(left_lines) < 4 or len(right_lines) < 4:
        result.update({"column_count": 1, "reason": "insufficient_two_column_evidence"})
        return result

    left_x0, left_x1 = column_bounds(left_lines)
    right_x0, right_x1 = column_bounds(right_lines)
    left_width = left_x1 - left_x0
    right_width = right_x1 - right_x0
    gutter = right_x0 - left_x1
    similar_widths = abs(left_width - right_width) <= max(12.0, max(left_width, right_width) * 0.18)
    valid_gutter = page_width * 0.02 <= gutter <= page_width * 0.20
    if left_width <= 0 or right_width <= 0 or not similar_widths or not valid_gutter:
        result.update(
            {
                "column_count": 1,
                "reason": "column_geometry_not_consistent",
                "candidate_columns": {
                    "left": [round(left_x0, 3), round(left_x1, 3)],
                    "right": [round(right_x0, 3), round(right_x1, 3)],
                    "gutter_width": round(gutter, 3),
                },
            }
        )
        return result

    total_width = right_x1 - left_x0
    result.update(
        {
            "column_count": 2,
            "reason": "two_column_text_geometry",
            "columns": [
                {"index": 1, "x0": round(left_x0, 3), "x1": round(left_x1, 3), "width": round(left_width, 3)},
                {"index": 2, "x0": round(right_x0, 3), "x1": round(right_x1, 3), "width": round(right_width, 3)},
            ],
            "gutter_width": round(gutter, 3),
            "body_total_width": round(total_width, 3),
        }
    )
    return result


def document_summary(pages: list[dict[str, Any]]) -> dict[str, Any]:
    two_column = [page for page in pages if page.get("column_count") == 2]
    # A document style is normally shared across pages.  Two independently
    # observed double-column pages are sufficient to establish that model;
    # title, figure-only, and sparse pages do not need to rediscover it.
    if len(two_column) < 2:
        return {
            "detected_two_column_pages": len(two_column),
            "minimum_evidence_pages": 2,
            "column_model": None,
        }
    return {
        "detected_two_column_pages": len(two_column),
        "minimum_evidence_pages": 2,
        "column_model": {
            "column_count": 2,
            "left_column_width": round(median(page["columns"][0]["width"] for page in two_column), 3),
            "right_column_width": round(median(page["columns"][1]["width"] for page in two_column), 3),
            "gutter_width": round(median(page["gutter_width"] for page in two_column), 3),
            "body_total_width": round(median(page["body_total_width"] for page in two_column), 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout_json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    layout = json.loads(args.layout_json.read_text(encoding="utf-8"))
    if not isinstance(layout, dict) or not isinstance(layout.get("pdf_info"), list):
        raise SystemExit("layout_json must contain a pdf_info array")
    pages = [analyze_page(page) for page in layout["pdf_info"] if isinstance(page, dict)]
    summary = document_summary(pages)
    document_model = summary.get("column_model")
    if isinstance(document_model, dict):
        for page in pages:
            page["document_column_model"] = document_model
            page["document_column_model_source"] = "two_or_more_observed_pages"
    report = {
        "source_layout": str(args.layout_json.resolve()),
        "coordinate_authority": "mineru_layout_json",
        "summary": summary,
        "pages": pages,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
