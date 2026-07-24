"""Targeted native-PDF facts for format-review evidence recovery.

The normal venue extractor remains the primary source of review facts.  This
module is deliberately narrow: it is invoked only after reflection identifies
missing PDF-local evidence, then returns compact, rule-scoped measurements
that are absent from the normal review payload.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .pdf_layout import NativePdfLayout, extract_native_pdf_layout

_SUPPORTED_CATEGORIES = {
    "author_identity",
    "heading",
    "page_layout",
    "reference",
}


def supplement_unit_evidence(
    *,
    pdf_path: str | Path,
    unit: dict[str, Any],
    rules_by_id: dict[str, dict[str, Any]],
    unresolved_rule_ids: list[str],
    existing_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only native facts relevant to unresolved rules in one unit.

    A supplement never guesses semantic content.  When the required raw
    observable cannot be identified, it returns no fact and lets the review
    remain unverifiable instead of manufacturing a recovery.
    """

    target_rules = [
        rules_by_id[rule_id]
        for rule_id in unresolved_rule_ids
        if rule_id in rules_by_id
        and str(rules_by_id[rule_id].get("rule_category") or "") in _SUPPORTED_CATEGORIES
    ]
    if not target_rules:
        return []
    layout = extract_native_pdf_layout(str(pdf_path))
    if not layout.available:
        return []

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in target_rules:
        by_category[str(rule.get("rule_category") or "")].append(rule)

    supplements: list[dict[str, Any]] = []
    if "page_layout" in by_category:
        supplements.append(_page_boundary_fact(layout, by_category["page_layout"]))
    if "heading" in by_category:
        fact = _heading_case_fact(unit, existing_facts, by_category["heading"])
        if fact is not None:
            supplements.append(fact)
    if "reference" in by_category:
        fact = _reference_indent_fact(layout, existing_facts, by_category["reference"])
        if fact is not None:
            supplements.append(fact)
    if "author_identity" in by_category:
        fact = _front_matter_fact(layout, by_category["author_identity"])
        if fact is not None:
            supplements.append(fact)
    return supplements


def _page_boundary_fact(
    layout: NativePdfLayout, rules: list[dict[str, Any]]
) -> dict[str, Any]:
    overflow: list[dict[str, Any]] = []
    for item in [*layout.spans, *layout.objects]:
        bbox = item.get("bbox")
        width = item.get("page_width_pt")
        height = item.get("page_height_pt")
        if (
            not _bbox(bbox)
            or not isinstance(width, (int, float))
            or not isinstance(height, (int, float))
        ):
            continue
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
            overflow.append(
                {
                    "page_number": item.get("page_number"),
                    "bbox": bbox,
                    "object_type": item.get("object_type") or "text_span",
                }
            )
    return _fact(
        name="page_boundary",
        role="derived_supplement_page_boundary",
        page_number=1,
        bbox=_page_bbox(layout, 1),
        rules=rules,
        measurements={
            "coverage": {
                "page_count": layout.page_count,
                "span_count": len(layout.spans),
                "object_count": len(layout.objects),
            },
            "overflow_count": len(overflow),
            "overflow_items": overflow[:16],
            "derivation": "native_pdf_bbox_vs_page_bounds",
            "confidence": 1.0,
        },
    )


def _heading_case_fact(
    unit: dict[str, Any], existing_facts: list[dict[str, Any]], rules: list[dict[str, Any]]
) -> dict[str, Any] | None:
    start, end = _page_range(unit)
    headings = [
        item
        for item in existing_facts
        if start <= int(item.get("page_number") or 0) <= end
        and (
            "heading" in str(item.get("role") or "").lower()
            or str(item.get("role") or "").lower() in {"title", "document_title"}
        )
        and str(item.get("quote") or "").strip()
        and _bbox(item.get("bbox"))
    ]
    if not headings:
        return None
    rows = [
        {
            "page_number": item.get("page_number"),
            "bbox": item.get("bbox"),
            "text": str(item.get("quote") or "")[:240],
            "title_case_minor_words": _minor_words(str(item.get("quote") or "")),
        }
        for item in headings[:32]
    ]
    return _fact(
        name="heading_case",
        role="derived_supplement_heading_case",
        page_number=int(rows[0]["page_number"]),
        bbox=rows[0]["bbox"],
        rules=rules,
        measurements={
            "coverage": {"heading_count": len(headings), "page_range": [start, end]},
            "headings": rows,
            "minor_word_policy": (
                "articles, conjunctions, short prepositions, and copular or auxiliary "
                "forms may be lowercase internally"
            ),
            "derivation": "semantic_heading_inventory",
            "confidence": 1.0,
        },
    )


def _reference_indent_fact(
    layout: NativePdfLayout,
    existing_facts: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> dict[str, Any] | None:
    reference_pages = {
        int(item.get("page_number") or 0)
        for item in existing_facts
        if "reference" in " ".join(
            str(item.get(key) or "").lower() for key in ("role", "section_title", "quote")
        )
    }
    reference_pages.discard(0)
    if not reference_pages:
        return None
    lines_by_page = _native_lines(layout.spans, reference_pages)
    samples: list[dict[str, Any]] = []
    for page_number, lines in sorted(lines_by_page.items()):
        for previous, current in zip(lines, lines[1:], strict=False):
            vertical_gap = current["bbox"][1] - previous["bbox"][3]
            if vertical_gap <= 0 or vertical_gap > 18:
                continue
            samples.append(
                {
                    "page_number": page_number,
                    "first_line_bbox": previous["bbox"],
                    "continuation_line_bbox": current["bbox"],
                    "continuation_indent_pt": round(current["bbox"][0] - previous["bbox"][0], 4),
                }
            )
    if not samples:
        return None
    return _fact(
        name="reference_indentation",
        role="derived_supplement_reference_indentation",
        page_number=samples[0]["page_number"],
        bbox=samples[0]["first_line_bbox"],
        rules=rules,
        measurements={
            "coverage": {
                "reference_pages": sorted(reference_pages),
                "line_pair_count": len(samples),
            },
            "line_indent_samples": samples[:32],
            "derivation": "native_pdf_adjacent_reference_lines",
            "confidence": 0.7,
        },
    )


def _front_matter_fact(
    layout: NativePdfLayout, rules: list[dict[str, Any]]
) -> dict[str, Any] | None:
    page_one = [
        item
        for item in layout.spans
        if item.get("page_number") == 1 and _bbox(item.get("bbox"))
    ]
    if not page_one:
        return None
    candidates = [
        {
            "bbox": item["bbox"],
            "text": str(item.get("text") or "")[:180],
            "font_name": item.get("font_name"),
            "font_size_pt": item.get("font_size_pt"),
            "font_flags": item.get("font_flags"),
        }
        for item in sorted(page_one, key=lambda item: (item["bbox"][1], item["bbox"][0]))[:48]
    ]
    return _fact(
        name="front_matter_spans",
        role="derived_supplement_front_matter",
        page_number=1,
        bbox=_page_bbox(layout, 1),
        rules=rules,
        measurements={
            "coverage": {"page_number": 1, "span_count": len(page_one)},
            "spans": candidates,
            "derivation": "native_pdf_page_one_spans",
            "confidence": 1.0,
        },
    )


def _native_lines(
    spans: list[dict[str, Any]], pages: set[int]
) -> dict[int, list[dict[str, Any]]]:
    lines: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        page_number = int(span.get("page_number") or 0)
        bbox = span.get("bbox")
        if page_number not in pages or not _bbox(bbox):
            continue
        lines[page_number].append({"bbox": bbox, "text": span.get("text")})
    for page_lines in lines.values():
        page_lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return lines


def _fact(
    *,
    name: str,
    role: str,
    page_number: int,
    bbox: list[float] | None,
    rules: list[dict[str, Any]],
    measurements: dict[str, Any],
) -> dict[str, Any]:
    rule_ids = [str(rule.get("rule_id")) for rule in rules if rule.get("rule_id")]
    return {
        "evidence_id": f"P-SUP-{name}",
        "block_id": f"derived-supplement-{name}",
        "page_number": page_number,
        "bbox": bbox,
        "quote": name,
        "role": role,
        "section_title": name,
        "measurements": {**measurements, "supplemented_rule_ids": rule_ids},
        "source": "format_evidence_supplementer",
        "confidence": float(measurements.get("confidence") or 0.0),
        "source_uri": f"supplement://{name}",
    }


def _page_range(unit: dict[str, Any]) -> tuple[int, int]:
    value = unit.get("page_range")
    if isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) for item in value):
        return value[0], value[1]
    return 1, 1


def _page_bbox(layout: NativePdfLayout, page_number: int) -> list[float] | None:
    for item in [*layout.spans, *layout.objects]:
        if item.get("page_number") != page_number:
            continue
        width, height = item.get("page_width_pt"), item.get("page_height_pt")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            return [0.0, 0.0, float(width), float(height)]
    return None


def _minor_words(text: str) -> list[str]:
    minor_words = {
        "a", "an", "and", "am", "are", "as", "at", "be", "been", "being", "but", "by",
        "for", "from", "in", "is", "nor", "of", "on", "or", "so", "the", "to", "was", "were",
        "with", "yet",
    }
    return [
        word
        for word in text.replace("-", " ").split()
        if word.lower().strip(".,:;()") in minor_words
    ]


def _bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and value[2] >= value[0]
        and value[3] >= value[1]
    )
