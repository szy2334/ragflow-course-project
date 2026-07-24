"""Venue-specific fused layout extraction for format review.

The ICML and NeurIPS extractors live at the repository root because they are
also useful as standalone audit tools.  This adapter keeps their output schema
stable at the workflow boundary and reuses the MinerU artifact produced during
paper ingestion instead of uploading the same PDF a second time.
"""

# Compact review-inventory dictionaries intentionally retain audit fields.
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

SUPPORTED_VENUE_EXTRACTORS = {
    "icml": "icml_build_fused_layout_facts.py",
    "neurips": "neurips_rule_evidence.py",
}

# PDF extraction occasionally emits author names as a full given-name/surname
# sequence (for example, ``Blaise Aguera y Arcas, M.``), and occasionally
# starts an institutional entry with a lowercase product name.  The pattern
# deliberately requires a multi-character first token so ``R., Ramapuram``
# remains a wrapped author-list continuation instead of a new entry.
_REFERENCE_AUTHOR_PATTERN = (
    r"^((?:[A-Z][A-Za-z'’¨\-]+(?:\s+(?:[A-Z][A-Za-z'’¨\-]*|[a-z]+)){0,4})"
    r"|(?:[a-z][A-Za-z'’\-]*(?:\s+[a-z][A-Za-z'’\-]*){0,3}))\s*,\s*[A-Z](?:\.[A-Z])?\."
)


class VenueLayoutError(RuntimeError):
    """The selected venue extractor or its retained MinerU input is unavailable."""


def locate_mineru_artifact(
    object_storage_path: Path, paper_id: str, paper_version_id: str
) -> Path | None:
    root = object_storage_path.resolve() / paper_id / paper_version_id / "mineru" / "raw_mineru"
    if not root.is_dir():
        return None
    layout = next(iter(sorted(root.rglob("layout.json"))), None)
    if layout is not None:
        return layout
    return next(iter(sorted(root.rglob("*_content_list.json"))), None)


def build_venue_layout_facts(
    *, venue_id: str, pdf_path: Path, mineru_json_path: Path
) -> dict[str, Any]:
    venue = venue_id.strip().lower()
    script_name = SUPPORTED_VENUE_EXTRACTORS.get(venue)
    if script_name is None:
        raise VenueLayoutError(f"no venue-specific layout extractor for {venue_id!r}")
    if not pdf_path.is_file():
        raise VenueLayoutError(f"PDF not found: {pdf_path}")
    if not mineru_json_path.is_file():
        raise VenueLayoutError(f"MinerU artifact not found: {mineru_json_path}")

    module = _load_extractor(script_name)
    payload = json.loads(mineru_json_path.read_text(encoding="utf-8"))
    if venue == "icml":
        native = module.extract_pymupdf(pdf_path)
        semantic = module.parse_mineru(payload, native["pages"])
        fused, quality = module.fuse_blocks(native, semantic)
        structure = module.derive_structure(fused, native["pages"])
        columns = module.derive_column_geometry(native, int(structure["main_body_page_count"]))
        derived = {
            "page_geometry": {
                "page_count": len(native["pages"]),
                "pages": native["pages"],
                "all_pages_same_size": len(
                    {
                        (round(page["width_pt"], 2), round(page["height_pt"], 2))
                        for page in native["pages"]
                    }
                )
                == 1,
            },
            "document_structure": structure,
            "column_geometry": columns,
            "front_matter": module.derive_front_matter(fused, native, columns),
            "abstract": module.derive_abstract(fused, columns),
            "captions": module.derive_captions(fused),
            "references": module.derive_references(fused, native, structure),
        }
        output = {
            "schema_version": module.SCHEMA_VERSION,
            "source": {"pdf": str(pdf_path.resolve()), "mineru": str(mineru_json_path.resolve())},
            "quality": quality,
            "derived_facts": derived,
            "rule_evidence_index": module.build_rule_evidence_index(derived),
            "fused_blocks": fused,
            "native_facts": native,
        }
        return module._round_floats(output)

    native = module.extract_pymupdf(pdf_path)
    semantic = module.parse_mineru(payload, native["pages"])
    fused = module.fuse_blocks(native, semantic)
    report = module.evaluate(native, fused)
    derived = module.build_derived_facts(native, report, fused)
    output = {
        "schema_version": module.SCHEMA_VERSION,
        "source": {"pdf": str(pdf_path.resolve()), "mineru": str(mineru_json_path.resolve())},
        "quality": {
            "mineru_block_count": len(fused),
            "matched_block_count": sum(item["match_status"] == "matched" for item in fused),
            "unmatched_block_count": sum(item["match_status"] != "matched" for item in fused),
            "pymupdf_span_count": len(native["spans"]),
            "pymupdf_line_count": len(native["lines"]),
            "pymupdf_object_count": len(native["objects"]),
        },
        "derived_facts": derived,
        "rule_evidence_index": module.build_rule_evidence_index(report["rules"]),
        "fused_blocks": module.serialize_fused_blocks(fused),
        "native_facts": native,
    }
    return module.round_floats(output)


def review_facts_from_fused(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten fused output without losing the extractor's structured measurements."""

    pages = {
        int(item.get("page_number") or 0): item
        for item in payload.get("native_facts", {}).get("pages", [])
        if isinstance(item, dict)
    }
    facts: list[dict[str, Any]] = []
    current_section = ""
    for block in payload.get("fused_blocks", []):
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "").strip()
        page_number = int(block.get("page_number") or 0)
        if not text or page_number <= 0:
            continue
        role = str(block.get("semantic_role") or "paragraph")
        if "heading" in role or role in {"title", "document_title"}:
            current_section = text
        style = block.get("style") if isinstance(block.get("style"), dict) else {}
        match = block.get("match") if isinstance(block.get("match"), dict) else {}
        page = pages.get(page_number, {})
        facts.append(
            {
                "evidence_id": f"P{len(facts) + 1}",
                "block_id": str(block.get("fused_block_id") or f"fused-{len(facts) + 1}"),
                "page_number": page_number,
                "bbox": block.get("bbox"),
                "quote": text,
                "role": role,
                "heading_level": block.get("heading_level"),
                "section_title": current_section,
                "font_name": style.get("dominant_font"),
                "font_size_pt": style.get("font_size_median_pt"),
                "is_bold": (
                    float(style.get("bold_character_ratio") or 0) >= 0.5
                    if "bold_character_ratio" in style
                    else None
                ),
                "baseline_gap_pt": style.get("baseline_gap_median_pt"),
                "page_width_pt": page.get("width_pt"),
                "page_height_pt": page.get("height_pt"),
                "source": "venue_fused_layout",
                "confidence": match.get("confidence"),
                "source_uri": f"fused://{block.get('fused_block_id') or len(facts) + 1}",
            }
        )

    derived = payload.get("derived_facts")
    if isinstance(derived, dict):
        for name, value in derived.items():
            # NeurIPS rule assessments are projected below as one compact,
            # category-routable fact per parsed rule.  Keeping the original
            # list here would repeat every measurement in every unit.
            if value is None or name == "rule_assessments":
                continue
            page_number = _measurement_page(value) or 1
            page = pages.get(page_number, next(iter(pages.values()), {}))
            width = float(page.get("width_pt") or 0)
            height = float(page.get("height_pt") or 0)
            facts.append(
                {
                    "evidence_id": f"P{len(facts) + 1}",
                    "block_id": f"derived-{name}",
                    "page_number": page_number,
                    "bbox": [0.0, 0.0, width, height] if width and height else None,
                    "quote": f"{name}: {_compact_json(value)}",
                    "role": f"derived_{name}",
                    "section_title": name,
                    "measurements": value,
                    "page_width_pt": width or None,
                    "page_height_pt": height or None,
                    "source": "venue_derived_facts",
                    "confidence": 1.0,
                    "source_uri": f"derived://{name}",
                }
            )
        facts.extend(_neurips_rule_assessment_facts(derived, pages, start_index=len(facts)))
        if "caption_geometry" not in derived:
            caption_geometry = _derive_caption_geometry(facts)
            if caption_geometry is not None:
                page_number = _measurement_page(caption_geometry) or 1
                page = pages.get(page_number, next(iter(pages.values()), {}))
                width = float(page.get("width_pt") or 0)
                height = float(page.get("height_pt") or 0)
                facts.append(
                    {
                        "evidence_id": f"P{len(facts) + 1}",
                        "block_id": "derived-caption-geometry",
                        "page_number": page_number,
                        "bbox": [0.0, 0.0, width, height] if width and height else None,
                        "quote": "caption_geometry",
                        "role": "derived_caption_geometry",
                        "section_title": "caption_geometry",
                        "measurements": caption_geometry,
                        "page_width_pt": width or None,
                        "page_height_pt": height or None,
                        "source": "venue_derived_facts",
                        "confidence": 1.0,
                        "source_uri": "derived://caption_geometry",
                    }
                )

        # These inventories retain the complete extraction population for
        # cross-block rules.  They are derived from the same fused/native PDF
        # facts and work for every paper, rather than carrying paper-specific
        # annotations into the review prompt.
        for name, value in _derive_review_aggregates(payload).items():
            if value is None:
                continue
            page_number = _measurement_page(value) or 1
            page = pages.get(page_number, next(iter(pages.values()), {}))
            width = float(page.get("width_pt") or 0)
            height = float(page.get("height_pt") or 0)
            facts.append(
                {
                    "evidence_id": f"P{len(facts) + 1}",
                    "block_id": f"derived-{name}",
                    "page_number": page_number,
                    "bbox": [0.0, 0.0, width, height] if width and height else None,
                    "quote": name,
                    "role": f"derived_{name}",
                    "section_title": name,
                    "measurements": value,
                    "page_width_pt": width or None,
                    "page_height_pt": height or None,
                    "source": "venue_derived_facts",
                    "confidence": float(value.get("confidence") or 0.0),
                    "source_uri": f"derived://{name}",
                }
            )

    for item in payload.get("native_facts", {}).get("objects", []):
        if not isinstance(item, dict) or not item.get("bbox"):
            continue
        page_number = int(item.get("page_number") or 0)
        if page_number <= 0:
            continue
        page = pages.get(page_number, {})
        object_type = str(item.get("object_type") or item.get("type") or "graphic")
        facts.append(
            {
                "evidence_id": f"P{len(facts) + 1}",
                "block_id": str(item.get("object_id") or f"native-object-{len(facts) + 1}"),
                "page_number": page_number,
                "bbox": item.get("bbox"),
                "quote": f"{object_type} object geometry; visual content not inspected",
                "role": f"native_{object_type}_object",
                "section_title": "",
                "page_width_pt": page.get("width_pt"),
                "page_height_pt": page.get("height_pt"),
                "source": "venue_native_object",
                "confidence": 1.0,
                "source_uri": f"native-object://{item.get('object_id') or len(facts) + 1}",
            }
        )
    return facts


def _neurips_rule_assessment_facts(
    derived: dict[str, Any], pages: dict[int, dict[str, Any]], *, start_index: int
) -> list[dict[str, Any]]:
    """Project NeurIPS deterministic checks without their nested raw prose.

    The NeurIPS extractor already checks complete populations for its ten
    numbered rule groups.  A group is still only evidence, not a model result:
    the LLM compares its immutable measurements against the atomic S* rule.
    Splitting the source list gives category selection an exact, compact
    inventory and avoids repeatedly passing title/abstract/body text.
    """

    assessments = derived.get("rule_assessments")
    if not isinstance(assessments, list):
        return []
    categories_by_rule = {
        "NIPS-01": ["page_layout"],
        "NIPS-02": ["page_layout"],
        "NIPS-03": ["page_layout"],
        "NIPS-04": ["heading"],
        "NIPS-05": ["abstract"],
        "NIPS-06": ["heading"],
        "NIPS-07": ["heading"],
        "NIPS-08": ["heading"],
        "NIPS-09": ["reference"],
        "NIPS-10": ["figure", "table"],
    }
    facts: list[dict[str, Any]] = []
    for item in assessments:
        if not isinstance(item, dict):
            continue
        rule_group = str(item.get("rule_id") or "")
        categories = categories_by_rule.get(rule_group)
        if not categories:
            continue
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        page_number = _measurement_page(evidence) or 1
        page = pages.get(page_number, next(iter(pages.values()), {}))
        width, height = _number(page.get("width_pt")), _number(page.get("height_pt"))
        facts.append(
            {
                "evidence_id": f"P{start_index + len(facts) + 1}",
                "block_id": f"derived-neurips-{rule_group.lower()}",
                "page_number": page_number,
                "bbox": [0.0, 0.0, width, height] if width and height else None,
                "quote": "derived_neurips_rule_assessment",
                "role": "derived_rule_assessment",
                "section_title": rule_group,
                "measurements": {
                    "parser_rule_group": rule_group,
                    "rule_categories": categories,
                    "parser_assessment": item.get("result"),
                    "confidence": _number(item.get("confidence")),
                    "coverage": "complete deterministic NeurIPS rule-group extraction",
                    "evidence": _compact_neurips_assessment_evidence(rule_group, evidence),
                },
                "page_width_pt": width,
                "page_height_pt": height,
                "source": "venue_derived_facts",
                "confidence": _number(item.get("confidence")) or 1.0,
                "source_uri": f"derived://neurips/{rule_group}",
            }
        )
    return facts


def _compact_neurips_assessment_evidence(rule_group: str, evidence: dict[str, Any]) -> dict[str, Any]:
    if rule_group == "NIPS-01":
        return {"body_geometry": _compact_neurips_body_geometry(evidence.get("body_geometry"))}
    if rule_group == "NIPS-02":
        return {
            key: evidence.get(key)
            for key in ("body_font_size_pt", "font_name", "times_compatible", "baseline_gap_pt")
        }
    if rule_group == "NIPS-03":
        return {"paragraph_metrics": _compact_neurips_paragraph_metrics(evidence.get("paragraph_metrics"))}
    if rule_group == "NIPS-04":
        return {"title": _compact_neurips_front_matter(evidence.get("title"))}
    if rule_group == "NIPS-05":
        return {"abstract": _compact_neurips_abstract(evidence.get("abstract"))}
    if rule_group in {"NIPS-06", "NIPS-07", "NIPS-08"}:
        return {"headings": _compact_neurips_headings(evidence.get("headings"))}
    if rule_group == "NIPS-09":
        return {
            "references_heading": _compact_neurips_reference_heading(evidence.get("references_heading")),
            "reference_entry_count": evidence.get("reference_entry_count"),
        }
    if rule_group == "NIPS-10":
        return _compact_neurips_caption_inventory(evidence)
    return {}


def _compact_neurips_body_geometry(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        key: source.get(key)
        for key in (
            "page_width_pt", "page_height_pt", "left_pt", "right_pt", "width_pt", "top_pt",
            "font_size_mode_pt", "font_name_mode", "font_name_times_compatible",
            "baseline_gap_median_pt", "sample_line_count", "sample_pages",
        )
    }


def _compact_neurips_paragraph_metrics(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        key: source.get(key)
        for key in (
            "paragraph_count_sampled", "first_line_indent_median_pt", "first_line_indent_abs_p90_pt",
            "paragraph_extra_gap_median_pt", "paragraph_extra_gap_samples",
        )
    }


def _compact_neurips_front_matter(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    title = source.get("title") if isinstance(source.get("title"), dict) else {}
    style = title.get("native_style") if isinstance(title.get("native_style"), dict) else {}
    title_bbox = source.get("title_bbox") or title.get("native_bbox") or title.get("bbox")
    top_rule_ys = _neurips_rule_ys(source.get("top_rule_candidates"))
    bottom_rule_ys = _neurips_rule_ys(source.get("bottom_rule_candidates"))
    title_between_rules = (
        isinstance(title_bbox, list)
        and len(title_bbox) == 4
        and bool(top_rule_ys)
        and bool(bottom_rule_ys)
        and max(top_rule_ys) <= float(title_bbox[1])
        and min(bottom_rule_ys) >= float(title_bbox[3])
    )
    return {
        "title_page_number": title.get("page_number"),
        "title_bbox": title_bbox,
        "title_style": {key: style.get(key) for key in ("font_name", "font_size_pt", "bold_ratio", "line_count")},
        "title_alignment_delta_pt": source.get("title_alignment_delta_pt"),
        "top_rule_count": len(source.get("top_rule_candidates") or []),
        "bottom_rule_count": len(source.get("bottom_rule_candidates") or []),
        "horizontal_rule_count": len(source.get("horizontal_rules") or []),
        "top_rule_y_positions_pt": top_rule_ys,
        "bottom_rule_y_positions_pt": bottom_rule_ys,
        "title_between_top_and_bottom_rules": title_between_rules,
    }


def _neurips_rule_ys(value: Any) -> list[float]:
    rows = value if isinstance(value, list) else []
    positions: list[float] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        box = _bbox(item.get("bbox"))
        if box is not None:
            positions.append(round((box[1] + box[3]) / 2, 4))
            continue
        start, end = item.get("start"), item.get("end")
        if (
            isinstance(start, list)
            and isinstance(end, list)
            and len(start) == len(end) == 2
            and isinstance(start[1], (int, float))
            and isinstance(end[1], (int, float))
        ):
            positions.append(round((float(start[1]) + float(end[1])) / 2, 4))
    return sorted(positions)


def _compact_neurips_abstract(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    heading = source.get("heading") if isinstance(source.get("heading"), dict) else {}
    style = source.get("style") if isinstance(source.get("style"), dict) else {}
    heading_style = source.get("heading_style") if isinstance(source.get("heading_style"), dict) else {}
    return {
        "heading_page_number": heading.get("page_number"),
        "heading_bbox": heading.get("native_bbox") or heading.get("bbox"),
        "heading_style": {key: heading_style.get(key) for key in ("font_name", "font_size_pt", "bold_ratio", "line_count")},
        "heading_alignment_delta_pt": source.get("heading_alignment_delta_pt"),
        "body_bbox": source.get("body_bbox"),
        "paragraph_count": source.get("paragraph_count"),
        "line_count": source.get("line_count"),
        "left_indent_pt": source.get("left_indent_pt"),
        "right_indent_pt": source.get("right_indent_pt"),
        "body_style": {key: style.get(key) for key in ("font_name", "font_size_pt", "bold_ratio", "baseline_gap_pt")},
    }


def _compact_neurips_headings(value: Any) -> dict[str, Any]:
    rows = value if isinstance(value, list) else []
    inventory: dict[str, Any] = {}
    for depth in (1, 2, 3):
        items = [item for item in rows if isinstance(item, dict) and item.get("depth") == depth]
        styles = [item.get("style") for item in items if isinstance(item.get("style"), dict)]
        sizes = [item.get("font_size_pt") for item in styles if isinstance(item.get("font_size_pt"), (int, float))]
        bold = [item.get("bold_ratio") for item in styles if isinstance(item.get("bold_ratio"), (int, float))]
        lefts = [item["bbox"][0] for item in items if isinstance(item.get("bbox"), list) and item["bbox"]]
        inventory[f"depth_{depth}"] = {
            "count": len(items),
            "pages": sorted({item.get("page_number") for item in items if isinstance(item.get("page_number"), int)}),
            "font_size_values_pt": sorted({round(float(item), 4) for item in sizes}),
            "bold_ratio_range": [round(min(bold), 4), round(max(bold), 4)] if bold else None,
            "left_x_values_pt": sorted({round(float(item), 4) for item in lefts}),
        }
    return inventory


def _compact_neurips_reference_heading(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    style = value.get("native_style") if isinstance(value.get("native_style"), dict) else {}
    return {
        "page_number": value.get("page_number"),
        "text": value.get("text"),
        "bbox": value.get("native_bbox") or value.get("bbox"),
        "is_unnumbered": not bool(
            re.match(r"^\d+(?:\.\d+)*\.?\s+", str(value.get("text") or "").strip())
        ),
        "style": {key: style.get(key) for key in ("font_name", "font_size_pt", "bold_ratio", "line_count")},
    }


def _compact_neurips_caption_inventory(value: dict[str, Any]) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for kind in ("figures", "tables"):
        source = value.get(kind)
        if not isinstance(source, dict):
            continue
        items = [item for item in source.get("items", []) if isinstance(item, dict)]
        positions = [item.get("position") for item in items]
        centered = [item.get("centered") for item in items]
        inventory[kind] = {
            "count": source.get("count"),
            "numbers": source.get("numbers"),
            "unique": source.get("unique"),
            "continuous": source.get("continuous"),
            "caption_below_count": sum(position == "caption_below" for position in positions),
            "caption_not_below_numbers": [
                item.get("number") for item in items if item.get("position") != "caption_below"
            ],
            "locally_centered_count": sum(item is True for item in centered),
            "not_centered_numbers": [
                item.get("number") for item in items if item.get("centered") is False
            ],
        }
    return inventory


def _derive_review_aggregates(payload: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    """Build compact, auditable full-population facts for cross-block rules."""

    derived = payload.get("derived_facts")
    derived = derived if isinstance(derived, dict) else {}
    blocks = [
        item
        for item in payload.get("fused_blocks", [])
        if isinstance(item, dict) and _block_bbox(item) is not None
    ]
    if not blocks:
        return {}

    structure = derived.get("document_structure")
    structure = structure if isinstance(structure, dict) else {}
    references_page = _nested_page(structure.get("references_start"))
    appendix_page = _nested_page(structure.get("appendix_start"))
    columns = derived.get("column_geometry")
    columns = columns if isinstance(columns, dict) else {}
    main_paragraphs, appendix_paragraphs = _review_paragraph_populations(
        blocks, references_page, appendix_page
    )
    return {
        "front_matter_geometry": _front_matter_geometry(derived, columns),
        "typography_inventory": _typography_inventory(
            main_paragraphs, appendix_paragraphs, references_page, appendix_page
        ),
        "heading_inventory": _heading_inventory(blocks, columns, references_page, appendix_page),
        "citation_inventory": _citation_inventory(main_paragraphs),
        "reference_inventory": _reference_inventory(blocks, references_page, appendix_page),
        "appendix_layout": _appendix_layout_inventory(
            payload, main_paragraphs, appendix_paragraphs, columns, appendix_page
        ),
    }


def _front_matter_geometry(
    derived: dict[str, Any], columns: dict[str, Any]
) -> dict[str, Any] | None:
    front_matter = derived.get("front_matter")
    if not isinstance(front_matter, dict):
        return None
    title = front_matter.get("title")
    title = title if isinstance(title, dict) else {}
    title_bbox = _bbox(title.get("bbox"))
    authors = [item for item in front_matter.get("author_blocks", []) if isinstance(item, dict)]
    author_bboxes = [_bbox(item.get("bbox")) for item in authors]
    author_bboxes = [item for item in author_bboxes if item is not None]
    text_left = _number(columns.get("text_left_pt"))
    text_right = _number(columns.get("text_right_pt"))
    body_center = (text_left + text_right) / 2 if text_left is not None and text_right is not None else None
    rules = [item for item in title.get("horizontal_rules", []) if isinstance(item, dict)]
    rule_rows = []
    for rule in rules:
        start, end = rule.get("start"), rule.get("end")
        if not (
            isinstance(start, list)
            and isinstance(end, list)
            and len(start) == 2
            and len(end) == 2
            and all(isinstance(value, (int, float)) for value in start + end)
        ):
            continue
        if abs(float(start[1]) - float(end[1])) > 1.0:
            continue
        rule_rows.append(
            {
                "y_pt": round((float(start[1]) + float(end[1])) / 2, 4),
                "width_pt": round(float(rule.get("width_pt") or 0), 4),
                "length_pt": round(abs(float(end[0]) - float(start[0])), 4),
            }
        )
    rule_rows.sort(key=lambda item: item["y_pt"])
    title_center = (title_bbox[0] + title_bbox[2]) / 2 if title_bbox else None
    author_center = (
        (min(item[0] for item in author_bboxes) + max(item[2] for item in author_bboxes)) / 2
        if author_bboxes
        else None
    )
    lower_rule = rule_rows[-1] if rule_rows else None
    author_top = min(item[1] for item in author_bboxes) if author_bboxes else None
    return {
        "page_number": 1,
        "coverage": {
            "title_present": bool(title_bbox),
            "author_block_count": len(authors),
            "horizontal_rule_count": len(rule_rows),
        },
        "body_center_x_pt": _rounded(body_center),
        "title_center_offset_pt": _rounded(abs(title_center - body_center)) if title_center is not None and body_center is not None else None,
        "author_center_offset_pt": _rounded(abs(author_center - body_center)) if author_center is not None and body_center is not None else None,
        "horizontal_rules": rule_rows,
        "author_after_lower_title_rule": bool(author_top is not None and lower_rule and author_top > lower_rule["y_pt"]),
        "author_gap_from_lower_title_rule_pt": _rounded(author_top - lower_rule["y_pt"]) if author_top is not None and lower_rule else None,
        "title_style": _style_inventory([{"style": title.get("style")}]),
        "author_style": _style_inventory(authors),
        "derivation": "front-matter title, author blocks, and native drawing rules",
        "confidence": 1.0,
    }


def _typography_inventory(
    main_paragraphs: list[dict[str, Any]],
    appendix_paragraphs: list[dict[str, Any]],
    references_page: int | None,
    appendix_page: int | None,
) -> dict[str, Any]:
    main_prose, excluded_formula_like = _main_prose_typography_population(main_paragraphs)
    appendix_prose, appendix_excluded_formula_like = _main_prose_typography_population(
        appendix_paragraphs
    )
    return {
        "page_number": 1,
        "main_body": _population_style_inventory(main_prose),
        "appendix_body": _population_style_inventory(appendix_prose),
        "coverage": {
            "main_body_end_before_references_page": references_page,
            "appendix_start_page": appendix_page,
            "main_body_source_paragraph_count": len(main_paragraphs),
            "main_body_excluded_formula_like_count": len(excluded_formula_like),
            "appendix_source_paragraph_count": len(appendix_paragraphs),
            "appendix_excluded_formula_like_count": len(appendix_excluded_formula_like),
        },
        "derivation": (
            "all fused paragraph blocks after the first numbered heading, excluding "
            "figure/table captions and formula-like paragraphs identified by math font "
            "plus multiple LaTex markers"
        ),
        "confidence": 0.98,
    }


def _heading_inventory(
    blocks: list[dict[str, Any]],
    columns: dict[str, Any],
    references_page: int | None,
    appendix_page: int | None,
) -> dict[str, Any]:
    numbered_pattern = re.compile(r"^\d+(?:\.\d+)*\.\s+")
    headings = [
        item
        for item in blocks
        if str(item.get("semantic_role") or "") == "section_heading"
        and _is_main_heading(item, references_page, appendix_page, numbered_pattern)
    ]
    levels = {
        "level_1": re.compile(r"^\d+\.\s+"),
        "level_2": re.compile(r"^\d+\.\d+\.\s+"),
        "level_3": re.compile(r"^\d+\.\d+\.\d+\.\s+"),
    }
    text_left, text_right = _number(columns.get("text_left_pt")), _number(columns.get("text_right_pt"))
    right_start = _number(columns.get("right_column_start_pt"))
    left_start = _number(columns.get("left_column_start_pt"))
    level_rows: dict[str, dict[str, Any]] = {}
    for label, pattern in levels.items():
        items = [item for item in headings if pattern.match(str(item.get("text") or "").strip())]
        offsets = []
        overlap_count = 0
        for item in items:
            bbox = _block_bbox(item)
            if bbox is None:
                continue
            expected_start = left_start
            if right_start is not None and text_left is not None and text_right is not None:
                expected_start = right_start if (bbox[0] + bbox[2]) / 2 > (text_left + text_right) / 2 else left_start
            if expected_start is not None:
                offsets.append(abs(bbox[0] - expected_start))
            if _heading_overlaps_text_neighbor(item, blocks):
                overlap_count += 1
        level_rows[label] = {
            "count": len(items),
            "format_match_count": len(items),
            "styles": _style_inventory(items),
            "left_alignment_within_3pt_count": sum(value <= 3 for value in offsets),
            "left_alignment_max_offset_pt": _rounded(max(offsets)) if offsets else None,
            "overlap_count": overlap_count,
            "samples": [_block_sample(item) for item in items[:8]],
        }
    all_numbered = [
        item
        for item in headings
        if any(pattern.match(str(item.get("text") or "").strip()) for pattern in levels.values())
    ]
    return {
        "page_number": 1,
        "coverage": {"numbered_heading_count": len(all_numbered), "pages": _pages_for(all_numbered)},
        "levels": level_rows,
        "title_case": _title_case_inventory(all_numbered),
        "max_detected_level": max((label for label, row in level_rows.items() if row["count"]), default=None),
        "derivation": "all fused section_heading blocks matched against ICML dotted-number patterns",
        "confidence": 0.98,
    }


def _citation_inventory(main_paragraphs: list[dict[str, Any]]) -> dict[str, Any]:
    author_year = re.compile(r"(?:\([A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?(?:,\s*|\s+)\d{4}[a-z]?(?:;[^)]*)?\)|\b[A-Z][A-Za-z'’-]+\s+et\s+al\.,?\s*\d{4}[a-z]?)")
    numeric = re.compile(r"\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]")
    text = "\n".join(str(item.get("text") or "") for item in main_paragraphs)
    author_year_count = len(author_year.findall(text))
    numeric_count = len(numeric.findall(text))
    total = author_year_count + numeric_count
    return {
        "page_number": 1,
        "coverage": {"paragraph_count": len(main_paragraphs), "pages": _pages_for(main_paragraphs)},
        "author_year_citation_count": author_year_count,
        "numeric_citation_count": numeric_count,
        "author_year_majority": author_year_count > numeric_count,
        "numeric_majority": numeric_count > author_year_count,
        "detected_citation_count": total,
        "derivation": "whole-body lexical citation detectors for author-year and bracketed numeric forms",
        "confidence": 0.95,
    }


def _reference_inventory(
    blocks: list[dict[str, Any]], references_page: int | None, appendix_page: int | None
) -> dict[str, Any] | None:
    entries = [
        item
        for item in blocks
        if str(item.get("semantic_role") or "") == "reference_entry"
        and (references_page is None or int(item.get("page_number") or 0) >= references_page)
        and (appendix_page is None or int(item.get("page_number") or 0) < appendix_page)
    ]
    if not entries:
        return None
    entries = _reference_reading_order(entries)
    logical_entries = _logical_reference_entries(entries)
    extracted_author_keys = [
        _first_author_key(str(item.get("text") or "")) for item in logical_entries
    ]
    author_keys = [key for key in extracted_author_keys if key is not None]
    ordering_violations = [
        {"previous": previous, "current": current}
        for previous, current in zip(author_keys, author_keys[1:], strict=False)
        if current < previous
    ]
    years_present = [
        bool(re.search(r"\b(?:19|20)\d{2}[a-z]?\b", str(item.get("text") or "")))
        for item in logical_entries
    ]
    text_font_entries = [
        item for item in logical_entries if not _is_url_dominant_reference_entry(item)
    ]
    heading = next(
        (
            item
            for item in blocks
            if str(item.get("semantic_role") or "")
            in {"section_heading", "title", "document_title"}
            and str(item.get("text") or "").strip().casefold() == "references"
        ),
        None,
    )
    primary_headings = [
        item
        for item in blocks
            if str(item.get("semantic_role") or "")
            in {"section_heading", "title", "document_title"}
            and re.match(r"^\d+\.\s+", str(item.get("text") or "").strip())
    ]
    heading_style = _style_inventory([heading]) if heading else None
    primary_style = _style_inventory(primary_headings) if primary_headings else None
    return {
        "page_number": int(heading.get("page_number") or references_page or 1) if heading else references_page or 1,
        "coverage": {
            "source_block_count": len(entries),
            "entry_count": len(logical_entries),
            "pages": _pages_for(logical_entries),
            "url_dominant_entry_count": len(logical_entries) - len(text_font_entries),
            "sortable_first_author_count": len(author_keys),
            "unparsed_first_author_count": len(logical_entries) - len(author_keys),
        },
        "styles": _style_inventory(text_font_entries),
        "year_present_count": sum(years_present),
        "all_entries_have_year": all(years_present),
        "first_author_keys_in_reading_order": author_keys,
        "first_author_ordering_violations": ordering_violations,
        "first_author_ordered": not ordering_violations,
        "reading_order": "page ascending, left column top-to-bottom, right column top-to-bottom",
        "reference_heading": {
            "text": str(heading.get("text") or "") if heading else None,
            "is_unnumbered": bool(heading) and not bool(re.match(r"^\d+", str(heading.get("text") or "").strip())),
            "style": heading_style,
            "matches_primary_heading_style": _style_signature(heading_style) == _style_signature(primary_style)
            if heading_style and primary_style
            else None,
        },
        "derivation": (
            "all fused reference_entry blocks reordered by page, left-column-to-right-column "
            "reading order and y position, then parser-split continuations merged"
        ),
        "confidence": 0.99,
    }


def _appendix_layout_inventory(
    payload: dict[str, Any],
    main_paragraphs: list[dict[str, Any]],
    appendix_paragraphs: list[dict[str, Any]],
    columns: dict[str, Any],
    appendix_page: int | None,
) -> dict[str, Any]:
    appendix_headings = [
        item
        for item in payload.get("fused_blocks", [])
        if isinstance(item, dict)
        and str(item.get("semantic_role") or "") == "section_heading"
        and (appendix_page is not None and int(item.get("page_number") or 0) >= appendix_page)
        and re.match(r"^[A-Z]\.\s+", str(item.get("text") or "").strip())
    ]
    letters = [str(item.get("text") or "").strip()[0] for item in appendix_headings]
    expected_letters = [chr(ord("A") + index) for index in range(len(letters))]
    native = payload.get("native_facts") if isinstance(payload.get("native_facts"), dict) else {}
    pages = native.get("pages") if isinstance(native.get("pages"), list) else []
    lines = native.get("lines") if isinstance(native.get("lines"), list) else []
    body_pages = _pages_for(main_paragraphs)
    appendix_pages = _pages_for(appendix_paragraphs)
    return {
        "page_number": appendix_page or 1,
        "coverage": {
            "appendix_page_count": len(appendix_pages),
            "appendix_pages": appendix_pages,
            "body_pages": body_pages,
        },
        "body_columns": _column_inventory(main_paragraphs, columns),
        "appendix_columns": _column_inventory(appendix_paragraphs, columns),
        "body_page_numbers": _page_number_inventory(lines, pages, set(body_pages)),
        "appendix_page_numbers": _page_number_inventory(lines, pages, set(appendix_pages)),
        "appendix_heading_count": len(letters),
        "appendix_letters": letters,
        "appendix_letters_continuous": letters == expected_letters,
        "derivation": "fused paragraph geometry, native page text lines, and appendix heading sequence",
        "confidence": 0.97,
    }


def _review_paragraph_populations(
    blocks: list[dict[str, Any]], references_page: int | None, appendix_page: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first_main_heading = min(
        (
            item
            for item in blocks
            if re.match(r"^\d+(?:\.\d+)*\.?\s+", str(item.get("text") or "").strip())
        ),
        key=lambda item: (int(item.get("page_number") or 0), _block_bbox(item)[1]),
        default=None,
    )
    first_page = int(first_main_heading.get("page_number") or 0) if first_main_heading else None
    first_y = _block_bbox(first_main_heading)[1] if first_main_heading else None
    main: list[dict[str, Any]] = []
    appendix: list[dict[str, Any]] = []
    for item in blocks:
        if str(item.get("semantic_role") or "") != "paragraph":
            continue
        page = int(item.get("page_number") or 0)
        bbox = _block_bbox(item)
        if page <= 0 or bbox is None:
            continue
        if appendix_page is not None and page >= appendix_page:
            appendix.append(item)
            continue
        if references_page is not None and page > references_page:
            continue
        if first_page is not None and page == first_page and first_y is not None and bbox[1] < first_y:
            continue
        if first_page is not None and page < first_page:
            continue
        main.append(item)
    return main, appendix


def _population_style_inventory(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(items),
        "pages": _pages_for(items),
        "styles": _style_inventory(items),
    }


def _main_prose_typography_population(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prose: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for item in items:
        if _is_formula_like_paragraph(item):
            excluded.append(item)
        else:
            prose.append(item)
    return prose, excluded


def _is_formula_like_paragraph(item: dict[str, Any]) -> bool:
    """Keep mathematical display text out of a prose-main-font population."""

    style = item.get("style") if isinstance(item.get("style"), dict) else {}
    font = str(style.get("dominant_font") or "")
    math_font = bool(re.match(r"^(?:CM(?:MI|R|SY|EX)|LM(?:Math|Mono))", font, re.IGNORECASE))
    monospace_font = "mono" in font.casefold()
    text = str(item.get("text") or "")
    latex_marker_count = len(re.findall(r"\\[A-Za-z]+|[_^{}]", text))
    code_marker_count = len(re.findall(r"#|\b\w+_\w+\b|\w+\[[^\]]+\]", text))
    return (math_font and latex_marker_count >= 2) or (monospace_font and code_marker_count >= 2)


def _style_inventory(items: list[dict[str, Any]]) -> dict[str, Any]:
    styles = [item.get("style") for item in items if isinstance(item.get("style"), dict)]
    fonts = sorted({str(style.get("dominant_font")) for style in styles if style.get("dominant_font")})
    sizes = [_number(style.get("font_size_median_pt")) for style in styles]
    baselines = [_number(style.get("baseline_gap_median_pt")) for style in styles]
    bold = [_number(style.get("bold_character_ratio")) for style in styles]
    sizes = [item for item in sizes if item is not None]
    baselines = [item for item in baselines if item is not None]
    bold = [item for item in bold if item is not None]
    return {
        "font_names": fonts,
        "font_size_mode_pt": _rounded_mode(sizes),
        "baseline_gap_median_pt": _rounded(median(baselines)) if baselines else None,
        "bold_character_ratio_min": _rounded(min(bold)) if bold else None,
        "bold_character_ratio_median": _rounded(median(bold)) if bold else None,
        "styled_block_count": len(styles),
    }


def _column_inventory(items: list[dict[str, Any]], columns: dict[str, Any]) -> dict[str, Any]:
    bboxes = [_block_bbox(item) for item in items]
    bboxes = [item for item in bboxes if item is not None]
    left_start = _number(columns.get("left_column_start_pt"))
    right_start = _number(columns.get("right_column_start_pt"))
    if not bboxes or left_start is None or right_start is None:
        return {"paragraph_count": len(items)}
    center = (left_start + right_start) / 2
    left = [bbox[0] for bbox in bboxes if (bbox[0] + bbox[2]) / 2 < center]
    right = [bbox[0] for bbox in bboxes if (bbox[0] + bbox[2]) / 2 >= center]
    left_mode = _rounded_mode(left)
    right_mode = _rounded_mode(right)
    return {
        "paragraph_count": len(items),
        "left_column_start_pt": left_mode,
        "right_column_start_pt": right_mode,
        "gutter_pt": _rounded(right_mode - left_mode - float(columns.get("column_width_pt") or 0)) if left_mode is not None and right_mode is not None else None,
    }


def _page_number_inventory(lines: list[Any], pages: list[Any], page_set: set[int]) -> dict[str, Any]:
    heights = {
        int(item.get("page_number") or 0): _number(item.get("height_pt"))
        for item in pages
        if isinstance(item, dict)
    }
    positions: Counter[str] = Counter()
    numbered_pages: set[int] = set()
    for line in lines:
        if not isinstance(line, dict):
            continue
        page = int(line.get("page_number") or 0)
        bbox = _bbox(line.get("bbox"))
        text = str(line.get("text") or "").strip()
        height = heights.get(page)
        if page not in page_set or bbox is None or height is None or not re.fullmatch(r"\d+", text):
            continue
        if bbox[1] <= height * 0.12:
            positions["top"] += 1
            numbered_pages.add(page)
        elif bbox[3] >= height * 0.88:
            positions["bottom"] += 1
            numbered_pages.add(page)
    return {
        "covered_page_count": len(page_set),
        "numbered_page_count": len(numbered_pages),
        "positions": dict(sorted(positions.items())),
    }


def _is_main_heading(
    item: dict[str, Any],
    references_page: int | None,
    appendix_page: int | None,
    numbered_pattern: re.Pattern[str],
) -> bool:
    page = int(item.get("page_number") or 0)
    text = str(item.get("text") or "").strip()
    if appendix_page is not None and page >= appendix_page:
        return False
    if references_page is not None and page >= references_page:
        return False
    return bool(numbered_pattern.match(text))


def _heading_overlaps_text_neighbor(item: dict[str, Any], blocks: list[dict[str, Any]]) -> bool:
    bbox = _block_bbox(item)
    if bbox is None:
        return False
    page = int(item.get("page_number") or 0)
    for other in blocks:
        if other is item or int(other.get("page_number") or 0) != page:
            continue
        if str(other.get("semantic_role") or "") not in {"paragraph", "section_heading"}:
            continue
        if _rectangles_overlap(bbox, _block_bbox(other)):
            return True
    return False


def _block_bbox(item: dict[str, Any] | None) -> list[float] | None:
    return _bbox(item.get("bbox")) if isinstance(item, dict) else None


def _block_sample(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_number": int(item.get("page_number") or 0),
        "bbox": _block_bbox(item),
        "text": str(item.get("text") or "")[:160],
    }


def _pages_for(items: list[dict[str, Any]]) -> list[int]:
    return sorted({int(item.get("page_number") or 0) for item in items if int(item.get("page_number") or 0) > 0})


def _nested_page(value: Any) -> int | None:
    if isinstance(value, dict):
        page = value.get("page_number")
        return int(page) if isinstance(page, int) and page > 0 else None
    return int(value) if isinstance(value, int) and value > 0 else None


def _rectangles_overlap(first: list[float] | None, second: list[float] | None) -> bool:
    if first is None or second is None:
        return False
    horizontal = min(first[2], second[2]) - max(first[0], second[0])
    vertical = min(first[3], second[3]) - max(first[1], second[1])
    return horizontal > 0.5 and vertical > 0.5


def _first_author_key(text: str) -> str | None:
    """Return a sortable surname after removing PDF page/header URL fragments."""

    normalized = re.sub(r"^\s*(?:NIPS\s+\d{4}\s+|https?://(?:www\.)?\s*)", "", text)
    match = re.search(_REFERENCE_AUTHOR_PATTERN, normalized)
    if match is not None:
        return _reference_sort_key(match.group(1))
    # A small number of institutional authors do not use the surname/initial
    # syntax, but are still real bibliography entries rather than continuations.
    corporate = re.match(r"^([A-Z][A-Za-z'’\-]+)\.", normalized)
    return _reference_sort_key(corporate.group(1)) if corporate is not None else None


def _reference_sort_key(value: str) -> str | None:
    """Normalize a Latin surname while retaining standalone accent collation."""

    # Some Type-1 fonts expose a diaeresis as a separate glyph before the
    # accented vowel (``K¨arkk¨ainen``).  Keep it as a high collation marker
    # rather than dropping the author or silently treating ä as plain a.
    normalized = unicodedata.normalize("NFKD", value.replace("¨", "~"))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    key = re.sub(r"[^a-z~]", "", normalized.casefold())
    return key or None


def _is_reference_entry_start(text: str) -> bool:
    """Identify a new bibliography entry without treating wrapped prose as one."""

    normalized = re.sub(r"^\s*(?:NIPS\s+\d{4}\s+|https?://(?:www\.)?\s*)", "", text)
    if re.match(_REFERENCE_AUTHOR_PATTERN, normalized):
        return True
    return bool(re.match(r"^(?:Google)\.", normalized))


def _reference_reading_order(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct the visual reading order for a two-column bibliography.

    MinerU block order is an ingestion detail, not a semantic contract.  For
    each page, the largest horizontal gap between reference block starts is
    used as the column gutter; blocks are then read top-to-bottom in the left
    column before the right column.  A single-column page remains top-to-bottom.
    """

    by_page: dict[int, list[dict[str, Any]]] = {}
    for item in entries:
        page = int(item.get("page_number") or 0)
        bbox = _block_bbox(item)
        if page > 0 and bbox is not None:
            by_page.setdefault(page, []).append(item)

    ordered: list[dict[str, Any]] = []
    for page in sorted(by_page):
        page_entries = by_page[page]
        starts = sorted({round(float(_block_bbox(item)[0]), 2) for item in page_entries})
        split_at: float | None = None
        if len(starts) > 1:
            gaps = [
                (right - left, left, right)
                for left, right in zip(starts, starts[1:], strict=False)
            ]
            gap, left, right = max(gaps, key=lambda row: row[0])
            if gap >= 24.0:
                split_at = (left + right) / 2

        def key(
            item: dict[str, Any], split: float | None = split_at
        ) -> tuple[int, float, float]:
            bbox = _block_bbox(item) or [0.0, 0.0, 0.0, 0.0]
            column = 0 if split is None or bbox[0] < split else 1
            return column, float(bbox[1]), float(bbox[0])

        ordered.extend(sorted(page_entries, key=key))
    return ordered


def _logical_reference_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge parser-split bibliography continuations before checking sort order."""

    logical: list[dict[str, Any]] = []
    for item in entries:
        text = str(item.get("text") or "").strip()
        if logical and not _is_reference_entry_start(text):
            logical[-1] = {**logical[-1], "text": f"{logical[-1]['text']} {text}"}
        else:
            logical.append(dict(item))
    return logical


def _is_url_dominant_reference_entry(item: dict[str, Any]) -> bool:
    """Exclude URL typesetting from the bibliography's prose-font population."""

    style = item.get("style") if isinstance(item.get("style"), dict) else {}
    font = str(style.get("dominant_font") or "").casefold()
    text = str(item.get("text") or "")
    is_monospace = "mono" in font or font.startswith("nimbusmon")
    return is_monospace and bool(re.search(r"\b(?:url|https?://|www\.)", text, re.IGNORECASE))


def _title_case_inventory(headings: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate all numbered headings without requiring raw heading prose in prompts."""

    violations: list[str] = []
    for item in headings:
        text = re.sub(r"^\d+(?:\.\d+)*\.\s+", "", str(item.get("text") or "").strip())
        if not _is_title_case(text):
            violations.append(text[:160])
    return {
        "checked_heading_count": len(headings),
        "all_content_words_title_case": not violations,
        "violations": violations[:16],
    }


def _is_title_case(text: str) -> bool:
    minor_words = {
        "a", "an", "and", "am", "are", "as", "at", "be", "been", "being", "but", "by",
        "for", "from", "in", "is", "nor", "of", "on", "or", "so", "the", "to", "was", "were",
        "with", "yet",
    }
    # PyMuPDF exposes common ligatures as one Unicode code point. Normalize
    # them before tokenization so ``Sufﬁciently`` stays one title-case word.
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’-]*", text) if word]
    for index, word in enumerate(words):
        normalized = word.casefold().strip("'’-" )
        if index not in {0, len(words) - 1} and normalized in minor_words:
            continue
        if word[0].islower():
            return False
    return True


def _style_signature(style: dict[str, Any] | None) -> tuple[Any, Any, Any] | None:
    if not isinstance(style, dict):
        return None
    return (
        tuple(style.get("font_names") or []),
        style.get("font_size_mode_pt"),
        style.get("bold_character_ratio_median"),
    )


def _rounded_mode(values: list[float]) -> float | None:
    if not values:
        return None
    counts = Counter(round(float(item), 1) for item in values)
    return _rounded(min((value for value, count in counts.items() if count == max(counts.values())), default=None))


def _rounded(value: float | None) -> float | None:
    return round(float(value), 4) if value is not None else None


def _load_extractor(script_name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[3] / script_name
    if not script_path.is_file():
        raise VenueLayoutError(f"venue extractor not found: {script_path}")
    module_name = f"format_review_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise VenueLayoutError(f"cannot load venue extractor: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measurement_page(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("page_number", "page", "references_start", "appendix_start"):
            candidate = value.get(key)
            if isinstance(candidate, int) and candidate > 0:
                return candidate
        for nested in value.values():
            candidate = _measurement_page(nested)
            if candidate:
                return candidate
    elif isinstance(value, list):
        for nested in value:
            candidate = _measurement_page(nested)
            if candidate:
                return candidate
    return None


def _derive_caption_geometry(facts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Derive complete caption alignment and body-containment aggregates.

    This is based entirely on extracted bboxes, styles, and the venue's
    column geometry.  It deliberately reports unavailable geometry instead
    of fabricating a body region when the extractor cannot establish one.
    """

    captions = next(
        (
            item.get("measurements")
            for item in facts
            if item.get("role") == "derived_captions" and isinstance(item.get("measurements"), dict)
        ),
        None,
    )
    columns = next(
        (
            item.get("measurements")
            for item in facts
            if item.get("role") in {"derived_column_geometry", "derived_body_geometry"}
            and isinstance(item.get("measurements"), dict)
        ),
        None,
    )
    if not isinstance(captions, dict) or not isinstance(columns, dict):
        return None
    text_left = _number(columns.get("text_left_pt") or columns.get("left_pt"))
    text_right = _number(columns.get("text_right_pt") or columns.get("right_pt"))
    if text_left is None or text_right is None or text_right <= text_left:
        return None

    left_column_end = _number(columns.get("left_ink_end_pt"))
    right_column_start = _number(columns.get("right_column_start_pt"))
    body_y_by_page = _body_vertical_envelopes(facts, text_left, text_right)
    result: dict[str, Any] = {
        "body_horizontal_bounds_pt": [round(text_left, 4), round(text_right, 4)],
        "body_vertical_envelope_page_count": len(body_y_by_page),
    }
    for kind in ("figures", "tables"):
        source = captions.get(kind)
        if not isinstance(source, dict):
            continue
        items = [item for item in source.get("items", []) if isinstance(item, dict)]
        single_count = centered_count = multi_count = left_aligned_count = 0
        not_centered: list[int] = []
        not_left_aligned: list[int] = []
        inside_count = 0
        outside: list[int] = []
        geometry_complete = 0
        for item in items:
            number = item.get("number")
            label = number if isinstance(number, int) else None
            caption_bbox = _bbox(item.get("caption_bbox"))
            object_bbox = _bbox(item.get("paired_object_bbox"))
            if caption_bbox is not None:
                line_count = _caption_line_count(caption_bbox, item.get("caption_style"))
                region_left, region_right = _caption_region(
                    caption_bbox,
                    text_left,
                    text_right,
                    left_column_end,
                    right_column_start,
                )
                if line_count == 1:
                    single_count += 1
                    caption_center = (caption_bbox[0] + caption_bbox[2]) / 2
                    region_center = (region_left + region_right) / 2
                    center_offset = abs(caption_center - region_center)
                    if center_offset <= 8:
                        centered_count += 1
                    elif label is not None:
                        not_centered.append(label)
                else:
                    multi_count += 1
                    left_offset = abs(caption_bbox[0] - region_left)
                    if left_offset <= 3:
                        left_aligned_count += 1
                    elif label is not None:
                        not_left_aligned.append(label)
            if object_bbox is not None:
                page_number = int(item.get("page_number") or 0)
                vertical = body_y_by_page.get(page_number)
                if vertical is not None:
                    geometry_complete += 1
                    inside = (
                        object_bbox[0] >= text_left - 3
                        and object_bbox[2] <= text_right + 3
                        and object_bbox[1] >= vertical[0] - 3
                        and object_bbox[3] <= vertical[1] + 3
                    )
                    if inside:
                        inside_count += 1
                    elif label is not None:
                        outside.append(label)
        result[kind] = {
            "count": source.get("count"),
            "single_line_caption_count": single_count,
            "single_line_centered_count": centered_count,
            "single_line_not_centered_numbers": sorted(not_centered),
            "multiline_caption_count": multi_count,
            "multiline_left_aligned_count": left_aligned_count,
            "multiline_not_left_aligned_numbers": sorted(not_left_aligned),
            "object_geometry_complete_count": geometry_complete,
            "object_inside_body_count": inside_count,
            "object_outside_body_numbers": sorted(outside),
        }
    return result


def _body_vertical_envelopes(
    facts: list[dict[str, Any]], text_left: float, text_right: float
) -> dict[int, tuple[float, float]]:
    page_bounds: dict[int, tuple[float, float]] = {}
    for fact in facts:
        page_number = int(fact.get("page_number") or 0)
        height = _number(fact.get("page_height_pt"))
        if page_number > 0 and height is not None and height > 0:
            # Match the page-band guard used by the venue column extractor.
            # It is stable even when a page starts with a figure or table.
            page_bounds[page_number] = (height * 0.06, height * 0.93)
    if page_bounds:
        return page_bounds
    by_page: dict[int, list[list[float]]] = {}
    for fact in facts:
        if fact.get("role") not in {"paragraph", "section_heading"}:
            continue
        bbox = _bbox(fact.get("bbox"))
        page_number = int(fact.get("page_number") or 0)
        if bbox is None or page_number <= 0 or bbox[2] < text_left or bbox[0] > text_right:
            continue
        by_page.setdefault(page_number, []).append(bbox)
    return {
        page: (min(box[1] for box in boxes), max(box[3] for box in boxes))
        for page, boxes in by_page.items()
    }


def _caption_region(
    bbox: list[float],
    text_left: float,
    text_right: float,
    left_column_end: float | None,
    right_column_start: float | None,
) -> tuple[float, float]:
    if (
        left_column_end is None
        or right_column_start is None
        or bbox[0] <= left_column_end + 3 and bbox[2] >= right_column_start - 3
    ):
        return text_left, text_right
    split = (left_column_end + right_column_start) / 2
    center = (bbox[0] + bbox[2]) / 2
    return (
        (text_left, left_column_end)
        if center < split
        else (right_column_start, text_right)
    )


def _caption_line_count(bbox: list[float], style: Any) -> int:
    style_data = style if isinstance(style, dict) else {}
    line_height = _number(style_data.get("baseline_gap_median_pt"))
    if line_height is None:
        font_size = _number(style_data.get("font_size_median_pt"))
        line_height = font_size * 1.2 if font_size is not None else None
    if line_height is None or line_height <= 0:
        return 1
    return max(1, round((bbox[3] - bbox[1]) / line_height))


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    return [float(item) for item in value]


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _compact_json(value: Any, limit: int = 1800) -> str:
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"
