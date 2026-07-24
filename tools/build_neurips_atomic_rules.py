"""Build one-rule-per-chunk NeurIPS 2020 rules from the supplied DOCX."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from extract_docx_blocks import extract

DATASET_ID = "e3c8be1a84a811f1bd3a97a1481915ff"
DOCUMENT_IDS = {
    "shared": "6bd5126e850711f1b211d112fde53137",
    "initial_submission": "6becebd2850711f1b211d112fde53137",
    "camera_ready": "6c0d37d4850711f1b211d112fde53137",
    "preprint": "6c294fe6850711f1b211d112fde53137",
}

SECTIONS: dict[int, dict[str, Any]] = {
    1: {"category": "page_layout", "units": ["global"], "selectors": ["page_geometry"]},
    2: {"category": "page_layout", "units": ["global"], "selectors": ["font_style", "page_geometry"]},
    3: {"category": "page_layout", "units": ["global"], "selectors": ["font_style", "text_content"]},
    4: {"category": "heading", "units": ["front_matter"], "selectors": ["font_style", "author_identity", "page_geometry"]},
    5: {"category": "abstract", "units": ["abstract"], "selectors": ["font_style", "page_geometry", "text_content"]},
    6: {"category": "heading", "units": ["body_section"], "selectors": ["font_style", "page_geometry", "text_content"]},
    7: {"category": "heading", "units": ["body_section"], "selectors": ["font_style", "text_content"]},
    8: {"category": "heading", "units": ["body_section"], "selectors": ["font_style", "text_content"]},
    9: {"category": "reference", "units": ["global"], "selectors": ["reference_entry", "text_content", "page_geometry"]},
    10: {"category": "figure_table", "units": ["figure_table"], "selectors": ["caption", "object_geometry", "text_content"]},
}

SKIP_TEXT = {"Times 兼容字体白名单"}
NON_RULE_DIRECTIVES = {
    "文档中不存在二级标题时，本规则输出不适用。",
    "文档中不存在三级标题时，本规则输出不适用。",
    "文档中既不存在图也不存在表时，本规则输出不适用。",
}

ATOMIC_SPLITS: dict[str, list[str]] = {
    "图号和表号分别从 1 开始连续编号": [
        "图号从 1 开始连续编号。",
        "表号从 1 开始连续编号。",
    ],
    "图号和表号的编号不得重复。": [
        "图号的编号不得重复。",
        "表号的编号不得重复。",
    ],
    "图题和表题相对所属栏或跨栏正文区域居中，中心偏差不超过 10 pt。": [
        "图题相对所属栏或跨栏正文区域居中，中心偏差不超过 10 pt。",
        "表题相对所属栏或跨栏正文区域居中，中心偏差不超过 10 pt。",
    ],
}

REPHRASES: dict[tuple[int, str], str] = {
    (2, "正文主字体必须为 Times New Roman 或属于以下 Times 兼容字体白名单。"): "正文主字体必须为 Times New Roman，或属于下列 Times 兼容字体白名单。",
    (4, "标题字号：16.5–17.5 pt。"): "论文标题字号：16.5–17.5 pt。",
    (4, "标题粗体字符比例至少为 0.9。"): "论文标题的粗体字符比例至少为 0.9。",
    (4, "标题中心与正文区域中心偏差不超过 8 pt。"): "论文标题中心与正文区域中心的偏差不超过 8 pt。",
    (4, "标题上方和下方必须各检测到至少一条水平线。"): "论文标题上方和下方必须各检测到至少一条水平线。",
    (5, "摘要正文相对正文区域左右各额外缩进：31–41 pt。"): "摘要正文相对正文区域左、右边界的额外缩进均为 31–41 pt。",
    (5, "摘要正文字号：9.5–10.5 pt。"): "摘要正文字号：9.5–10.5 pt。",
    (5, "摘要正文相邻行基线距离：10.25–11.75 pt。"): "摘要正文相邻行基线距离：10.25–11.75 pt。",
    (5, "Abstract 标题字号：11.5–12.5 pt。"): "摘要标题“Abstract”的字号：11.5–12.5 pt。",
    (5, "Abstract 标题粗体字符比例至少为 0.9。"): "摘要标题“Abstract”的粗体字符比例至少为 0.9。",
    (5, "Abstract 标题中心与正文区域中心偏差不超过 8 pt。"): "摘要标题“Abstract”中心与正文区域中心的偏差不超过 8 pt。",
    (6, "一级标题字号：11.5–12.5 pt。"): "一级章节标题字号：11.5–12.5 pt。",
    (6, "一级标题粗体字符比例至少为 0.8。"): "一级章节标题的粗体字符比例至少为 0.8。",
    (6, "一级标题左边界与正文区域左边界偏差不超过 3 pt。"): "一级章节标题左边界与正文区域左边界的偏差不超过 3 pt。",
    (7, "二级标题字号：9.5–10.5 pt。"): "二级章节标题字号：9.5–10.5 pt；如文档不存在二级章节标题，则本规则不适用。",
    (8, "三级标题字号：9.5–10.5 pt。"): "三级章节标题字号：9.5–10.5 pt；如文档不存在三级章节标题，则本规则不适用。",
    (9, "必须存在无编号的 References 一级标题。"): "必须存在无编号的 References 一级章节标题。",
    (9, "References 标题在文档阅读顺序上位于正文之后。"): "References 标题在文档阅读顺序上必须位于正文之后。",
    (9, "References 标题之后必须存在可识别的参考文献条目。"): "References 标题之后必须存在可识别的参考文献条目。",
    (10, "图题必须位于图形下方。"): "图题必须位于对应图形下方。",
    (10, "表题必须位于表格下方。"): "表题必须位于对应表格下方。",
}

WHITELIST_RULE = "正文主字体必须为 Times New Roman，或属于下列 Times 兼容字体白名单。"

CROSS_UNIT_RULES = {
    "摘要正文相对正文区域左、右边界的额外缩进均为 31–41 pt。": ["abstract", "body_section"],
    "摘要标题“Abstract”中心与正文区域中心的偏差不超过 8 pt。": ["abstract", "body_section"],
    "图题相对所属栏或跨栏正文区域居中，中心偏差不超过 10 pt。": ["figure_table", "body_section"],
    "表题相对所属栏或跨栏正文区域居中，中心偏差不超过 10 pt。": ["figure_table", "body_section"],
    "References 标题在文档阅读顺序上必须位于正文之后。": ["reference", "body_section"],
}

KEYWORDS: dict[str, list[str]] = {
    WHITELIST_RULE: ["body_text", "font_family", "font_whitelist", "times_compatible"],
    "论文标题中心与正文区域中心的偏差不超过 8 pt。": ["paper_title", "front_matter", "body_section", "cross_unit", "alignment"],
    "摘要正文相对正文区域左、右边界的额外缩进均为 31–41 pt。": ["abstract", "body_section", "cross_unit", "indent"],
    "摘要标题“Abstract”中心与正文区域中心的偏差不超过 8 pt。": ["abstract_heading", "body_section", "cross_unit", "alignment"],
    "一级章节标题左边界与正文区域左边界的偏差不超过 3 pt。": ["section_heading", "body_section", "cross_unit", "alignment"],
    "图题相对所属栏或跨栏正文区域居中，中心偏差不超过 10 pt。": ["figure", "figure_caption", "body_section", "cross_unit", "alignment"],
    "表题相对所属栏或跨栏正文区域居中，中心偏差不超过 10 pt。": ["table", "table_caption", "body_section", "cross_unit", "alignment"],
}


def _atomic_texts(text: str) -> list[str]:
    return ATOMIC_SPLITS.get(text, [text])


def _rephrase(section: int, text: str) -> str:
    return REPHRASES.get((section, text), text)


def _keywords(section: int, category: str, text: str, is_global: bool, requires_cross_unit: bool) -> list[str]:
    base = ["neurips", "2020", "format_rule", "shared", category]
    section_terms = {
        1: ["page_geometry", "body_layout", "margin"],
        2: ["body_text", "font", "line_spacing"],
        3: ["paragraph", "indent", "spacing"],
        4: ["paper_title", "front_matter", "heading"],
        5: ["abstract", "font", "spacing"],
        6: ["section_heading", "heading_level_1"],
        7: ["section_heading", "heading_level_2"],
        8: ["section_heading", "heading_level_3"],
        9: ["reference", "references", "document_structure"],
        10: ["figure", "table", "caption", "numbering"],
    }
    values = re.findall(r"\d+(?:\.\d+)?(?:–|-\d+(?:\.\d+)?)?\s*pt?", text, flags=re.I)
    terms = KEYWORDS.get(text, section_terms.get(section, []))
    if is_global:
        terms = [*terms, "global", "whole_document"]
    if requires_cross_unit:
        terms = [*terms, "cross_unit"]
    return list(dict.fromkeys([*base, *terms, *values]))


def _content(rule_id: str, section_title: str, category: str, text: str, attachment: str | None) -> str:
    lines = [
        f"规则ID：{rule_id}",
        "会议：Conference on Neural Information Processing Systems (NeurIPS)",
        "规范版本：2020.1",
        "投稿模式：shared",
        f"章节：{section_title}",
        f"规则类别：{category}",
        "规则原文：",
        text,
        "原子检查要求：",
        text,
    ]
    if attachment:
        lines.extend(["字体白名单：", attachment])
    return "\n".join(lines)


def _build_rule(section: int, section_title: str, text: str, attachment: str | None = None) -> dict[str, Any]:
    config = SECTIONS[section]
    category = str(config["category"])
    units = list(config["units"])
    selectors = list(config["selectors"])
    if section == 10:
        if text.startswith("图号") or text.startswith("图题"):
            category, units = "figure", ["figure_table"]
            selectors = ["caption", "object_geometry", "text_content"]
        elif text.startswith("表号") or text.startswith("表题"):
            category, units = "table", ["figure_table"]
            selectors = ["caption", "object_geometry", "text_content"]
    if section == 9 and text != "必须存在无编号的 References 一级章节标题。":
        units = ["reference"]
        selectors = ["reference_entry", "text_content", "page_geometry"]
    if text == WHITELIST_RULE:
        units = ["body_section"]
    is_global = (
        section in {1, 2, 3} and text != WHITELIST_RULE
    ) or text == "必须存在无编号的 References 一级章节标题。"
    cross_kinds = CROSS_UNIT_RULES.get(text, [])
    requires_cross_unit = bool(cross_kinds)
    digest = hashlib.sha256(f"{section}|shared|{text}".encode("utf-8")).hexdigest()[:16]
    rule_id = f"neurips-2020-shared-{section:02d}-{digest}"
    conditions: dict[str, list[str]] = {}
    if category in {"figure", "table"}:
        conditions["requires_object_types"] = [category]
    content = _content(rule_id, section_title, category, text, attachment)
    return {
        "content": content,
        "important_keywords": _keywords(section, category, text, is_global, requires_cross_unit),
        "questions": [f"论文是否满足单一要求：{text}"],
        "metadata": {
            "venue_id": "neurips",
            "format_version": "2020.1",
            "submission_mode": "shared",
            "target_document": DOCUMENT_IDS["shared"],
            "source_document_id": DOCUMENT_IDS["shared"],
            "canonical_rule_id": rule_id,
            "rule_category": category,
            "section_path": section_title,
            "effective_from": "2020-01-01",
            "status": "active",
            "source_text": text,
            "atomic_requirement": text,
            **({"source_attachment": attachment} if attachment else {}),
        },
        "manifest": {
            "rule_id": rule_id,
            "canonical_rule_id": rule_id,
            "title": f"{section_title}｜{text}",
            "description": content,
            "rule_text": text,
            "atomic_requirement": text,
            "keywords": ["neurips", "2020", "shared", category],
            "venue_id": "neurips",
            "format_version": "2020.1",
            "submission_mode": "shared",
            "target_document": DOCUMENT_IDS["shared"],
            "source_document_id": DOCUMENT_IDS["shared"],
            "section_path": section_title,
            "effective_from": "2020-01-01",
            "status": "active",
            "rule_category": category,
            "applicable_unit_kinds": units,
            "is_global": is_global,
            "requires_cross_unit": requires_cross_unit,
            "cross_unit_kinds": cross_kinds or (["global"] if is_global else []),
            "applicability_conditions": conditions,
            "evidence_selector": selectors,
            "assessment_mode": "strict",
            "supported_checks": [rule_id],
            "source_text": text,
            **({"source_attachment": attachment} if attachment else {}),
        },
    }


def build(source: Path) -> list[dict[str, Any]]:
    blocks = extract(source)
    whitelist = "\n".join(
        line
        for block in blocks if block["type"] == "table"
        for row in block.get("rows", [])
        for cell in row
        for line in str(cell).splitlines() if line
    )
    rules: list[dict[str, Any]] = []
    section = 0
    section_title = ""
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if block["type"] == "table":
            continue
        heading = re.fullmatch(r"(\d+)\.\s+(.+)", text)
        if heading and int(heading.group(1)) in SECTIONS:
            section = int(heading.group(1))
            section_title = text
            continue
        if not section or text in SKIP_TEXT or text in NON_RULE_DIRECTIVES:
            continue
        atomic_source = _rephrase(section, text)
        for atomic in _atomic_texts(atomic_source):
            attachment = whitelist if atomic == WHITELIST_RULE else None
            rules.append(_build_rule(section, section_title, atomic, attachment))
    return rules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules = build(args.source)
    ids = [r["metadata"]["canonical_rule_id"] for r in rules]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate canonical rule IDs")
    if len(rules) != 35:
        raise RuntimeError(f"Expected 35 atomic NeurIPS rules, got {len(rules)}")
    by_mode = {"shared": rules, "initial_submission": [], "camera_ready": [], "preprint": []}
    for mode, rows in by_mode.items():
        (args.output_dir / f"neurips_2020_{mode}.jsonl").write_text(
            "".join(json.dumps({k: v for k, v in row.items() if k != "manifest"}, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = [r["manifest"] for r in rules]
    (args.output_dir / "neurips_2020_rule_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    structure = {
        "schema_version": "neurips_atomic_rules_v1",
        "source": str(args.source),
        "dataset_id": DATASET_ID,
        "document_ids": DOCUMENT_IDS,
        "total_chunks": len(rules),
        "chunks_by_mode": {mode: len(rows) for mode, rows in by_mode.items()},
        "global_rule_count": sum(r["manifest"]["is_global"] for r in rules),
        "cross_unit_rule_count": sum(r["manifest"]["requires_cross_unit"] for r in rules),
        "font_whitelist_rule_count": sum(bool(r["metadata"].get("source_attachment")) for r in rules),
        "non_rule_directives": sorted(NON_RULE_DIRECTIVES),
    }
    (args.output_dir / "structure.json").write_text(json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(structure, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
