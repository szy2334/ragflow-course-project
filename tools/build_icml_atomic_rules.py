"""Build one-rule-per-chunk ICML 2026 data from the revised DOCX."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from extract_docx_blocks import extract

DOCUMENT_IDS = {
    "shared": "e8cc32bc850611f1b211d112fde53137",
    "camera_ready": "e8a6511e850611f1b211d112fde53137",
    "initial_submission": "e889e812850611f1b211d112fde53137",
}

SECTION_CONFIG: dict[int, dict[str, Any]] = {
    1: {"mode": "shared", "category": "page_layout", "units": ["global"], "selectors": ["page_geometry", "font_style", "text_content"]},
    2: {"mode": "shared", "category": "heading", "units": ["front_matter"], "selectors": ["page_geometry", "font_style", "text_content"]},
    3: {"mode": "shared", "category": "heading", "units": ["body_section"], "selectors": ["font_style", "text_content"]},
    4: {"mode": "shared", "category": "figure", "units": ["figure_table"], "selectors": ["caption", "object_geometry", "font_style", "text_content"]},
    5: {"mode": "shared", "category": "table", "units": ["figure_table"], "selectors": ["caption", "object_geometry", "font_style", "text_content"]},
    6: {"mode": "shared", "category": "reference", "units": ["reference"], "selectors": ["reference_entry", "font_style", "text_content"]},
    7: {"mode": "shared", "category": "appendix", "units": ["appendix"], "selectors": ["page_geometry", "font_style", "text_content"]},
    8: {"mode": "initial_submission", "category": "author_identity", "units": ["front_matter"], "selectors": ["author_identity", "text_content", "page_geometry"]},
    9: {"mode": "initial_submission", "category": "page_layout", "units": ["global"], "selectors": ["page_geometry", "text_content"]},
    10: {"mode": "camera_ready", "category": "page_layout", "units": ["global"], "selectors": ["page_geometry"]},
    11: {"mode": "camera_ready", "category": "author_identity", "units": ["front_matter"], "selectors": ["author_identity", "font_style", "text_content", "page_geometry"]},
    12: {"mode": "camera_ready", "category": "abstract", "units": ["abstract"], "selectors": ["page_geometry", "font_style", "text_content"]},
    13: {"mode": "camera_ready", "category": "page_layout", "units": ["global"], "selectors": ["page_geometry", "text_content"]},
}

SKIP_TEXT = {
    "正文文本字体白名单",
    "文本字体白名单",
    "仅在初稿模式下执行。",
    "检查采用固定区域定位和确定性正则匹配。",
}

REMOVED_RULES = {
    "数学内容允许字体：CMR*、CMMI*、CMSY*、CambriaMath",
    "代码和等宽文本允许字体：CMTT*、NimbusMonL-Regu",
    "符号字符允许字体：Dingbats",
}

REMOVED_SECTION_RULE_COUNTS = {9: 3, 13: 3}
REMOVED_SECTIONS = set(REMOVED_SECTION_RULE_COUNTS)

LEGACY_SAME_CATEGORY_SPLITS: dict[str, list[str]] = {
    "正文、标题、章节标题、题注和参考文献字体必须属于下列正文文本字体白名单。": [
        "正文字体必须属于正文文本字体白名单。",
        "论文标题字体必须属于正文文本字体白名单。",
        "章节标题字体必须属于正文文本字体白名单。",
        "图题字体必须属于正文文本字体白名单。",
        "表题字体必须属于正文文本字体白名单。",
        "参考文献字体必须属于正文文本字体白名单。",
    ],
    "标题上下各有一条水平线。": [
        "标题上方必须有一条水平线。",
        "标题下方必须有一条水平线。",
    ],
    "一级字号：11.5–12.3 pt，粗体比例至少为 0.9。": [
        "一级标题字号必须为 11.5–12.3 pt。",
        "一级标题粗体字符比例必须至少为 0.9。",
    ],
    "二级字号：9.5–10.5 pt，粗体比例至少为 0.9。": [
        "二级标题字号必须为 9.5–10.5 pt。",
        "二级标题粗体字符比例必须至少为 0.9。",
    ],
    "图号从 1 开始连续编号。": [
        "图号必须从 1 开始编号。",
        "图号必须连续编号。",
    ],
    "表号从 1 开始连续编号。": [
        "表号必须从 1 开始编号。",
        "表号必须连续编号。",
    ],
    "References 必须为无编号一级标题。": [
        "References 必须为一级标题。",
        "References 标题不得编号。",
    ],
    "页面尺寸与正文一致。": [
        "附录页面宽度必须与正文页面宽度一致。",
        "附录页面高度必须与正文页面高度一致。",
    ],
    "页码存在状态和位置与正文一致。": [
        "附录页码的存在状态必须与正文一致。",
        "附录页码的位置必须与正文一致。",
    ],
    "双栏附录的栏起点与正文差异不超过 2 pt。": [
        "双栏附录的左栏起点与正文左栏起点差异不得超过 2 pt。",
        "双栏附录的右栏起点与正文右栏起点差异不得超过 2 pt。",
    ],
    "附录标题采用 A.、B. 等连续编号。": [
        "附录标题必须采用 A.、B. 等字母编号。",
        "附录标题必须连续编号。",
    ],
    "所有页面尺寸一致。": [
        "所有页面的宽度必须一致。",
        "所有页面的高度必须一致。",
    ],
    "单位编号为上标且不加粗。": [
        "单位编号必须为上标。",
        "单位编号不得加粗。",
    ],
    "单位编号从 1 开始连续出现。": [
        "单位编号必须从 1 开始。",
        "单位编号必须连续出现。",
    ],
    "单位信息集中在首页同一个脚注块中。": [
        "单位信息必须位于首页。",
        "单位信息必须集中在同一个脚注块中。",
    ],
}

# A compound rule is split only when its requirements belong to different
# rule categories. Requirements that remain in one category stay together.
ATOMIC_SPLITS: dict[str, list[str]] = {
    "正文、标题、章节标题、题注和参考文献字体必须属于下列正文文本字体白名单。": [
        "正文字体必须属于下列文本字体白名单。",
        "论文标题字体必须属于下列文本字体白名单。",
        "章节标题字体必须属于下列文本字体白名单。",
        "图题字体必须属于下列文本字体白名单。",
        "表题字体必须属于下列文本字体白名单。",
        "参考文献字体必须属于下列文本字体白名单。",
    ],
}

FONT_WHITELIST_RULES = {
    "正文字体必须属于下列文本字体白名单。",
    "论文标题字体必须属于下列文本字体白名单。",
    "章节标题字体必须属于下列文本字体白名单。",
    "图题字体必须属于下列文本字体白名单。",
    "表题字体必须属于下列文本字体白名单。",
    "参考文献字体必须属于下列文本字体白名单。",
    "附录正文字体必须属于下列文本字体白名单。",
}

RULE_OVERRIDES: dict[str, dict[str, Any]] = {
    "正文字体必须属于下列文本字体白名单。": {
        "category": "page_layout",
        "units": ["body_section"],
        "selectors": ["font_style", "text_content"],
        "attach_font_whitelist": True,
    },
    "论文标题字体必须属于下列文本字体白名单。": {
        "category": "heading",
        "units": ["front_matter"],
        "selectors": ["font_style", "text_content"],
        "attach_font_whitelist": True,
    },
    "章节标题字体必须属于下列文本字体白名单。": {
        "category": "heading",
        "units": ["body_section"],
        "selectors": ["font_style", "text_content"],
        "attach_font_whitelist": True,
    },
    "图题字体必须属于下列文本字体白名单。": {
        "category": "figure",
        "units": ["figure_table"],
        "selectors": ["caption", "font_style", "text_content"],
        "object_type": "figure",
        "attach_font_whitelist": True,
    },
    "表题字体必须属于下列文本字体白名单。": {
        "category": "table",
        "units": ["figure_table"],
        "selectors": ["caption", "font_style", "text_content"],
        "object_type": "table",
        "attach_font_whitelist": True,
    },
    "参考文献字体必须属于下列文本字体白名单。": {
        "category": "reference",
        "units": ["reference"],
        "selectors": ["reference_entry", "font_style", "text_content"],
        "attach_font_whitelist": True,
    },
    "附录正文字体必须属于下列文本字体白名单。": {
        "category": "appendix",
        "units": ["appendix"],
        "selectors": ["font_style", "text_content"],
        "attach_font_whitelist": True,
    },
    "正文中的文内引用主要采用作者—年份格式。": {
        "category": "reference",
        "units": ["body_section"],
        "selectors": ["text_content"],
    },
    "正文中的数字式文内引用不得占主要比例。": {
        "category": "reference",
        "units": ["body_section"],
        "selectors": ["text_content"],
    },
}

for _cross_unit_rule, _selectors in {
    "附录页面宽度和高度必须分别与正文页面宽度和高度一致。": ["page_geometry"],
    "附录正文字号众数与正文主体字号众数的差异不得超过 0.5 pt。": ["font_style", "text_content"],
    "附录正文基线间距中位数与正文主体基线间距中位数的差异不得超过 1 pt。": ["font_style", "text_content"],
    "双栏附录的左、右栏起点必须分别与正文双栏对应栏起点比对，差异均不得超过 2 pt。": ["page_geometry", "text_content"],
    "双栏附录的栏间距与正文双栏栏间距的差异不得超过 2 pt。": ["page_geometry", "text_content"],
    "附录页码的存在状态和位置必须分别与正文页码的存在状态和位置一致。": ["page_geometry", "text_content"],
}.items():
    RULE_OVERRIDES[_cross_unit_rule] = {
        "category": "appendix",
        "units": ["appendix"],
        "selectors": _selectors,
        "requires_cross_unit": True,
        "cross_unit_kinds": ["appendix", "body_section"],
    }

KEYWORD_OVERRIDES: dict[str, list[str]] = {
    "正文字体必须属于下列文本字体白名单。": [
        "body_text", "font_family", "font_whitelist"
    ],
    "论文标题字体必须属于下列文本字体白名单。": [
        "paper_title", "front_matter", "font_family", "font_whitelist"
    ],
    "章节标题字体必须属于下列文本字体白名单。": [
        "section_heading", "body_section", "font_family", "font_whitelist"
    ],
    "图题字体必须属于下列文本字体白名单。": [
        "figure", "figure_caption", "font_family", "font_whitelist"
    ],
    "表题字体必须属于下列文本字体白名单。": [
        "table", "table_caption", "font_family", "font_whitelist"
    ],
    "参考文献字体必须属于下列文本字体白名单。": [
        "reference", "reference_entry", "bibliography", "font_family", "font_whitelist"
    ],
    "附录正文字体必须属于下列文本字体白名单。": [
        "appendix", "appendix_body", "font_family", "font_whitelist"
    ],
    "正文中的文内引用主要采用作者—年份格式。": [
        "reference", "body_section", "in_text_citation", "author_year_citation"
    ],
    "正文中的数字式文内引用不得占主要比例。": [
        "reference", "body_section", "in_text_citation", "numeric_citation", "citation_ratio"
    ],
    "附录页面宽度和高度必须分别与正文页面宽度和高度一致。": [
        "appendix", "body_section", "cross_unit", "page_width", "page_height"
    ],
    "附录正文字号众数与正文主体字号众数的差异不得超过 0.5 pt。": [
        "appendix", "body_section", "cross_unit", "font_size_mode", "font_size_difference"
    ],
    "附录正文基线间距中位数与正文主体基线间距中位数的差异不得超过 1 pt。": [
        "appendix", "body_section", "cross_unit", "baseline_gap_median", "baseline_difference"
    ],
    "双栏附录的左、右栏起点必须分别与正文双栏对应栏起点比对，差异均不得超过 2 pt。": [
        "appendix", "body_section", "cross_unit", "left_column_start", "right_column_start"
    ],
    "双栏附录的栏间距与正文双栏栏间距的差异不得超过 2 pt。": [
        "appendix", "body_section", "cross_unit", "column_gap", "gutter"
    ],
    "附录页码的存在状态和位置必须分别与正文页码的存在状态和位置一致。": [
        "appendix", "body_section", "cross_unit", "page_number_presence", "page_number_position"
    ],
}

SECTION_TITLE_REPHRASES = {
    "2. 标题": "2. 论文标题",
}

# Context-qualified wording removes ambiguous uses of “标题/正文” without
# changing the measured object, threshold, comparison, or submission mode.
RULE_REPHRASES: dict[tuple[int, str], str] = {
    (1, "正文必须为双栏。"): "正文区域必须采用双栏布局。",
    (1, "总宽度：486 ± 2 pt。"): "正文双栏区域总宽度：486 ± 2 pt。",
    (1, "左栏起点：55.44 ± 2 pt。"): "正文左栏起点横坐标：55.44 ± 2 pt。",
    (1, "右栏起点：307.44 ± 2 pt。"): "正文右栏起点横坐标：307.44 ± 2 pt。",
    (1, "单栏宽度：234 ± 2 pt。"): "正文单栏宽度：234 ± 2 pt。",
    (1, "栏间距：18 ± 2 pt。"): "正文双栏的栏间距：18 ± 2 pt。",
    (1, "基线间距中位数：11–12.5 pt。"): "正文基线间距中位数：11–12.5 pt。",
    (1, "正文文本字体白名单"): "文本字体白名单",
    (2, "字号：13.8–14.8 pt。"): "论文标题字号：13.8–14.8 pt。",
    (2, "粗体字符比例至少为 0.9。"): "论文标题的粗体字符比例至少为 0.9。",
    (2, "标题中心与正文区域中心偏差不超过 8 pt。"): "论文标题中心与正文区域中心的偏差不超过 8 pt。",
    (2, "标题上下各有一条水平线。"): "论文标题上方和下方各有一条水平线。",
    (2, "横线宽度：0.8–1.2 pt。"): "论文标题上下水平线的线宽：0.8–1.2 pt。",
    (2, "上横线距页顶：70–75 pt。"): "论文标题上方水平线距页顶：70–75 pt。",
    (2, "内容词采用标题式大小写。"): "论文标题中的内容词采用标题式大小写。",
    (3, "一级标题格式：1. Title。"): "一级章节标题格式：1. Title。",
    (3, "二级标题格式：1.1. Title。"): "二级章节标题格式：1.1. Title。",
    (3, "三级标题格式：1.1.1. Title。"): "三级章节标题格式：1.1.1. Title。",
    (3, "一级字号：11.5–12.3 pt，粗体比例至少为 0.9。"): "一级章节标题字号：11.5–12.3 pt，粗体字符比例至少为 0.9。",
    (3, "二级字号：9.5–10.5 pt，粗体比例至少为 0.9。"): "二级章节标题字号：9.5–10.5 pt，粗体字符比例至少为 0.9。",
    (3, "三级字号：9.5–10.5 pt。"): "三级章节标题字号：9.5–10.5 pt。",
    (3, "标题左边界与所在栏起点偏差不超过 3 pt。"): "章节标题左边界与所在栏起点的偏差不超过 3 pt。",
    (3, "标题不得与相邻内容重叠。"): "章节标题不得与相邻内容重叠。",
    (3, "标题最多三级。"): "章节标题层级最多为三级。",
    (3, "内容词采用标题式大小写。"): "章节标题中的内容词采用标题式大小写。",
    (6, "正文引用主要采用作者—年份格式。"): "正文中的文内引用主要采用作者—年份格式。",
    (6, "数字引用格式不得占主要比例。"): "正文中的数字式文内引用不得占主要比例。",
    (6, "References 必须为无编号一级标题。"): "References 必须采用无编号的一级章节标题格式。",
    (6, "悬挂缩进：8–12 pt。"): "参考文献条目的悬挂缩进：8–12 pt。",
    (7, "页面尺寸与正文一致。"): "附录页面宽度和高度必须分别与正文页面宽度和高度一致。",
    (7, "正文使用规则 1 的字体白名单。"): "附录正文字体必须属于下列文本字体白名单。",
    (7, "字号众数与正文差异不超过 0.5 pt。"): "附录正文字号众数与正文主体字号众数的差异不得超过 0.5 pt。",
    (7, "基线间距中位数与正文差异不超过 1 pt。"): "附录正文基线间距中位数与正文主体基线间距中位数的差异不得超过 1 pt。",
    (7, "双栏附录的栏起点与正文差异不超过 2 pt。"): "双栏附录的左、右栏起点必须分别与正文双栏对应栏起点比对，差异均不得超过 2 pt。",
    (7, "双栏附录的栏间距与正文差异不超过 2 pt。"): "双栏附录的栏间距与正文双栏栏间距的差异不得超过 2 pt。",
    (7, "页码存在状态和位置与正文一致。"): "附录页码的存在状态和位置必须分别与正文页码的存在状态和位置一致。",
    (8, "标题下横线与 Abstract 标题之间不得存在身份信息块。"): "论文标题下方水平线与摘要标题“Abstract”之间不得存在身份信息块。",
    (8, "该区域只允许空白或 Anonymous Authors、Anonymous Submission、Under Review。"): "论文标题下方水平线与摘要标题“Abstract”之间的区域只允许为空，或包含 Anonymous Authors、Anonymous Submission、Under Review。",
    (11, "作者信息位于标题下横线之后。"): "作者信息位于论文标题下方水平线之后。",
    (11, "作者姓名或上标距下横线：19–24 pt。"): "作者姓名文本或其上标距论文标题下方水平线：19–24 pt。",
    (11, "作者行中心与正文区域中心偏差不超过 8 pt。"): "作者姓名行中心与正文区域中心的偏差不超过 8 pt。",
    (12, "Abstract 标题居中。"): "摘要标题“Abstract”居中。",
    (12, "标题字号：11.5–12.3 pt。"): "摘要标题“Abstract”的字号：11.5–12.3 pt。",
    (12, "标题粗体比例至少为 0.9。"): "摘要标题“Abstract”的粗体字符比例至少为 0.9。",
    (12, "正文字号：9.5–10.5 pt。"): "摘要正文字号：9.5–10.5 pt。",
    (12, "基线间距中位数：11–12.5 pt。"): "摘要正文的基线间距中位数：11–12.5 pt。",
    (12, "左侧额外缩进：15–21 pt。"): "摘要正文左侧额外缩进：15–21 pt。",
    (12, "右侧额外缩进：15–21 pt。"): "摘要正文右侧额外缩进：15–21 pt。",
    (12, "摘要后空白：26–32 pt。"): "摘要正文结束后空白：26–32 pt。",
}


def _atomic_texts(text: str) -> list[str]:
    if text in REMOVED_RULES:
        return []
    return ATOMIC_SPLITS.get(text, [text])


def _controlled_keywords(*, section: int, category: str, mode: str, atomic_requirement: str) -> list[str]:
    """Return short lexical anchors instead of embedding whole prose sentences."""

    base = ["icml", "2026", "format_rule", mode, category]
    section_terms = {
        1: ["page_layout", "body", "column", "font", "margin", "spacing"],
        2: ["title", "front_matter", "font", "alignment"],
        3: ["section_heading", "heading", "font", "numbering"],
        4: ["figure", "caption", "object_geometry"],
        5: ["table", "caption", "object_geometry"],
        6: ["reference", "citation", "bibliography"],
        7: ["appendix", "page_layout", "font", "spacing"],
        8: ["initial_submission", "anonymity", "author_identity"],
        9: ["initial_submission", "page_limit", "main_body_page_count"],
        10: ["camera_ready", "page_geometry", "letter"],
        11: ["camera_ready", "author_identity", "front_matter"],
        12: ["camera_ready", "abstract", "font", "spacing"],
        13: ["camera_ready", "page_limit", "main_body_page_count"],
    }
    category_terms = {
        "page_layout": ["page_layout", "page_geometry", "body_layout"],
        "heading": ["heading", "title", "section_heading"],
        "figure": ["figure", "figure_caption"],
        "table": ["table", "table_caption"],
        "reference": ["reference", "bibliography"],
        "appendix": ["appendix"],
        "author_identity": ["author_identity", "front_matter"],
        "abstract": ["abstract"],
    }
    property_terms = [
        ("字号", "font_size"),
        ("字体", "font_family"),
        ("粗体", "boldness"),
        ("横线", "horizontal_rule"),
        ("页", "page"),
        ("栏", "column"),
        ("边界", "bbox"),
        ("缩进", "indent"),
        ("间距", "spacing"),
        ("编号", "numbering"),
        ("引用", "citation"),
        ("作者", "author_identity"),
        ("摘要", "abstract"),
        ("表题", "table_caption"),
        ("图题", "figure_caption"),
    ]
    values = re.findall(r"\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?\s*(?:pt|MB)?", atomic_requirement, flags=re.I)
    semantic = [token for needle, token in property_terms if needle in atomic_requirement]
    keyword_override = KEYWORD_OVERRIDES.get(atomic_requirement)
    structural = keyword_override or category_terms.get(category, [])
    if keyword_override is None and category == SECTION_CONFIG[section]["category"]:
        structural = [*structural, *section_terms.get(section, [])]
    return list(dict.fromkeys([*base, *structural, *semantic, *values]))


def _content(rule_id: str, section_title: str, mode: str, category: str, source_text: str, atomic_requirement: str, source_attachment: str | None) -> str:
    return "\n".join(
        [
            f"规则ID：{rule_id}",
            "会议：International Conference on Machine Learning (ICML)",
            "规范版本：2026.1",
            f"投稿模式：{mode}",
            f"章节：{section_title}",
            f"规则类别：{category}",
            "规则原文：",
            source_text,
            "原子检查要求：",
            atomic_requirement,
        ]
        + (["字体白名单：", source_attachment] if source_attachment else [])
    )


def _build_rule(*, section: int, section_title: str, source_text: str, atomic_requirement: str, source_attachment: str | None = None) -> dict[str, Any]:
    section_config = SECTION_CONFIG[section]
    override = RULE_OVERRIDES.get(atomic_requirement, {})
    mode = str(section_config["mode"])
    category = str(override.get("category") or section_config["category"])
    units = list(override.get("units") or section_config["units"])
    selectors = list(override.get("selectors") or section_config["selectors"])
    digest = hashlib.sha256(f"{section}|{mode}|{atomic_requirement}".encode("utf-8")).hexdigest()[:16]
    rule_id = f"icml-2026-{mode}-{section:02d}-{digest}"
    is_global = units == ["global"]
    requires_cross_unit = bool(override.get("requires_cross_unit", is_global))
    cross_unit_kinds = list(
        override.get("cross_unit_kinds") or (["global"] if is_global else [])
    )
    conditions: dict[str, list[str]] = {}
    if mode != "shared":
        conditions["requires_submission_mode"] = [mode]
    object_type = str(override.get("object_type") or "")
    if object_type:
        conditions["requires_object_types"] = [object_type]
    elif section == 4:
        conditions["requires_object_types"] = ["figure"]
    elif section == 5:
        conditions["requires_object_types"] = ["table"]
    content = _content(rule_id, section_title, mode, category, source_text, atomic_requirement, source_attachment)
    return {
        "content": content,
        "important_keywords": _controlled_keywords(
            section=section,
            category=category,
            mode=mode,
            atomic_requirement=atomic_requirement,
        ),
        "questions": [f"论文是否满足单一要求：{atomic_requirement}"],
        "metadata": {
            "venue_id": "icml",
            "format_version": "2026.1",
            "submission_mode": mode,
            "target_document": DOCUMENT_IDS[mode],
            "source_document_id": DOCUMENT_IDS[mode],
            "canonical_rule_id": rule_id,
            "rule_category": category,
            "section_path": section_title,
            "effective_from": "2026-01-01",
            "status": "active",
            "source_text": source_text,
            "atomic_requirement": atomic_requirement,
            **({"source_attachment": source_attachment} if source_attachment else {}),
        },
        "manifest": {
            "rule_id": rule_id,
            "canonical_rule_id": rule_id,
            "title": f"{section_title}｜{atomic_requirement}",
            "description": content,
            "rule_text": source_text,
            "atomic_requirement": atomic_requirement,
            "keywords": list(dict.fromkeys(["icml", "2026", mode, category])),
            "venue_id": "icml",
            "format_version": "2026.1",
            "submission_mode": mode,
            "target_document": DOCUMENT_IDS[mode],
            "source_document_id": DOCUMENT_IDS[mode],
            "section_path": section_title,
            "effective_from": "2026-01-01",
            "status": "active",
            "rule_category": category,
            "applicable_unit_kinds": units,
            "is_global": is_global,
            "requires_cross_unit": requires_cross_unit,
            "cross_unit_kinds": cross_unit_kinds,
            "applicability_conditions": conditions,
            "evidence_selector": selectors,
            "assessment_mode": "strict",
            "supported_checks": [rule_id],
            "source_text": source_text,
            **({"source_attachment": source_attachment} if source_attachment else {}),
        },
    }


def build(source: Path) -> list[dict[str, Any]]:
    blocks = extract(source)
    rules: list[dict[str, Any]] = []
    section = 0
    section_title = ""
    font_whitelist = [
        line
        for block in blocks
        if block["type"] == "table"
        for row in block.get("rows", [])
        for cell in row
        for line in str(cell).splitlines()
        if line
    ]
    for index, block in enumerate(blocks):
        text = str(block.get("text") or "").strip()
        if block["type"] == "table":
            continue
        heading = re.fullmatch(r"(\d+)\.\s+(.+)", text)
        if heading and int(heading.group(1)) in SECTION_CONFIG:
            section = int(heading.group(1))
            section_title = text
            continue
        if not section or section in REMOVED_SECTIONS or text in SKIP_TEXT or text in REMOVED_RULES or index < 3:
            continue
        for atomic in _atomic_texts(text):
            attachment = None
            if atomic in FONT_WHITELIST_RULES:
                attachment = "\n".join(font_whitelist)
            rules.append(
                _build_rule(
                    section=section,
                    section_title=section_title,
                    source_text=atomic,
                    atomic_requirement=atomic,
                    source_attachment=attachment,
                )
            )
    return rules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules = build(args.source)
    ids = [item["metadata"]["canonical_rule_id"] for item in rules]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate canonical rule IDs")
    source_blocks = extract(args.source)
    expected_source_texts: list[str] = []
    active_section = False
    active_removed_section = False
    for block in source_blocks:
        text = str(block.get("text") or "").strip()
        if block["type"] == "table":
            continue
        heading = re.fullmatch(r"(\d+)\.\s+(.+)", text)
        if heading and int(heading.group(1)) in SECTION_CONFIG:
            active_section = True
            active_removed_section = int(heading.group(1)) in REMOVED_SECTIONS
            continue
        if active_section and not active_removed_section and text and text not in SKIP_TEXT and text not in REMOVED_RULES:
            expected_source_texts.append(text)
    expected_atomic_pairs = [
        (atomic, atomic)
        for source_text in expected_source_texts
        for atomic in _atomic_texts(source_text)
    ]
    actual_atomic_pairs = [
        (str(item["metadata"]["source_text"]), str(item["metadata"]["atomic_requirement"]))
        for item in rules
    ]
    if actual_atomic_pairs != expected_atomic_pairs:
        raise RuntimeError("Generated rules do not preserve the revised DOCX bullets exactly")
    whitelist_rules = [item for item in rules if item["metadata"].get("source_attachment")]
    if len(whitelist_rules) != len(FONT_WHITELIST_RULES):
        raise RuntimeError("The DOCX font whitelist table was not attached to every font rule")
    by_mode = {mode: [item for item in rules if item["metadata"]["submission_mode"] == mode] for mode in DOCUMENT_IDS}
    for mode, rows in by_mode.items():
        path = args.output_dir / f"icml_2026_{mode}.jsonl"
        path.write_text("".join(json.dumps({k: v for k, v in row.items() if k != "manifest"}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = [item["manifest"] for item in rules]
    (args.output_dir / "icml_2026_rule_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    structure = {
        "schema_version": "icml_atomic_rules_v1",
        "source": str(args.source),
        "dataset_id": "d6a59c2684a811f1bd3a97a1481915ff",
        "document_ids": DOCUMENT_IDS,
        "total_chunks": len(rules),
        "input_source_rule_count": len(expected_source_texts),
        "configured_removed_rule_count": len(REMOVED_RULES),
        "removed_section_count": len(REMOVED_SECTIONS),
        "removed_section_rule_count": sum(REMOVED_SECTION_RULE_COUNTS.values()),
        "new_source_rule_count": len(rules),
        "compound_source_rule_count": len(ATOMIC_SPLITS),
        "atomic_chunk_count": len(rules),
        "font_whitelist_rule_count": len(whitelist_rules),
        "necessary_cross_unit_rule_count": sum(
            1
            for item in rules
            if item["manifest"]["requires_cross_unit"] and not item["manifest"]["is_global"]
        ),
        "chunks_by_mode": {mode: len(rows) for mode, rows in by_mode.items()},
        "chunks_by_section": {str(section): sum(1 for row in rules if row["metadata"]["section_path"].startswith(f"{section}.")) for section in SECTION_CONFIG},
        "system_added_format_rules": 0,
        "non_rule_directives": {
            "initial_submission_scope": "仅在初稿模式下执行。",
            "initial_submission_check_method": "检查采用固定区域定位和确定性正则匹配。",
        },
    }
    (args.output_dir / "structure.json").write_text(json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(structure, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
