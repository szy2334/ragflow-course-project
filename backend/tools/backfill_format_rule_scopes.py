"""Backfill deterministic V1.1 scope fields for existing format-rule manifests.

The script never changes rule prose, canonical IDs, RAGFlow document bindings, or
profile versions. It only fills missing structured scope fields using a visible,
repeatable rule-category/text mapping. Administrators can inspect with the
default dry run and apply a chosen profile explicitly.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

# Support direct ``python tools/backfill_format_rule_scopes.py`` execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.db.base import build_engine, build_session_factory
from app.db.models import FormatProfile


def scope_for(rule: dict[str, Any]) -> dict[str, Any]:
    return scope_for_v2(rule)


def scope_for_v2(rule: dict[str, Any]) -> dict[str, Any]:
    """Map imported prose to the small, PDF-observable V1.1 execution set.

    A format profile may preserve source-only and qualitative provisions for
    administrators, but only rules that the current PDF extractor can cite by
    page and layout evidence are active in the end-user review flow.
    """
    text = " ".join(
        [
            str(rule.get("title") or ""),
            str(rule.get("section_path") or ""),
            str(rule.get("rule_text") or rule.get("description") or ""),
        ]
    ).lower()

    category = "body"
    kind = "body_section"
    global_rule = False
    selectors = ["text_content"]
    conditions: dict[str, list[str]] = {}
    observable = False

    if "authors' names are bold" in text or "authors’ names are bold" in text:
        category, kind = "author_identity", "front_matter"
        selectors = ["author_identity", "font_style", "text_content"]
        observable = True
    elif "confine text within" in text or "general formatting instructions" in text:
        category, kind, global_rule = "page_layout", "global", True
        selectors = ["page_geometry", "font_style", "text_content"]
        observable = True
    elif "abstract paragraph" in text or "abstract formatting" in text:
        category, kind = "abstract", "abstract"
        selectors = ["page_geometry", "font_style", "text_content"]
        conditions = {"requires_section_roles": ["abstract"]}
        observable = True
    elif any(
        phrase in text
        for phrase in ("first-level headings", "second-level headings", "third-level headings")
    ):
        category, kind = "heading", "body_section"
        selectors = ["font_style", "text_content"]
        observable = True
    elif "figure number and caption" in text or "all artwork must" in text:
        category, kind = "figure", "figure_table"
        selectors = ["caption", "font_style", "text_content"]
        conditions = {"requires_object_types": ["figure"]}
        observable = True
    elif "table number and title" in text or "all tables must" in text:
        category, kind = "table", "figure_table"
        selectors = ["caption", "font_style", "text_content"]
        conditions = {"requires_object_types": ["table"]}
        observable = True
    elif "do not include acknowledgments" in text:
        category, kind, global_rule = "anonymity", "global", True
        selectors = ["author_identity", "text_content"]
        observable = True

    reason = None
    if not observable:
        if any(
            phrase in text
            for phrase in (
                "style file",
                "style files",
                "pdflatex",
                "usepackage",
                "final option",
                "preprint option",
                "electronic submission",
            )
        ):
            reason = "Requires source files, compiler options, or submission-system evidence; PDF-only review cannot verify it."
        else:
            reason = "Not part of the initial PDF-observable execution set; it needs qualitative or process evidence."

    return {
        "status": "active" if observable else "disabled",
        "observability": "pdf_observable" if observable else "excluded",
        "excluded_reason": reason,
        "rule_category": category,
        "applicable_unit_kinds": [kind],
        "is_global": global_rule,
        "requires_cross_unit": global_rule,
        "cross_unit_kinds": ["global"] if global_rule else [],
        "applicability_conditions": conditions,
        "evidence_selector": selectors,
    }


def _legacy_scope_for(rule: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        [
            str(rule.get("title") or ""),
            str(rule.get("section_path") or ""),
            str(rule.get("rule_category") or ""),
            str(rule.get("description") or ""),
        ]
    ).lower()
    if any(term in text for term in ("page limit", "margin", "paper size", "pdf files", "general formatting", "page_layout")):
        kind, global_rule, selectors, conditions, category = "global", True, ["page_geometry", "font_style", "text_content"], {}, "page_layout"
    elif any(term in text for term in ("abstract", "摘要")):
        kind, global_rule, selectors, conditions, category = "abstract", False, ["font_style", "text_content"], {"requires_section_roles": ["abstract"]}, "abstract"
    elif any(term in text for term in ("figure", "figures", "图题", "图片")):
        kind, global_rule, selectors, conditions, category = "figure_table", False, ["caption", "font_style", "text_content"], {"requires_object_types": ["figure"]}, "figure"
    elif any(term in text for term in ("table", "tables", "表格", "表题")):
        kind, global_rule, selectors, conditions, category = "figure_table", False, ["caption", "font_style", "text_content"], {"requires_object_types": ["table"]}, "table"
    elif any(term in text for term in ("reference", "citation", "bibliography", "footnote", "参考文献", "引用")):
        kind, global_rule, selectors, conditions, category = "reference", False, ["reference_entry", "text_content"], {}, "reference"
    elif any(term in text for term in ("author", "anonym", "acknowledg", "funding", "preprint", "camera-ready", "final version")):
        kind, global_rule, selectors, conditions, category = "front_matter", False, ["author_identity", "text_content"], {}, "anonymity"
    elif any(term in text for term in ("heading", "headings", "section", "标题")):
        kind, global_rule, selectors, conditions, category = "body_section", False, ["font_style", "text_content"], {}, "heading"
    elif any(term in text for term in ("style file", "template", "latex")):
        kind, global_rule, selectors, conditions, category = "global", True, ["text_content"], {}, "template"
    else:
        kind, global_rule, selectors, conditions, category = "body_section", False, ["font_style", "text_content"], {}, "body"
    return {
        "rule_category": category,
        "applicable_unit_kinds": [kind],
        "is_global": global_rule,
        "requires_cross_unit": global_rule,
        "cross_unit_kinds": ["global"] if global_rule else [],
        "applicability_conditions": conditions,
        "evidence_selector": selectors,
    }


async def run(profile_key: str | None, apply: bool) -> None:
    settings = get_settings()
    engine = build_engine(settings.database_url)
    sessions = build_session_factory(engine)
    try:
        async with sessions() as session:
            statement = select(FormatProfile).order_by(FormatProfile.profile_key, FormatProfile.version)
            if profile_key:
                statement = statement.where(FormatProfile.profile_key == profile_key)
            profiles = list((await session.scalars(statement)).all())
            for profile in profiles:
                updated = 0
                rules: list[dict[str, Any]] = []
                for item in profile.rules_json or []:
                    rule = dict(item)
                    missing = any(
                        field not in rule
                        for field in (
                            "applicable_unit_kinds",
                            "is_global",
                            "requires_cross_unit",
                            "cross_unit_kinds",
                            "applicability_conditions",
                            "evidence_selector",
                            "scope_backfill_version",
                        )
                    )
                    if str(rule.get("status") or "active") == "active" and (
                        missing or str(rule.get("scope_backfill_version") or "") != "v3"
                    ):
                        rule = {**rule, **scope_for(rule), "scope_backfill_version": "v3"}
                        updated += 1
                    rules.append(rule)
                print(f"{profile.profile_key}:{profile.version} rules={len(rules)} backfilled={updated}")
                if apply and updated:
                    profile.rules_json = rules
            if apply:
                await session.commit()
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-key", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.profile_key, args.apply))


if __name__ == "__main__":
    main()
