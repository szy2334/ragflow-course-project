"""Repair observable format-rule scopes and register the ICML 2026 profile.

This administrative migration is idempotent. It reads approved rule chunks from
the already configured RAGFlow datasets, then stores only their immutable IDs
and text in the server-controlled ``FormatProfile`` manifest.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.db.base import build_engine, build_session_factory
from app.db.models import FormatProfile
from app.format_review.workflow import _ragflow_endpoint
from tools.backfill_format_rule_scopes import scope_for_v2

ICML_DATASET_ID = "d6a59c2684a811f1bd3a97a1481915ff"
ICML_DOCUMENTS = {
    "shared": "e8cc32bc850611f1b211d112fde53137",
    "camera_ready": "e8a6511e850611f1b211d112fde53137",
    "initial_submission": "e889e812850611f1b211d112fde53137",
}
NEURIPS_REFERENCE_RULE_ID = "6e1de687fb74c69b"


def _section(text: str) -> str:
    match = re.search(r"章节：([^\r\n]+)", text)
    return match.group(1).strip() if match else "ICML 2026 格式规则"


def _mode(text: str, document_id: str) -> str:
    match = re.search(r"投稿模式：([a-z_]+)", text)
    if match:
        return match.group(1)
    return next((mode for mode, value in ICML_DOCUMENTS.items() if value == document_id), "shared")


def _keywords(text: str) -> list[str]:
    match = re.search(r"检索关键词：([^\r\n]+)", text)
    if not match:
        return []
    return [item.strip() for item in re.split(r"[、，,；;]\s*", match.group(1)) if item.strip()]


def _reference_scope() -> dict[str, Any]:
    return {
        "status": "active",
        "observability": "pdf_observable",
        "excluded_reason": None,
        "rule_category": "reference",
        "applicable_unit_kinds": ["reference"],
        "is_global": False,
        "requires_cross_unit": False,
        "cross_unit_kinds": [],
        # Reference headings are not consistently labelled by every PDF parser.
        # Allocate the rule to the reference/global unit and let missing facts
        # produce an explicit unverifiable result rather than silently skipping it.
        "applicability_conditions": {},
        "evidence_selector": ["reference_entry", "font_style", "text_content"],
        "scope_backfill_version": "v5",
    }


def _observable_scope(
    category: str,
    unit_kind: str,
    selectors: list[str],
    *,
    is_global: bool = False,
    assessment_mode: str = "strict",
    supported_checks: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": "active",
        "observability": "pdf_observable",
        "excluded_reason": None,
        "rule_category": category,
        "applicable_unit_kinds": [unit_kind],
        "is_global": is_global,
        "requires_cross_unit": is_global,
        "cross_unit_kinds": ["global"] if is_global else [],
        "applicability_conditions": {},
        "evidence_selector": selectors,
        # A source rule often bundles checks which are not all observable in a
        # rendered PDF. These fields bound the end-user conclusion to what the
        # native extractor can actually locate and measure.
        "assessment_mode": assessment_mode,
        "supported_checks": supported_checks or [],
    }


def _excluded_scope(reason: str) -> dict[str, Any]:
    """Keep source-only or unimplemented rules out of PDF-only review."""

    return {
        "status": "disabled",
        "observability": "excluded",
        "excluded_reason": reason,
        "rule_category": "body",
        "applicable_unit_kinds": ["body_section"],
        "is_global": False,
        "requires_cross_unit": False,
        "cross_unit_kinds": [],
        "applicability_conditions": {},
        "evidence_selector": ["text_content"],
        "assessment_mode": "excluded",
        "supported_checks": [],
    }


def _icml_scope(section: str, description: str) -> dict[str, Any]:
    section_name = section.lower()
    lowered = f"{section} {description}".lower()
    if "self-citations" in section_name:
        return _excluded_scope(
            "Detecting an author's own work and third-person attribution requires identity "
            "evidence outside the PDF."
        )
    if "citation" in section_name or "reference" in section_name:
        return {
            **_reference_scope(),
            "assessment_mode": "sampled",
            "supported_checks": ["reference_heading", "reference_style_samples"],
        }
    if "figures" in section_name:
        return _observable_scope(
            "figure",
            "figure_table",
            ["caption", "object_geometry", "font_style", "text_content"],
            assessment_mode="sampled",
            supported_checks=["caption_position", "caption_font", "object_caption_geometry"],
        )
    if "tables" in section_name:
        return _observable_scope(
            "table",
            "figure_table",
            ["caption", "object_geometry", "font_style", "text_content"],
            assessment_mode="sampled",
            supported_checks=["caption_position", "caption_font", "object_caption_geometry"],
        )
    if "abstract" in section_name:
        return _observable_scope(
            "abstract",
            "abstract",
            ["page_geometry", "font_style", "text_content"],
            supported_checks=["abstract_heading_style", "abstract_body_font_size"],
        )
    if "author information" in section_name:
        return _observable_scope(
            "author_identity",
            "front_matter",
            ["author_identity", "font_style", "text_content"],
            assessment_mode="sampled",
            supported_checks=["author_name_visibility", "author_name_style_samples"],
        )
    if "title" in section_name:
        return _observable_scope(
            "heading",
            "front_matter",
            ["page_geometry", "font_style", "text_content"],
            supported_checks=["title_font_size", "title_weight", "title_alignment"],
        )
    if "dimensions" in section_name:
        return _observable_scope(
            "page_layout",
            "global",
            ["page_geometry", "font_style", "text_content"],
            is_global=True,
            supported_checks=["page_dimensions", "text_font_size", "typeface_samples"],
        )
    if section_name == "2 paper format":
        return _excluded_scope(
            "The rule is a general instruction and has no atomic PDF-measurable requirement."
        )
    if "sections and subsections" in section_name or "partitioning the text" in section_name:
        if "partitioning" in section_name:
            return _excluded_scope(
                "Document organization is qualitative; the extractor has no reliable "
                "structure-completeness signal."
            )
        return _observable_scope(
            "heading",
            "body_section",
            ["font_style", "text_content"],
            supported_checks=[
                "heading_numbering",
                "heading_font",
                "heading_weight",
                "heading_alignment",
            ],
        )
    if "paragraphs and footnotes" in section_name:
        return _excluded_scope(
            "Indentation, blank lines, column footnotes, and line spacing are not reliable "
            "PDF facts in the current extractor."
        )
    if "algorithms" in section_name or "theorems and similar environments" in section_name:
        return _excluded_scope(
            "The extractor does not identify algorithm or theorem environments and their "
            "numbering semantics."
        )
    if "appendix formatting" in section_name:
        return _observable_scope(
            "page_layout",
            "appendix",
            ["page_geometry", "font_style", "text_content"],
            assessment_mode="sampled",
            supported_checks=["appendix_column_layout_samples", "appendix_font_style_samples"],
        )
    if "acknowledgements" in section_name:
        return _excluded_scope(
            "Optional acknowledgement content and its exact placement require document-wide "
            "semantic interpretation."
        )
    if "impact statement" in section_name:
        return _excluded_scope(
            "Impact-statement sufficiency is a semantic requirement, not a reliable layout-only "
            "check."
        )
    if (
        "electronic submission summary" in section_name
        and "main body" in lowered
        and "page" in lowered
    ):
        return _excluded_scope(
            "Main-body page limits require reliable body/reference boundaries and file metadata, "
            "which are not yet an atomic PDF fact."
        )
    scope = scope_for_v2({"title": section, "description": description})
    return {**scope, "scope_backfill_version": "v5"}


def _icml_rules(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_documents = set(ICML_DOCUMENTS.values())
    for chunk in chunks:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        document_id = str(chunk.get("document_id") or metadata.get("document_id") or "")
        rule_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
        description = str(chunk.get("content") or chunk.get("text") or "").strip()
        if (
            not rule_id
            or not description
            or document_id not in allowed_documents
            or rule_id in seen
        ):
            continue
        seen.add(rule_id)
        section = _section(description)
        mode = _mode(description, document_id)
        rules.append(
            {
                "rule_id": rule_id,
                "canonical_rule_id": rule_id,
                "title": f"ICML 2026 · {section}",
                "description": description,
                "keywords": ["icml", "2026", *(_keywords(description))],
                "venue_id": "icml",
                "format_version": "2026.1",
                "submission_mode": mode,
                "target_document": document_id,
                "source_document_id": document_id,
                "section_path": section,
                "effective_from": "2026-01-01",
                **_icml_scope(section, description),
            }
        )
    return rules


async def _retrieve_icml_chunks() -> list[dict[str, Any]]:
    settings = get_settings()
    if not settings.ragflow_base_url or not settings.ragflow_api_key:
        raise RuntimeError("RAGFlow must be configured before registering the ICML profile")
    payload = {
        "question": "ICML 2026 paper submission formatting rules",
        "dataset_ids": [ICML_DATASET_ID],
        "document_ids": list(ICML_DOCUMENTS.values()),
        "top_k": 100,
    }
    headers = {"Authorization": f"Bearer {settings.ragflow_api_key.get_secret_value()}"}
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        response = await client.post(
            _ragflow_endpoint(settings.ragflow_base_url, "retrieval"),
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        body = response.json()
    data = body.get("data", body) if isinstance(body, dict) else {}
    raw = data.get("chunks", data.get("items", [])) if isinstance(data, dict) else []
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _repair_existing_profile(profile: FormatProfile) -> bool:
    changed = False
    repaired_names = {
        "degree_thesis_2026": "学位论文格式（2026）",
        "neurips_2020": "NeurIPS 2020 投稿格式",
        "neurips_2026": "NeurIPS 2026 投稿格式",
    }
    expected_name = repaired_names.get(profile.profile_key)
    if expected_name and profile.name != expected_name:
        profile.name = expected_name
        changed = True
    if profile.profile_key != "neurips_2020":
        return changed
    repaired_rules: list[dict[str, Any]] = []
    for raw_rule in profile.rules_json or []:
        rule = dict(raw_rule)
        if str(rule.get("canonical_rule_id") or rule.get("rule_id")) == NEURIPS_REFERENCE_RULE_ID:
            rule.update(_reference_scope())
            rule["title"] = "参考文献格式"
            rule["section_path"] = "References formatting"
            changed = True
        repaired_rules.append(rule)
    if changed:
        profile.rules_json = repaired_rules
    return changed


async def reconcile(*, apply: bool) -> None:
    settings = get_settings()
    chunks = await _retrieve_icml_chunks()
    icml_rules = _icml_rules(chunks)
    if not icml_rules:
        raise RuntimeError("No ICML 2026 rule chunks were returned by the configured dataset")

    engine = build_engine(settings.database_url)
    sessions = build_session_factory(engine)
    try:
        async with sessions() as session:
            profiles = list((await session.scalars(select(FormatProfile))).all())
            repaired = [
                profile.profile_key for profile in profiles if _repair_existing_profile(profile)
            ]
            icml = next(
                (
                    profile
                    for profile in profiles
                    if profile.profile_key == "icml_2026" and profile.version == "2026.1"
                ),
                None,
            )
            if icml is None:
                icml = FormatProfile(
                    profile_key="icml_2026",
                    name="ICML 2026 投稿格式",
                    version="2026.1",
                    description="ICML 2026 投稿与终稿 PDF 格式审查。",
                    ragflow_dataset_id=ICML_DATASET_ID,
                    retrieval_query="ICML 2026 paper submission formatting rules",
                    venue_id="icml",
                    # The published ICML example paper is camera-ready; this
                    # order is the UI default while leaving both modes visible.
                    allowed_submission_modes=["camera_ready", "initial_submission"],
                    shared_document_id=ICML_DOCUMENTS["shared"],
                    mode_document_mapping_json={
                        "camera_ready": ICML_DOCUMENTS["camera_ready"],
                        "initial_submission": ICML_DOCUMENTS["initial_submission"],
                    },
                    rules_json=icml_rules,
                    is_active=True,
                )
                session.add(icml)
                icml_action = "created"
            else:
                icml.name = "ICML 2026 投稿格式"
                icml.description = "ICML 2026 投稿与终稿 PDF 格式审查。"
                icml.ragflow_dataset_id = ICML_DATASET_ID
                icml.retrieval_query = "ICML 2026 paper submission formatting rules"
                icml.venue_id = "icml"
                icml.allowed_submission_modes = ["camera_ready", "initial_submission"]
                icml.shared_document_id = ICML_DOCUMENTS["shared"]
                icml.mode_document_mapping_json = {
                    "camera_ready": ICML_DOCUMENTS["camera_ready"],
                    "initial_submission": ICML_DOCUMENTS["initial_submission"],
                }
                icml.rules_json = icml_rules
                icml.is_active = True
                icml_action = "updated"
            repaired_text = ", ".join(repaired) or "none"
            print(
                f"ICML rules discovered: {len(icml_rules)}; "
                f"{icml_action}; repaired: {repaired_text}"
            )
            if apply:
                await session.commit()
                print("Changes committed.")
            else:
                await session.rollback()
                print("Dry run only. Re-run with --apply to commit.")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Commit the reconciled profiles")
    args = parser.parse_args()
    asyncio.run(reconcile(apply=args.apply))


if __name__ == "__main__":
    main()
