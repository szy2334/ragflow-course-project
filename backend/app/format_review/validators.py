"""Deterministic evidence gates for format-review output."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def validate_findings(
    candidates: Iterable[dict[str, Any]],
    *,
    paper_evidences: list[dict[str, Any]],
    standard_evidences: list[dict[str, Any]],
    coverage_report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Enforce the evidence closure before results become durable findings."""

    paper_by_id = {str(item["evidence_id"]): item for item in paper_evidences}
    standard_by_id = {str(item["evidence_id"]): item for item in standard_evidences}
    missing_categories = set(coverage_report.get("missing_categories", []))
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for candidate in candidates:
        category = str(candidate.get("category") or "body")
        aspect = str(candidate.get("aspect") or category)
        key = (category, aspect)
        if key in seen:
            continue
        seen.add(key)
        result = str(candidate.get("result") or "unverifiable")
        paper_refs = [
            paper_by_id[item]
            for item in candidate.get("paper_evidence_ids", [])
            if item in paper_by_id
        ]
        standard_refs = [
            standard_by_id[item]
            for item in candidate.get("standard_evidence_ids", [])
            if item in standard_by_id
        ]
        paper_location_complete = bool(paper_refs) and all(
            isinstance(item.get("bbox"), list)
            and len(item["bbox"]) == 4
            and item.get("page_number") is not None
            for item in paper_refs
        )
        evidence_complete = bool(standard_refs) and paper_location_complete
        reason = str(candidate.get("reason") or "")

        if category in missing_categories:
            result = "unverifiable"
            reason = reason or "适用规范未被完整检索，无法可靠判断。"
        elif result in {"compliant", "non_compliant"} and not evidence_complete:
            result = "unverifiable"
            reason = reason or "结论缺少规范原文或带页码坐标的论文证据。"

        validated.append(
            {
                **candidate,
                "category": category,
                "aspect": aspect,
                "result": result,
                "paper_evidences": paper_refs,
                "standard_evidences": standard_refs,
                "evidence_status": "complete" if evidence_complete else "incomplete",
                "reason": reason or None,
            }
        )
    return validated


def forced_unverifiable_findings(coverage_report: dict[str, Any]) -> list[dict[str, Any]]:
    missing = coverage_report.get("missing_categories", [])
    return [
        {
            "category": category,
            "aspect": f"{category} 类格式要求",
            "result": "unverifiable",
            "severity": "info",
            "finding": "未能完整检索该类别的适用格式规范，系统未作合规推断。",
            "suggestion": None,
            "paper_evidences": [],
            "standard_evidences": [],
            "evidence_status": "incomplete",
            "reason": "适用规范检索覆盖不足。",
        }
        for category in missing
    ]
