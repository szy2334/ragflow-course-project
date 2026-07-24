"""Structured contracts for the format-review graph only."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import Field

from app.ai.schemas.base import StrictModel

FindingResult = Literal["compliant", "non_compliant", "unverifiable", "not_applicable"]

RULE_UNIT_KINDS = frozenset(
    {"front_matter", "abstract", "body_section", "figure_table", "reference", "appendix", "global"}
)
RULE_EVIDENCE_SELECTORS = frozenset(
    {
        "page_geometry",
        "object_geometry",
        "font_style",
        "caption",
        "reference_entry",
        "author_identity",
        "text_content",
    }
)


class CandidateFinding(StrictModel):
    rule_ids: list[str] = Field(default_factory=list, max_length=20)
    category: str = Field(min_length=1, max_length=64)
    aspect: str = Field(min_length=1, max_length=500)
    result: FindingResult
    severity: Literal["info", "low", "medium", "high"] = "info"
    finding: str = Field(min_length=1, max_length=4000)
    suggestion: str | None = Field(default=None, max_length=4000)
    paper_evidence_ids: list[str] = Field(default_factory=list)
    standard_evidence_ids: list[str] = Field(default_factory=list)
    reason: str | None = Field(default=None, max_length=2000)


class CompositeFormatReviewOutput(StrictModel):
    summary_markdown: str = Field(min_length=1, max_length=16000)
    findings: list[CandidateFinding] = Field(default_factory=list, max_length=100)


class ReflectionOutput(StrictModel):
    decision: Literal[
        "confirmed",
        "recover_pdf_evidence",
        "retrieve_standard",
        "clarify_standard",
        "repair_check",
        "unverifiable",
    ]
    reason: str = Field(min_length=1, max_length=2000)


class FormatSynthesisOutput(StrictModel):
    """Summary only: final findings remain deterministic unit evidence records."""

    summary_markdown: str = Field(min_length=1, max_length=16000)


class FormatReviewState(TypedDict, total=False):
    task_id: str
    review_id: str
    task: dict[str, Any]
    review: dict[str, Any]
    snapshot: dict[str, Any]
    layout_facts: list[dict[str, Any]]
    layout_quality: dict[str, Any]
    retrieval_plan: dict[str, Any]
    standard_evidences: list[dict[str, Any]]
    coverage_report: dict[str, Any]
    review_units: list[dict[str, Any]]
    allocation_ledger: list[dict[str, Any]]
    unit_results: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    final_findings: list[dict[str, Any]]
    summary_markdown: str
    metrics: dict[str, Any]
    counters: dict[str, int]
    route: str
    sequence: int
    run_events: list[dict[str, Any]]
