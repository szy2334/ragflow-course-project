"""Serializable LangGraph state definition."""

from typing import Any, TypedDict


class ReviewGraphState(TypedDict, total=False):
    command: dict[str, Any]
    conversation_summary: str
    route_decision: dict[str, Any]
    paper_evidences: list[dict[str, Any]]
    standard_evidences: list[dict[str, Any]]
    paper_understanding: dict[str, Any] | None
    review_a: dict[str, Any] | None
    review_b: dict[str, Any] | None
    draft_answer: dict[str, Any] | None
    validation: dict[str, Any] | None
    final_answer: dict[str, Any] | None
    agent_results: list[dict[str, Any]]
    warnings: list[str]
    repair_count: int
    error_code: str | None
    error_message: str | None
    skip_reviews: bool
    sequence: int
