"""Final answer and workflow result contracts."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from .agents import ReviewOpinion
from .base import StrictModel
from .evidence import EvidenceItem, RouteType

ClaimVerdict = Literal[
    "supported",
    "refuted",
    "insufficient_evidence",
    "conflicting_evidence",
]
TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]
AgentName = Literal[
    "controller",
    "intent_router",
    "paper_understanding",
    "answer_generator",
    "review_a",
    "review_b",
]


class Claim(StrictModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    type: Literal["positive", "negative"] = "positive"
    verdict: ClaimVerdict
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def positive_claim_has_evidence(self) -> "Claim":
        if self.type == "positive" and not self.evidence_ids:
            raise ValueError("positive claims require evidence")
        return self


class ScoreView(StrictModel):
    dimension: str = Field(min_length=1)
    value: float = Field(ge=0)
    scale: float = Field(gt=0)
    rubric_version: str = Field(min_length=1)
    standard_evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def value_within_scale(self) -> "ScoreView":
        if self.value > self.scale:
            raise ValueError("score value cannot exceed its scale")
        return self


class AnswerDraft(StrictModel):
    route_type: RouteType
    answer: str = Field(min_length=1, max_length=12000)
    claims: list[Claim] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    score: ScoreView | None = None
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    evidence_sufficient: bool = True
    evidence_gap_reason: str | None = None
    is_refusal: bool = False
    refusal_reason: str | None = None

    @model_validator(mode="after")
    def refusal_is_consistent(self) -> "AnswerDraft":
        if self.is_refusal and not self.refusal_reason:
            raise ValueError("refusal_reason is required when is_refusal is true")
        if not self.is_refusal and self.refusal_reason is not None:
            raise ValueError("refusal_reason must be null for a non-refusal answer")
        if self.score is not None and self.route_type != "score":
            raise ValueError("scores are only allowed for score routes")
        if not self.evidence_sufficient and not self.evidence_gap_reason:
            raise ValueError(
                "evidence_gap_reason is required when evidence_sufficient is false"
            )
        if self.evidence_sufficient and self.evidence_gap_reason is not None:
            raise ValueError(
                "evidence_gap_reason must be null when evidence_sufficient is true"
            )
        return self


class ReviewOpinionView(StrictModel):
    reviewer: Literal["review_a", "review_b"]
    position: Literal["critical", "supportive", "mixed"]
    summary: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)


class StandardReference(StrictModel):
    evidence_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class AnswerView(StrictModel):
    message_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    route_type: RouteType
    answer: str = Field(min_length=1, max_length=12000)
    claims: list[Claim] = Field(default_factory=list)
    evidences: list[EvidenceItem] = Field(default_factory=list)
    score: ScoreView | None = None
    review_opinions: list[ReviewOpinionView] = Field(default_factory=list)
    standards: list[StandardReference] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    evidence_sufficient: bool = True
    evidence_gap_reason: str | None = None
    is_refusal: bool
    refusal_reason: str | None = None
    completed_at: datetime


class AgentMetrics(StrictModel):
    latency_ms: int = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    model: str | None = None
    model_config_version: str | None = None
    retry_count: int = Field(default=0, ge=0)


class AgentResult(StrictModel):
    agent_name: AgentName
    status: TaskStatus
    summary: str
    claims: list[Claim] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    metrics: AgentMetrics


class WorkflowResult(StrictModel):
    answer: AnswerView
    agent_results: list[AgentResult] = Field(default_factory=list)
    workflow_summary: dict[str, Any]


class ValidationResult(StrictModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReviewSummary(StrictModel):
    review_a: list[ReviewOpinion] = Field(default_factory=list)
    review_b: list[ReviewOpinion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
