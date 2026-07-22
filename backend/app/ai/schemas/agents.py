"""Strict inputs and outputs for the four agents."""

from typing import Literal

from pydantic import Field, model_validator

from .base import StrictModel
from .evidence import EffectiveRouteType, RouteType


class RouteDecision(StrictModel):
    initial_route_type: RouteType
    effective_route_type: EffectiveRouteType
    standalone_question: str = Field(min_length=1)
    review_dimensions: list[str] = Field(default_factory=list)
    needs_public_kb: bool
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def route_is_consistent(self) -> "RouteDecision":
        if self.initial_route_type != "follow_up" and (
            self.initial_route_type != self.effective_route_type
        ):
            raise ValueError("only follow_up may resolve to a different effective route")
        requires_public = self.effective_route_type in {"review", "score"}
        if self.needs_public_kb != requires_public:
            raise ValueError("needs_public_kb must match the effective route")
        if requires_public and not self.review_dimensions:
            raise ValueError("review and score routes require at least one dimension")
        return self


EvidenceStatus = Literal["explicit", "directly_inferred", "missing"]


class PaperFact(StrictModel):
    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    evidence_status: EvidenceStatus
    confidence: float = Field(ge=0, le=1)


class PaperUnderstanding(StrictModel):
    answerable: bool
    facts: list[PaperFact] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    paper_summary: str = ""

    @model_validator(mode="after")
    def answerable_has_facts(self) -> "PaperUnderstanding":
        if self.answerable and not self.facts:
            raise ValueError("answerable paper understanding requires facts")
        return self


ReviewPosition = Literal["critical", "supportive", "mixed"]
Severity = Literal["low", "medium", "high"]
SupportVerdict = Literal["supported", "partially_supported", "unsupported", "not_applicable"]


class ReviewClaim(StrictModel):
    statement: str = Field(min_length=1)
    severity: Severity
    paper_evidence_ids: list[str] = Field(min_length=1)
    standard_evidence_ids: list[str] = Field(min_length=1)
    reasoning_summary: str = Field(min_length=1, max_length=2000)
    review_a_verdict: SupportVerdict | None = None


class ReviewOpinion(StrictModel):
    dimension: str = Field(min_length=1)
    position: ReviewPosition
    claims: list[ReviewClaim] = Field(default_factory=list)
    suggested_score: float | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class ReviewOpinions(StrictModel):
    opinions: list[ReviewOpinion] = Field(default_factory=list)
