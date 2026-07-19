"""Retrieval contracts shared with the RAG adapter owner."""

from typing import Any, Literal

from pydantic import Field, model_validator

from .base import StrictModel

SourceType = Literal["paper", "standard"]
ContentType = Literal[
    "text",
    "figure",
    "figure_caption",
    "table",
    "formula",
    "metadata",
    "reference",
]
RouteType = Literal["fact", "explain", "review", "score", "follow_up", "out_of_scope"]
EffectiveRouteType = Literal["fact", "explain", "review", "score", "out_of_scope"]


class EvidenceItem(StrictModel):
    evidence_id: str = Field(min_length=1)
    source_type: SourceType
    paper_id: str | None = None
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    content_type: ContentType
    quote: str = Field(min_length=1)
    section_title: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    source_uri: str = Field(min_length=1)
    retrieval_score: float = Field(ge=0)
    content_role: str | None = None
    object_id: str | None = None
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def paper_source_requires_paper_id(self) -> "EvidenceItem":
        if self.source_type == "paper" and not self.paper_id:
            raise ValueError("paper evidence requires paper_id")
        return self


class EvidenceSet(StrictModel):
    items: list[EvidenceItem] = Field(default_factory=list)
    query: str = Field(min_length=1)
    relaxed: bool = False
    warnings: list[str] = Field(default_factory=list)


class RetrieveEvidenceRequest(StrictModel):
    task_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    paper_ids: list[str] = Field(min_length=1)
    standalone_question: str = Field(min_length=1)
    route_type: EffectiveRouteType
    content_preferences: list[ContentType] = Field(default_factory=list)
    relaxed: bool = False


class RetrieveStandardsRequest(StrictModel):
    task_id: str = Field(min_length=1)
    standalone_question: str = Field(min_length=1)
    route_type: Literal["review", "score"]
    dimensions: list[str] = Field(min_length=1)
    standard_version: str | None = None
