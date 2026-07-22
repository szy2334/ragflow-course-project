"""Strict public request bodies. Internal IDs are intentionally absent."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RegisterInput(ApiModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=256)
    display_name: str = Field(min_length=1, max_length=120)

    @field_validator("email")
    @classmethod
    def basic_email(cls, value: str) -> str:
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError("invalid email")
        return value.lower()


class LoginInput(ApiModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class SessionCreateInput(ApiModel):
    title: str | None = Field(default=None, max_length=300)
    paper_ids: list[str] = Field(min_length=1, max_length=10)


class SessionUpdateInput(ApiModel):
    title: str = Field(min_length=1, max_length=300)


class QuestionInput(ApiModel):
    question: str = Field(min_length=1, max_length=8000)
    paper_ids: list[str] | None = Field(default=None, max_length=10)
    stream: bool = True


class PaperRetryInput(ApiModel):
    stage: Literal[
        "mineru_parsing",
        "ocr_processing",
        "cleaning",
        "quality_check",
        "understanding",
    ]
    force: bool = False


class CancelInput(ApiModel):
    reason: str | None = Field(default=None, max_length=500)


class FeedbackInput(ApiModel):
    feedback_type: Literal["like", "dislike", "issue"]
    reason: str | None = Field(default=None, max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=10)


class AnalysisInput(ApiModel):
    question: str | None = Field(default=None, max_length=8000)
    force_refresh: bool = False


class ComparisonInput(ApiModel):
    paper_ids: list[str] = Field(min_length=2, max_length=10)
    dimensions: list[str] = Field(min_length=1, max_length=12)
    question: str | None = Field(default=None, max_length=8000)


class ReadingReportInput(ApiModel):
    paper_ids: list[str] = Field(min_length=1, max_length=10)
    session_id: str | None = None
    template_key: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=300)


class ExportInput(ApiModel):
    format: Literal["markdown", "pdf", "docx"]


class EvaluationInput(ApiModel):
    dataset_id: str = Field(min_length=1, max_length=128)
    split: str = Field(default="default", min_length=1, max_length=64)
    experiment_type: str = Field(min_length=1, max_length=128)
    model_config_id: str | None = Field(default=None, max_length=128)
    sample_limit: int | None = Field(default=None, ge=1, le=10_000)
    random_seed: int | None = None


class FormatReviewInput(ApiModel):
    paper_id: str = Field(min_length=1, max_length=36)
    format_profile_id: str = Field(min_length=1, max_length=36)
    submission_mode: str = Field(min_length=1, max_length=64)
    # Kept only as a transitional, server-validated request member. New clients
    # must omit it: the workflow checks the complete applicable rule manifest.
    rule_ids: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("rule_ids")
    @classmethod
    def unique_rule_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("rule_ids must be unique")
        return value


class FormatProfileUpsertInput(ApiModel):
    profile_key: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=300)
    version: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=4000)
    ragflow_dataset_id: str = Field(min_length=1, max_length=128)
    retrieval_query: str = Field(min_length=1, max_length=4000)
    venue_id: str | None = Field(default=None, max_length=128)
    allowed_submission_modes: list[str] = Field(min_length=1, max_length=12)
    shared_document_id: str = Field(min_length=1, max_length=128)
    mode_document_mapping: dict[str, str] = Field(min_length=1)
    rules: list[dict[str, object]] = Field(min_length=1, max_length=50)
    is_active: bool = True

    @field_validator("allowed_submission_modes")
    @classmethod
    def unique_submission_modes(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("allowed_submission_modes must be unique")
        return value


class ConfigUpdateInput(ApiModel):
    value: dict[str, object]
