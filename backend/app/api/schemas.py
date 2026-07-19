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
    knowledge_base_id: str | None = Field(default=None, max_length=128)


class SessionUpdateInput(ApiModel):
    title: str = Field(min_length=1, max_length=300)


class QuestionInput(ApiModel):
    question: str = Field(min_length=1, max_length=8000)
    paper_ids: list[str] | None = Field(default=None, max_length=10)
    knowledge_base_id: str | None = Field(default=None, max_length=128)
    stream: bool = True


class PaperRetryInput(ApiModel):
    stage: Literal["mineru_parsing", "ocr_processing", "cleaning", "quality_check", "indexing"]
    force: bool = False


class CancelInput(ApiModel):
    reason: str | None = Field(default=None, max_length=500)


class FeedbackInput(ApiModel):
    feedback_type: Literal["like", "dislike", "issue"]
    reason: str | None = Field(default=None, max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=10)
