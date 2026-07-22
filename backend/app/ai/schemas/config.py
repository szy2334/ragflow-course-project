"""Non-secret, immutable configuration snapshots."""

from typing import Literal

from pydantic import Field, HttpUrl

from .base import StrictModel


class ModelConfigSnapshot(StrictModel):
    config_version: str = Field(min_length=1)
    provider: str = Field(default="openai-compatible", min_length=1)
    base_url: HttpUrl
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(default=45.0, gt=0, le=300)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int = Field(default=2048, ge=64, le=65536)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    structured_mode: Literal["json_schema", "json_object", "prompt_json"] = "json_schema"
    enable_thinking: bool | None = None


class ConfigurationSnapshot(StrictModel):
    graph_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    standard_version: str | None = None
    model: ModelConfigSnapshot
