"""Provider-neutral structured model interface."""

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel

from ..schemas import ModelConfigSnapshot

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ModelCallMetrics:
    latency_ms: int
    input_tokens: int
    output_tokens: int
    retry_count: int
    model: str
    model_config_version: str


@dataclass(frozen=True, slots=True)
class StructuredModelResult(Generic[TModel]):
    output: TModel
    metrics: ModelCallMetrics


class StructuredLlm(Protocol):
    async def invoke_structured(
        self,
        messages: list[ChatMessage],
        output_model: type[TModel],
        config: ModelConfigSnapshot,
    ) -> StructuredModelResult[TModel]: ...
