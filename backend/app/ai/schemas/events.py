"""Streaming and trace events emitted to infrastructure-owned sinks."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from .base import StrictModel

StreamEventType = Literal["status", "delta", "citation", "review_summary", "final", "error"]
TraceStatus = Literal["running", "succeeded", "retried", "failed", "skipped"]


class StreamEvent(StrictModel):
    event_id: str = Field(min_length=1)
    event_type: StreamEventType
    task_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)


class NodeTrace(StrictModel):
    request_id: str
    correlation_id: str
    task_id: str
    message_id: str
    node_name: str
    duration_ms: int = Field(ge=0)
    status: TraceStatus
    error_code: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
