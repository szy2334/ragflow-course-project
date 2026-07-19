"""Dependency-injection ports owned by non-AI backend modules."""

from dataclasses import dataclass
from typing import Protocol

from .schemas import (
    EvidenceSet,
    NodeTrace,
    RetrieveEvidenceRequest,
    RetrieveStandardsRequest,
    StreamEvent,
)


class RetrievalPort(Protocol):
    async def retrieve_paper(self, request: RetrieveEvidenceRequest) -> EvidenceSet: ...

    async def retrieve_standards(self, request: RetrieveStandardsRequest) -> EvidenceSet: ...


class ContextPort(Protocol):
    async def load_session_summary(self, *, user_id: str, session_id: str) -> str: ...


class EventSink(Protocol):
    async def emit(self, event: StreamEvent) -> None: ...


class CancellationPort(Protocol):
    async def is_cancelled(self, task_id: str) -> bool: ...


class TraceSink(Protocol):
    async def record(self, trace: NodeTrace) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkflowDependencies:
    retrieval: RetrievalPort
    context: ContextPort
    events: EventSink
    cancellation: CancellationPort
    trace: TraceSink

