from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from app.ai.llm import ModelCallMetrics, StructuredModelResult
from app.ai.ports import WorkflowDependencies
from app.ai.schemas import (
    ConfigurationSnapshot,
    EvidenceItem,
    EvidenceSet,
    ModelConfigSnapshot,
    StartQaWorkflowCommand,
)


class ScriptedLlm:
    def __init__(self, outputs: list[BaseModel | Exception]) -> None:
        self.outputs = deque(outputs)
        self.calls: list[str] = []

    async def invoke_structured(self, messages, output_model, config):
        self.calls.append(output_model.__name__)
        output = self.outputs.popleft()
        if isinstance(output, Exception):
            raise output
        assert isinstance(output, output_model), (output, output_model)
        return StructuredModelResult(
            output=output,
            metrics=ModelCallMetrics(
                latency_ms=5,
                input_tokens=10,
                output_tokens=5,
                retry_count=0,
                model=config.model,
                model_config_version=config.config_version,
            ),
        )


@dataclass
class FakeRetrieval:
    paper_items: list[EvidenceItem]
    standard_items: list[EvidenceItem] = field(default_factory=list)
    paper_requests: list[Any] = field(default_factory=list)
    standard_requests: list[Any] = field(default_factory=list)

    async def retrieve_paper(self, request):
        self.paper_requests.append(request)
        return EvidenceSet(
            items=self.paper_items,
            query=request.standalone_question,
            relaxed=request.relaxed,
        )

    async def retrieve_standards(self, request):
        self.standard_requests.append(request)
        return EvidenceSet(items=self.standard_items, query=request.standalone_question)


@dataclass
class FakeContext:
    summary: str = ""

    async def load_session_summary(self, *, user_id: str, session_id: str) -> str:
        return self.summary


@dataclass
class FakeEvents:
    items: list[Any] = field(default_factory=list)

    async def emit(self, event):
        self.items.append(event)


@dataclass
class FakeCancellation:
    cancelled: bool = False

    async def is_cancelled(self, task_id: str) -> bool:
        return self.cancelled


@dataclass
class FakeTrace:
    items: list[Any] = field(default_factory=list)

    async def record(self, trace):
        self.items.append(trace)


@dataclass
class FakePersistence:
    events: FakeEvents
    error: Exception | None = None
    commands: list[Any] = field(default_factory=list)
    final_emitted_before_persist: bool = False

    async def persist(self, command):
        self.final_emitted_before_persist = any(
            event.event_type == "final" for event in self.events.items
        )
        if self.error:
            raise self.error
        self.commands.append(command)


@pytest.fixture
def model_snapshot() -> ModelConfigSnapshot:
    return ModelConfigSnapshot(
        config_version="model-v1",
        base_url="https://models.example.test",
        model="test-model",
        timeout_seconds=5.0,
        temperature=0.1,
        max_output_tokens=1024,
        structured_mode="json_schema",
    )


@pytest.fixture
def command(model_snapshot: ModelConfigSnapshot) -> StartQaWorkflowCommand:
    return StartQaWorkflowCommand(
        request_id="request-1",
        correlation_id="correlation-1",
        task_id="task-1",
        message_id="message-1",
        user_id="user-1",
        session_id="session-1",
        paper_ids=["paper-1"],
        original_question="论文使用了什么数据集？",
        configuration=ConfigurationSnapshot(
            graph_version="v1",
            prompt_version="v1",
            schema_version="v1",
            standard_version="standards-v1",
            model=model_snapshot,
        ),
    )


@pytest.fixture
def paper_evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="P1",
        source_type="paper",
        paper_id="paper-1",
        document_id="document-1",
        chunk_id="chunk-1",
        content_type="text",
        quote="论文在实验中使用了 CIFAR-10 数据集，共 60000 张图像。",
        section_title="Experiments",
        page_number=5,
        source_uri="paper://paper-1/chunk-1",
        retrieval_score=0.91,
    )


@pytest.fixture
def standard_evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="S1",
        source_type="standard",
        paper_id=None,
        document_id="standard-document-1",
        chunk_id="standard-chunk-1",
        content_type="text",
        quote="实验充分性要求报告数据集、基线、消融和统计信息。",
        section_title="实验充分性",
        page_number=1,
        source_uri="standard://review/experimental-sufficiency",
        retrieval_score=0.88,
    )


def dependencies(
    retrieval: FakeRetrieval,
    *,
    cancelled: bool = False,
    persistence_error: Exception | None = None,
):
    events = FakeEvents()
    traces = FakeTrace()
    persistence = FakePersistence(events, error=persistence_error)
    return (
        WorkflowDependencies(
            retrieval=retrieval,
            context=FakeContext(),
            events=events,
            cancellation=FakeCancellation(cancelled),
            trace=traces,
            persistence=persistence,
        ),
        events,
        traces,
    )
