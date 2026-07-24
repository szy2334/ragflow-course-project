import pytest
from conftest import FakeRetrieval, ScriptedLlm, dependencies
from langgraph.checkpoint.memory import InMemorySaver

from app.ai import AiWorkflowService
from app.ai.agents.answer_generator import _prompt_evidence
from app.ai.errors import AnswerPersistenceFailed, ModelTransportError, WorkflowCancelled
from app.ai.schemas import AnswerDraft, Claim, PaperFact, PaperUnderstanding, RouteDecision


def fact_route() -> RouteDecision:
    return RouteDecision(
        initial_route_type="fact",
        effective_route_type="fact",
        standalone_question="论文使用了什么数据集？",
        review_dimensions=[],
        needs_public_kb=False,
        confidence=0.95,
    )


def evaluation_route() -> RouteDecision:
    """A model may emit the legacy route; reading must normalize it safely."""

    return RouteDecision(
        initial_route_type="review",
        effective_route_type="review",
        standalone_question="论文的实验设计是否充分？",
        review_dimensions=["实验充分性"],
        needs_public_kb=True,
        confidence=0.94,
    )


def out_of_scope_route() -> RouteDecision:
    return RouteDecision(
        initial_route_type="out_of_scope",
        effective_route_type="out_of_scope",
        standalone_question="今天天气如何？",
        review_dimensions=[],
        needs_public_kb=False,
        confidence=0.99,
    )


def general_chat_route() -> RouteDecision:
    return RouteDecision(
        initial_route_type="general_chat",
        effective_route_type="general_chat",
        standalone_question="你好，请介绍一下你自己。",
        review_dimensions=[],
        needs_public_kb=False,
        confidence=0.99,
    )


def paper_understanding() -> PaperUnderstanding:
    return PaperUnderstanding(
        answerable=True,
        facts=[
            PaperFact(
                claim="论文使用 CIFAR-10，共 60000 张图像",
                evidence_ids=["P1"],
                evidence_status="explicit",
                confidence=0.93,
            )
        ],
        missing_information=[],
        paper_summary="论文给出了实验数据集。",
    )


def fact_draft() -> AnswerDraft:
    return AnswerDraft(
        route_type="fact",
        answer="论文使用了 CIFAR-10 数据集，共 60000 张图像。",
        claims=[
            Claim(
                claim_id="C1",
                text="数据集包含 60000 张图像",
                verdict="supported",
                confidence=0.92,
                evidence_ids=["P1"],
                reason="论文原文明确说明",
            )
        ],
        evidence_ids=["P1"],
        confidence=0.92,
    )


def invalid_fact_draft() -> AnswerDraft:
    return AnswerDraft(
        route_type="fact",
        answer="论文使用了 70000 张图像。",
        claims=[
            Claim(
                claim_id="C1",
                text="数据集包含 70000 张图像",
                verdict="supported",
                confidence=0.9,
                evidence_ids=["P9"],
                reason="无效引用",
            )
        ],
        evidence_ids=["P9"],
        confidence=0.9,
    )


def insufficient_fact_draft() -> AnswerDraft:
    return AnswerDraft(
        route_type="fact",
        answer="现有片段只说明论文使用 CIFAR-10，未覆盖所询问的训练硬件。",
        claims=[
            Claim(
                claim_id="C1",
                text="论文使用 CIFAR-10 数据集。",
                verdict="supported",
                confidence=0.9,
                evidence_ids=["P1"],
                reason="论文原文明确说明。",
            ),
            Claim(
                claim_id="C2",
                text="当前检索证据未覆盖训练硬件。",
                type="negative",
                verdict="insufficient_evidence",
                confidence=0.4,
                evidence_ids=[],
                reason="检索片段中没有相关信息。",
            ),
        ],
        evidence_ids=["P1"],
        confidence=0.4,
        evidence_sufficient=False,
        evidence_gap_reason="当前检索片段未包含训练硬件信息。",
    )


def general_chat_draft() -> AnswerDraft:
    return AnswerDraft(
        route_type="general_chat",
        answer="你好，我可以回答通用问题，也可以在需要时查阅你选择的论文。",
        claims=[],
        evidence_ids=[],
        confidence=0.95,
    )


@pytest.mark.asyncio
async def test_reading_path_uses_only_local_paper_chunks(command, paper_evidence):
    llm = ScriptedLlm([fact_route(), fact_draft()])
    retrieval = FakeRetrieval([paper_evidence])
    deps, events, _ = dependencies(retrieval)
    saver = InMemorySaver()

    result = await AiWorkflowService(llm).run(command, deps, saver)

    assert result.answer.answer.startswith("论文使用了")
    assert retrieval.standard_requests == []
    assert llm.calls == ["RouteDecision", "AnswerDraft"]
    assert result.answer.review_opinions == []
    assert result.answer.standards == []
    assert result.answer.score is None
    assert events.items[-1].event_type == "final"
    assert len(deps.persistence.commands) == 1
    assert not deps.persistence.final_emitted_before_persist
    assert deps.persistence.commands[0].answer.completed_at
    assert [event.sequence for event in events.items] == list(range(1, len(events.items) + 1))
    assert saver.get_tuple({"configurable": {"thread_id": command.task_id}}) is not None


@pytest.mark.asyncio
async def test_evaluation_intent_is_converted_to_non_evaluative_reading(command, paper_evidence):
    command = command.model_copy(update={"original_question": "论文的实验设计是否充分？"})
    llm = ScriptedLlm([evaluation_route(), fact_draft()])
    retrieval = FakeRetrieval([paper_evidence])
    deps, events, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.answer.route_type == "fact"
    assert llm.calls == ["RouteDecision", "AnswerDraft"]
    assert retrieval.standard_requests == []
    assert not any(event.event_type == "review_summary" for event in events.items)
    assert any("non-evaluative paper reading" in warning for warning in result.answer.warnings)
    assert result.answer.review_opinions == []
    assert result.answer.standards == []


@pytest.mark.asyncio
async def test_semantic_validation_repairs_once(command, paper_evidence):
    llm = ScriptedLlm([fact_route(), invalid_fact_draft(), fact_draft()])
    retrieval = FakeRetrieval([paper_evidence])
    deps, _, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.workflow_summary["repair_count"] == 1
    assert llm.calls.count("AnswerDraft") == 2
    assert not result.answer.is_refusal


@pytest.mark.asyncio
async def test_semantic_repair_transport_failure_uses_next_bounded_repair(
    command, paper_evidence
):
    llm = ScriptedLlm(
        [
            fact_route(),
            invalid_fact_draft(),
            ModelTransportError("provider unavailable"),
            fact_draft(),
        ]
    )
    retrieval = FakeRetrieval([paper_evidence])
    deps, _, traces = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.workflow_summary["repair_count"] == 2
    assert llm.calls.count("AnswerDraft") == 3
    assert not result.answer.is_refusal
    failed_repair = next(
        item
        for item in traces.items
        if item.node_name == "generate_answer" and item.status == "failed"
    )
    assert failed_repair.error_code == "MODEL_TRANSPORT_ERROR"
    assert failed_repair.metrics["semantic_repair"] is True


@pytest.mark.asyncio
async def test_insufficient_evidence_is_a_validated_answer_not_a_refusal(
    command, paper_evidence
):
    llm = ScriptedLlm([fact_route(), insufficient_fact_draft()])
    retrieval = FakeRetrieval([paper_evidence])
    deps, events, traces = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.answer.evidence_sufficient is False
    assert result.answer.evidence_gap_reason
    assert result.answer.is_refusal is False
    assert "EVIDENCE_INSUFFICIENT" in result.answer.warnings
    assert events.items[-1].event_type == "final"
    assert traces.items[-1].node_name == "finalize_insufficient"


@pytest.mark.asyncio
async def test_out_of_scope_skips_retrieval(command):
    command = command.model_copy(update={"original_question": "今天天气如何？"})
    llm = ScriptedLlm([out_of_scope_route()])
    retrieval = FakeRetrieval([])
    deps, _, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.answer.is_refusal
    assert not retrieval.paper_requests


@pytest.mark.asyncio
async def test_general_chat_calls_model_without_retrieving_paper(command):
    command = command.model_copy(
        update={"original_question": "你好，请介绍一下你自己。"}
    )
    llm = ScriptedLlm([general_chat_route(), general_chat_draft()])
    retrieval = FakeRetrieval([])
    deps, _, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.answer.route_type == "general_chat"
    assert not result.answer.is_refusal
    assert not result.answer.evidences
    assert not retrieval.paper_requests
    assert llm.calls == ["RouteDecision", "AnswerDraft"]


def test_answer_prompt_evidence_is_compact_and_drops_internal_metadata(paper_evidence):
    oversized = paper_evidence.model_copy(
        update={
            "quote": "prefix " + ("irrelevant " * 5000) + "target method result",
            "metadata": {"raw_content": "duplicate " * 10000},
        }
    )

    payload = _prompt_evidence([oversized] * 15, "target method")

    assert len(payload) == 7
    assert sum(len(item["quote"]) for item in payload) <= 24_000
    assert all(len(item["quote"]) <= 3_500 for item in payload)
    assert all("metadata" not in item for item in payload)
    assert "target method" in payload[0]["quote"]


@pytest.mark.asyncio
async def test_empty_paper_evidence_returns_refusal(command):
    llm = ScriptedLlm([fact_route()])
    retrieval = FakeRetrieval([])
    deps, events, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.answer.is_refusal
    assert len(retrieval.paper_requests) == 2
    assert events.items[-1].event_type == "final"


@pytest.mark.asyncio
async def test_answer_transport_failure_is_not_reported_as_invalid_output(
    command, paper_evidence
):
    llm = ScriptedLlm([fact_route(), ModelTransportError("provider unavailable")])
    retrieval = FakeRetrieval([paper_evidence])
    deps, _, traces = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.answer.is_refusal
    assert "模型服务暂时不可用" in result.answer.answer
    generate_trace = next(item for item in traces.items if item.node_name == "generate_answer")
    assert generate_trace.error_code == "MODEL_TRANSPORT_ERROR"


@pytest.mark.asyncio
async def test_cancellation_emits_error(command, paper_evidence):
    llm = ScriptedLlm([])
    retrieval = FakeRetrieval([paper_evidence])
    deps, events, _ = dependencies(retrieval, cancelled=True)

    with pytest.raises(WorkflowCancelled):
        await AiWorkflowService(llm).run(command, deps)

    assert events.items[-1].event_type == "error"
    assert events.items[-1].data["code"] == "TASK_CANCELLED"


@pytest.mark.asyncio
async def test_persistence_failure_does_not_expose_internal_error(command, paper_evidence):
    llm = ScriptedLlm([fact_route(), fact_draft()])
    retrieval = FakeRetrieval([paper_evidence])
    deps, events, _ = dependencies(
        retrieval, persistence_error=RuntimeError("database password leaked")
    )

    with pytest.raises(AnswerPersistenceFailed):
        await AiWorkflowService(llm).run(command, deps)

    assert not any(event.event_type == "delta" for event in events.items)
    assert events.items[-1].event_type == "error"
    assert events.items[-1].data == {
        "code": "AI_WORKFLOW_ERROR",
        "message": "工作流未能完成，请稍后重试。",
        "retryable": True,
    }
