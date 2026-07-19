import pytest
from conftest import FakeRetrieval, ScriptedLlm, dependencies
from langgraph.checkpoint.memory import InMemorySaver

from app.ai import AiWorkflowService
from app.ai.errors import ModelOutputInvalid, WorkflowCancelled
from app.ai.schemas import (
    AnswerDraft,
    Claim,
    PaperFact,
    PaperUnderstanding,
    ReviewClaim,
    ReviewOpinion,
    ReviewOpinions,
    RouteDecision,
)


def fact_route():
    return RouteDecision(
        initial_route_type="fact",
        effective_route_type="fact",
        standalone_question="论文使用了什么数据集？",
        review_dimensions=[],
        needs_public_kb=False,
        confidence=0.95,
    )


def review_route():
    return RouteDecision(
        initial_route_type="review",
        effective_route_type="review",
        standalone_question="论文的实验设计是否充分？",
        review_dimensions=["实验充分性"],
        needs_public_kb=True,
        confidence=0.94,
    )


def score_route():
    return RouteDecision(
        initial_route_type="score",
        effective_route_type="score",
        standalone_question="请为论文实验充分性评分。",
        review_dimensions=["实验充分性"],
        needs_public_kb=True,
        confidence=0.94,
    )


def out_of_scope_route():
    return RouteDecision(
        initial_route_type="out_of_scope",
        effective_route_type="out_of_scope",
        standalone_question="今天天气如何？",
        review_dimensions=[],
        needs_public_kb=False,
        confidence=0.99,
    )


def paper_understanding():
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


def fact_draft():
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


def invalid_fact_draft():
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


def reviews(position, verdict=None):
    return ReviewOpinions(
        opinions=[
            ReviewOpinion(
                dimension="实验充分性",
                position=position,
                claims=[
                    ReviewClaim(
                        statement="论文报告了数据集，但仍需结合标准检查实验完整性。",
                        severity="medium",
                        paper_evidence_ids=["P1"],
                        standard_evidence_ids=["S1"],
                        reasoning_summary="论文事实与标准共同支持该判断。",
                        review_a_verdict=verdict,
                    )
                ],
                suggested_score=None,
                confidence=0.8,
            )
        ]
    )


def review_draft(with_standard=True):
    evidence_ids = ["P1", "S1"] if with_standard else ["P1"]
    return AnswerDraft(
        route_type="review",
        answer="论文报告了数据集，但实验完整性仍需结合对比、消融和统计信息判断。",
        claims=[
            Claim(
                claim_id="C1",
                text="当前证据只能支持有限的实验充分性判断",
                verdict="insufficient_evidence",
                confidence=0.75,
                evidence_ids=evidence_ids,
                reason="论文事实与评审标准的覆盖有限",
            )
        ],
        evidence_ids=evidence_ids,
        confidence=0.75,
        warnings=[] if with_standard else ["PUBLIC_KB_UNAVAILABLE"],
    )


@pytest.mark.asyncio
async def test_fact_path_skips_public_retrieval(command, paper_evidence):
    llm = ScriptedLlm([fact_route(), paper_understanding(), fact_draft()])
    retrieval = FakeRetrieval([paper_evidence])
    deps, events, _ = dependencies(retrieval)
    saver = InMemorySaver()

    result = await AiWorkflowService(llm).run(command, deps, saver)

    assert result.answer.answer.startswith("论文使用了")
    assert not retrieval.standard_requests
    assert llm.calls == ["RouteDecision", "PaperUnderstanding", "AnswerDraft"]
    assert events.items[-1].event_type == "final"
    assert [event.sequence for event in events.items] == list(range(1, len(events.items) + 1))
    checkpoint = saver.get_tuple({"configurable": {"thread_id": command.task_id}})
    assert checkpoint is not None


@pytest.mark.asyncio
async def test_review_path_runs_a_then_b(command, paper_evidence, standard_evidence):
    command = command.model_copy(update={"original_question": "论文的实验设计是否充分？"})
    llm = ScriptedLlm(
        [
            review_route(),
            paper_understanding(),
            reviews("critical"),
            reviews("mixed", "partially_supported"),
            review_draft(),
        ]
    )
    retrieval = FakeRetrieval([paper_evidence], [standard_evidence])
    deps, events, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.answer.route_type == "review"
    assert llm.calls == [
        "RouteDecision",
        "PaperUnderstanding",
        "ReviewOpinions",
        "ReviewOpinions",
        "AnswerDraft",
    ]
    assert len(retrieval.standard_requests) == 1
    assert any(event.event_type == "review_summary" for event in events.items)


@pytest.mark.asyncio
async def test_review_degrades_when_public_kb_is_empty(command, paper_evidence):
    command = command.model_copy(update={"original_question": "论文的实验设计是否充分？"})
    llm = ScriptedLlm(
        [review_route(), paper_understanding(), review_draft(with_standard=False)]
    )
    retrieval = FakeRetrieval([paper_evidence], [])
    deps, _, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert not result.answer.is_refusal
    assert "ReviewOpinions" not in llm.calls
    assert any("PUBLIC_KB_UNAVAILABLE" in item for item in result.workflow_summary["warnings"])


@pytest.mark.asyncio
async def test_score_refuses_when_public_kb_is_empty(command, paper_evidence):
    command = command.model_copy(update={"original_question": "请为论文实验充分性评分。"})
    llm = ScriptedLlm([score_route(), paper_understanding()])
    retrieval = FakeRetrieval([paper_evidence], [])
    deps, _, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.answer.is_refusal
    assert result.answer.score is None
    assert "公共评审标准不可用" in result.answer.answer


@pytest.mark.asyncio
async def test_review_a_failure_does_not_block_review_b(command, paper_evidence, standard_evidence):
    command = command.model_copy(update={"original_question": "论文的实验设计是否充分？"})
    llm = ScriptedLlm(
        [
            review_route(),
            paper_understanding(),
            ModelOutputInvalid("review A failed"),
            reviews("mixed", "not_applicable"),
            review_draft(),
        ]
    )
    retrieval = FakeRetrieval([paper_evidence], [standard_evidence])
    deps, _, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert not result.answer.is_refusal
    assert any("review A unavailable" in item for item in result.workflow_summary["warnings"])
    assert any(item.agent_name == "review_b" for item in result.agent_results)


@pytest.mark.asyncio
async def test_semantic_validation_repairs_once(command, paper_evidence):
    llm = ScriptedLlm(
        [fact_route(), paper_understanding(), invalid_fact_draft(), fact_draft()]
    )
    retrieval = FakeRetrieval([paper_evidence])
    deps, _, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.workflow_summary["repair_count"] == 1
    assert llm.calls.count("AnswerDraft") == 2
    assert not result.answer.is_refusal


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
async def test_empty_paper_evidence_returns_refusal(command):
    llm = ScriptedLlm([fact_route()])
    retrieval = FakeRetrieval([])
    deps, events, _ = dependencies(retrieval)

    result = await AiWorkflowService(llm).run(command, deps)

    assert result.answer.is_refusal
    assert len(retrieval.paper_requests) == 2
    assert events.items[-1].event_type == "final"


@pytest.mark.asyncio
async def test_cancellation_emits_error(command, paper_evidence):
    llm = ScriptedLlm([])
    retrieval = FakeRetrieval([paper_evidence])
    deps, events, _ = dependencies(retrieval, cancelled=True)

    with pytest.raises(WorkflowCancelled):
        await AiWorkflowService(llm).run(command, deps)

    assert events.items[-1].event_type == "error"
    assert events.items[-1].data["code"] == "TASK_CANCELLED"
