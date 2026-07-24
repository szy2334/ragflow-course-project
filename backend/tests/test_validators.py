from app.ai.schemas import (
    AnswerDraft,
    Claim,
    PaperFact,
    PaperUnderstanding,
    RouteDecision,
)
from app.ai.validators import AnswerValidationPipeline


def route(route_type="fact"):
    return RouteDecision(
        initial_route_type=route_type,
        effective_route_type=route_type,
        standalone_question="论文使用了什么数据集？",
        review_dimensions=["实验充分性"] if route_type in {"review", "score"} else [],
        needs_public_kb=route_type in {"review", "score"},
        confidence=0.9,
    )


def understanding():
    return PaperUnderstanding(
        answerable=True,
        facts=[
            PaperFact(
                claim="使用 CIFAR-10",
                evidence_ids=["P1"],
                evidence_status="explicit",
                confidence=0.9,
            )
        ],
        paper_summary="论文说明了数据集。",
    )


def test_valid_answer_passes(paper_evidence):
    draft = AnswerDraft(
        route_type="fact",
        answer="论文使用了 CIFAR-10 数据集，共 60000 张图像。",
        claims=[
            Claim(
                claim_id="C1",
                text="数据集包含 60000 张图像",
                verdict="supported",
                confidence=0.9,
                evidence_ids=["P1"],
                reason="原文明确陈述",
            )
        ],
        evidence_ids=["P1"],
        confidence=0.9,
    )
    result = AnswerValidationPipeline().validate(
        draft=draft,
        route=route(),
        understanding=understanding(),
        evidences=[paper_evidence],
        original_question="论文使用了什么数据集？",
    )
    assert result.valid


def test_unknown_citation_and_changed_number_fail(paper_evidence):
    draft = AnswerDraft(
        route_type="fact",
        answer="论文使用了 70000 张图像。",
        claims=[
            Claim(
                claim_id="C1",
                text="包含 70000 张图像",
                verdict="supported",
                confidence=0.9,
                evidence_ids=["P9"],
                reason="测试",
            )
        ],
        evidence_ids=["P9"],
        confidence=0.9,
    )
    result = AnswerValidationPipeline().validate(
        draft=draft,
        route=route(),
        understanding=understanding(),
        evidences=[paper_evidence],
        original_question="论文使用了什么数据集？",
    )
    assert not result.valid
    assert any("unknown evidence" in error for error in result.errors)
    assert any("70000" in error for error in result.errors)


def test_evidence_uuid_segments_are_not_treated_as_answer_numbers(paper_evidence):
    evidence_id = "25387b81-ae50-5768-9bab-12bc76fa01c4"
    evidence = paper_evidence.model_copy(update={"evidence_id": evidence_id})
    draft = AnswerDraft(
        route_type="fact",
        answer=f"论文使用了 CIFAR-10 数据集，共 60000 张图像。[证据：{evidence_id}]",
        claims=[
            Claim(
                claim_id="C1",
                text="数据集包含 60000 张图像",
                verdict="supported",
                confidence=0.9,
                evidence_ids=[evidence_id],
                reason="原文明确陈述",
            )
        ],
        evidence_ids=[evidence_id],
        confidence=0.9,
    )

    result = AnswerValidationPipeline().validate(
        draft=draft,
        route=route(),
        understanding=understanding(),
        evidences=[evidence],
        original_question="论文使用了什么数据集？",
    )

    assert result.valid


def test_general_chat_does_not_require_paper_evidence():
    draft = AnswerDraft(
        route_type="general_chat",
        answer="今天是一个适合安排学习计划的日子。",
        claims=[],
        evidence_ids=[],
        confidence=0.8,
    )
    result = AnswerValidationPipeline().validate(
        draft=draft,
        route=route("general_chat"),
        understanding=None,
        evidences=[],
        original_question="今天适合做什么？",
    )

    assert result.valid
