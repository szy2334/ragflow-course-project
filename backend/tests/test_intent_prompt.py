from app.ai.prompts import PromptRepository


def test_intent_prompt_delegates_paper_scope_decision_to_model():
    messages = PromptRepository().render(
        "intent_router",
        "v1",
        original_question="你好",
        conversation_summary="",
    )

    system = messages[0].content
    assert "selected paper" in system
    assert "general_chat" in system
    assert "Make this semantic decision yourself" in system


def test_general_answer_prompt_forbids_paper_attribution():
    messages = PromptRepository().render(
        "general_answer",
        "v1",
        original_question="你好",
        standalone_question="你好",
        answer_language="Simplified Chinese",
        conversation_summary="",
        previous_draft_json="null",
        validation_errors_json="[]",
    )

    assert "Do not inspect, cite, summarize" in messages[0].content
    assert '"route_type": "general_chat"' in messages[0].content


def test_paper_answer_prompt_lists_every_required_answer_field():
    messages = PromptRepository().render(
        "answer_generator",
        "v1",
        original_question="概括创新点",
        standalone_question="概括创新点",
        route_type="explain",
        answer_language="Simplified Chinese",
        paper_summary="summary",
        evidence_json="[]",
        warnings_json="[]",
        previous_draft_json="null",
        validation_errors_json="[]",
    )

    system = messages[0].content
    for field in (
        "route_type",
        "answer",
        "claims",
        "evidence_ids",
        "score",
        "confidence",
        "warnings",
        "evidence_sufficient",
        "evidence_gap_reason",
        "is_refusal",
        "refusal_reason",
    ):
        assert f'"{field}"' in system
