import pytest
from pydantic import ValidationError

from app.ai.schemas import StartQaWorkflowCommand
from app.format_review.schemas import ReflectionOutput


def test_command_forbids_client_routing_fields(command):
    payload = command.model_dump(mode="json")
    payload["route_type"] = "review"
    with pytest.raises(ValidationError):
        StartQaWorkflowCommand.model_validate(payload)


def test_command_rejects_duplicate_papers(command):
    payload = command.model_dump(mode="json")
    payload["paper_ids"] = ["paper-1", "paper-1"]
    with pytest.raises(ValidationError):
        StartQaWorkflowCommand.model_validate(payload)


def test_command_round_trips_strictly(command):
    parsed = StartQaWorkflowCommand.model_validate(command.model_dump(mode="json"))
    assert parsed == command


@pytest.mark.parametrize("decision", ["retrieve_standard", "clarify_standard"])
def test_format_reflection_accepts_standard_retrieval_routes(decision):
    output = ReflectionOutput.model_validate(
        {"decision": decision, "reason": "需要补充原子规则原文。"}
    )

    assert output.decision == decision
