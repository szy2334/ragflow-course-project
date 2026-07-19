"""Commands accepted from the authenticated API/task layer."""

from pydantic import Field, field_validator

from .answer import AgentResult, AnswerView
from .base import StrictModel
from .config import ConfigurationSnapshot


class StartQaWorkflowCommand(StrictModel):
    request_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    paper_ids: list[str] = Field(min_length=1, max_length=10)
    original_question: str = Field(min_length=1, max_length=8000)
    configuration: ConfigurationSnapshot

    @field_validator("paper_ids")
    @classmethod
    def unique_paper_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("paper_ids cannot contain blank identifiers")
        if len(value) != len(set(value)):
            raise ValueError("paper_ids must be unique")
        return value


class PersistAnswerCommand(StrictModel):
    request_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    answer: AnswerView
    agent_results: list[AgentResult] = Field(default_factory=list)
    configuration: ConfigurationSnapshot
