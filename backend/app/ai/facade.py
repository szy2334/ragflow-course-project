"""Single public entry point for the AI workflow package."""

import json
from contextlib import suppress
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from .agents import (
    AnswerGeneratorAgent,
    ControllerAgent,
    IntentRouterAgent,
    PaperUnderstandingAgent,
    ReviewAgentA,
    ReviewAgentB,
)
from .errors import AiWorkflowError
from .graph import (
    MasterController,
    WorkflowEventEmitter,
    WorkflowNodes,
    WorkflowPolicy,
    build_qa_workflow_graph,
    build_workflow_graph,
)
from .llm import StructuredLlm
from .ports import WorkflowDependencies
from .prompts import PromptRepository
from .runner import AgentRunner
from .schemas import (
    AgentResult,
    AnswerView,
    RouteDecision,
    StartQaWorkflowCommand,
    WorkflowResult,
)
from .validators import AnswerValidationPipeline


class AiWorkflowService:
    def __init__(
        self,
        llm: StructuredLlm,
        *,
        prompts: PromptRepository | None = None,
        policy: WorkflowPolicy | None = None,
        use_v3: bool = True,
    ) -> None:
        self._llm = llm
        self._prompts = prompts or PromptRepository()
        self._policy = policy or WorkflowPolicy()
        self._use_v3 = use_v3

    async def run(
        self,
        command: StartQaWorkflowCommand,
        dependencies: WorkflowDependencies,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> WorkflowResult:
        events = WorkflowEventEmitter(dependencies.events, command)
        if self._use_v3:
            runner = AgentRunner(self._llm, self._prompts)
            master = MasterController(
                dependencies=dependencies,
                intent_router=IntentRouterAgent(runner),
                answer_generator=AnswerGeneratorAgent(runner),
                validators=AnswerValidationPipeline(),
                events=events,
                policy=self._policy,
            )
            graph = build_qa_workflow_graph(
                master,
                policy=self._policy,
                checkpointer=checkpointer,
            )
        else:
            controller = ControllerAgent(self._llm, self._prompts)
            nodes = WorkflowNodes(
                dependencies=dependencies,
                controller=controller,
                paper_agent=PaperUnderstandingAgent(self._llm, self._prompts),
                review_a=ReviewAgentA(self._llm, self._prompts),
                review_b=ReviewAgentB(self._llm, self._prompts),
                validators=AnswerValidationPipeline(),
                events=events,
                policy=self._policy,
            )
            graph = build_workflow_graph(
                nodes,
                policy=self._policy,
                checkpointer=checkpointer,
            )
        initial_state = {
            "command": command.model_dump(mode="json"),
            "conversation_summary": "",
            "paper_evidences": [],
            "paper_summary": "",
            "standard_evidences": [],
            "paper_understanding": None,
            "review_a": None,
            "review_b": None,
            "draft_answer": None,
            "validation": None,
            "final_answer": None,
            "agent_results": [],
            "warnings": [],
            "repair_count": 0,
            "error_code": None,
            "error_message": None,
            "answer_streamed": False,
            "skip_reviews": False,
            "sequence": 0,
        }
        graph_config = {"configurable": {"thread_id": command.task_id}}
        try:
            state = await graph.ainvoke(initial_state, graph_config)
            answer = AnswerView.model_validate_json(json.dumps(state["final_answer"]))
        except Exception as exc:
            if not events.terminal:
                code = exc.code if isinstance(exc, AiWorkflowError) else "AI_WORKFLOW_ERROR"
                retryable = exc.retryable if isinstance(exc, AiWorkflowError) else False
                message = (
                    "任务已停止。" if code == "TASK_CANCELLED" else "工作流未能完成，请稍后重试。"
                )
                with suppress(Exception):
                    await events.emit(
                        "error",
                        {"code": code, "message": message, "retryable": retryable},
                    )
            raise

        route = RouteDecision.model_validate(state["route_decision"])
        agent_results = [AgentResult.model_validate(item) for item in state["agent_results"]]
        return WorkflowResult(
            answer=answer,
            agent_results=agent_results,
            workflow_summary={
                "graph_version": command.configuration.graph_version,
                "prompt_version": command.configuration.prompt_version,
                "schema_version": command.configuration.schema_version,
                "initial_route_type": route.initial_route_type,
                "effective_route_type": route.effective_route_type,
                "repair_count": state.get("repair_count", 0),
                "warnings": state.get("warnings", []),
                "final_sequence": state.get("sequence", events.sequence),
                "architecture": "qa-v3" if self._use_v3 else "legacy",
            },
        )
