"""V3 paper-reading graph with two model calls and deterministic validation."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..schemas import AnswerDraft, ReviewGraphState, RouteDecision, ValidationResult
from .master_controller import MasterController
from .policies import WorkflowPolicy


def build_qa_workflow_graph(
    controller: MasterController,
    *,
    policy: WorkflowPolicy,
    checkpointer: Any = None,
):
    graph = StateGraph(ReviewGraphState)
    graph.add_node("load_context", controller.load_context)
    graph.add_node("intent_route", controller.intent_route)
    graph.add_node("paper_retrieve", controller.paper_retrieve)
    graph.add_node("generate_answer", controller.generate_answer)
    graph.add_node("validate", controller.validate)
    graph.add_node("refuse_out_of_scope", controller.refuse_out_of_scope)
    graph.add_node("refuse_failed", controller.refuse_failed)
    graph.add_node("finalize", controller.finalize)
    graph.add_node("finalize_insufficient", controller.finalize_insufficient)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "intent_route")
    graph.add_conditional_edges(
        "intent_route",
        _after_route,
        {"refuse": "refuse_out_of_scope", "retrieve": "paper_retrieve"},
    )
    graph.add_conditional_edges(
        "paper_retrieve",
        _after_required_step,
        {"refuse": "refuse_failed", "continue": "generate_answer"},
    )
    graph.add_conditional_edges(
        "generate_answer",
        _after_required_step,
        {"refuse": "refuse_failed", "continue": "validate"},
    )
    graph.add_conditional_edges(
        "validate",
        lambda state: _after_validation(state, policy),
        {
            "finalize": "finalize",
            "insufficient": "finalize_insufficient",
            "repair": "generate_answer",
            "refuse": "refuse_failed",
        },
    )
    graph.add_edge("refuse_out_of_scope", "finalize")
    graph.add_edge("refuse_failed", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("finalize_insufficient", END)
    return graph.compile(checkpointer=checkpointer)


def _after_route(state: ReviewGraphState) -> str:
    route = RouteDecision.model_validate(state["route_decision"])
    return "refuse" if route.effective_route_type == "out_of_scope" else "retrieve"


def _after_required_step(state: ReviewGraphState) -> str:
    return "refuse" if state.get("error_code") else "continue"


def _after_validation(state: ReviewGraphState, policy: WorkflowPolicy) -> str:
    validation = ValidationResult.model_validate(state["validation"])
    if not validation.valid:
        return (
            "repair"
            if state.get("repair_count", 0) <= policy.max_semantic_repairs
            else "refuse"
        )
    draft = AnswerDraft.model_validate(state["draft_answer"])
    return "finalize" if draft.evidence_sufficient else "insufficient"
