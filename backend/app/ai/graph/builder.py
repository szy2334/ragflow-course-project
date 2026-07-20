"""StateGraph construction and conditional edges."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..schemas import ReviewGraphState, RouteDecision, ValidationResult
from .nodes import WorkflowNodes
from .policies import WorkflowPolicy


def build_workflow_graph(
    nodes: WorkflowNodes,
    *,
    policy: WorkflowPolicy,
    checkpointer: Any = None,
):
    graph = StateGraph(ReviewGraphState)
    graph.add_node("load_context", nodes.load_context)
    graph.add_node("route", nodes.route)
    graph.add_node("paper_retrieve", nodes.paper_retrieve)
    graph.add_node("paper_understand", nodes.paper_understand)
    graph.add_node("standard_retrieve", nodes.standard_retrieve)
    graph.add_node("review_a", nodes.review_a)
    graph.add_node("review_b", nodes.review_b)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("validate", nodes.validate)
    graph.add_node("safe_refusal", nodes.safe_refusal)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "route")
    graph.add_conditional_edges(
        "route",
        _after_route,
        {"refuse": "safe_refusal", "retrieve": "paper_retrieve"},
    )
    graph.add_conditional_edges(
        "paper_retrieve",
        _after_required_step,
        {"refuse": "safe_refusal", "continue": "paper_understand"},
    )
    graph.add_conditional_edges(
        "paper_understand",
        _after_understanding,
        {
            "refuse": "safe_refusal",
            "standards": "standard_retrieve",
            "synthesize": "synthesize",
        },
    )
    graph.add_conditional_edges(
        "standard_retrieve",
        _after_standards,
        {"review": "review_a", "synthesize": "synthesize", "refuse": "safe_refusal"},
    )
    graph.add_edge("review_a", "review_b")
    graph.add_edge("review_b", "synthesize")
    graph.add_conditional_edges(
        "synthesize",
        _after_synthesis,
        {"refuse": "safe_refusal", "validate": "validate"},
    )
    graph.add_conditional_edges(
        "validate",
        lambda state: _after_validation(state, policy),
        {"finalize": "finalize", "repair": "synthesize", "refuse": "safe_refusal"},
    )
    graph.add_edge("safe_refusal", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def _after_route(state: ReviewGraphState) -> str:
    route = RouteDecision.model_validate(state["route_decision"])
    return "refuse" if route.effective_route_type == "out_of_scope" else "retrieve"


def _after_required_step(state: ReviewGraphState) -> str:
    return "refuse" if state.get("error_code") else "continue"


def _after_understanding(state: ReviewGraphState) -> str:
    if state.get("error_code"):
        return "refuse"
    return "standards"


def _after_standards(state: ReviewGraphState) -> str:
    if state.get("error_code"):
        return "refuse"
    route = RouteDecision.model_validate(state["route_decision"])
    if route.effective_route_type in {"review", "score"} and not state.get("skip_reviews"):
        return "review"
    return "synthesize"


def _after_synthesis(state: ReviewGraphState) -> str:
    return "refuse" if state.get("error_code") else "validate"


def _after_validation(state: ReviewGraphState, policy: WorkflowPolicy) -> str:
    validation = ValidationResult.model_validate(state["validation"])
    if validation.valid:
        return "finalize"
    if state.get("repair_count", 0) <= policy.max_semantic_repairs:
        return "repair"
    return "refuse"
