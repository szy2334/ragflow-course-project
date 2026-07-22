"""Question-intent classifier for the V3 reading workflow."""

from ..runner import AgentRunner
from ..schemas import AgentResult, ConfigurationSnapshot, RouteDecision
from .common import agent_result


class IntentRouterAgent:
    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner

    async def run(
        self,
        *,
        original_question: str,
        conversation_summary: str,
        configuration: ConfigurationSnapshot,
    ) -> tuple[RouteDecision, AgentResult]:
        result = await self._runner.run(
            prompt_name="intent_router",
            prompt_version=configuration.prompt_version,
            output_model=RouteDecision,
            model_config=configuration.model,
            context={
                "original_question": original_question,
                "conversation_summary": conversation_summary,
            },
        )
        decision = normalize_reading_route(result.output, original_question)
        return decision, agent_result(
            name="intent_router",
            summary=f"routed as {decision.initial_route_type}",
            confidence=decision.confidence,
            metrics=result.metrics,
            warnings=decision.warnings,
        )


def normalize_reading_route(
    decision: RouteDecision,
    original_question: str = "",
) -> RouteDecision:
    if decision.effective_route_type == "out_of_scope" and _looks_paper_scoped(
        original_question
    ):
        warning = "out-of-scope decision overridden for an explicit paper-reading question"
        return decision.model_copy(
            update={
                "initial_route_type": "fact",
                "effective_route_type": "fact",
                "review_dimensions": [],
                "needs_public_kb": False,
                "warnings": list(dict.fromkeys([*decision.warnings, warning])),
            }
        )
    if decision.effective_route_type in {"fact", "explain", "out_of_scope"}:
        return decision
    warning = "evaluation intent was handled as non-evaluative paper reading"
    return decision.model_copy(
        update={
            "initial_route_type": "fact",
            "effective_route_type": "fact",
            "review_dimensions": [],
            "needs_public_kb": False,
            "warnings": list(dict.fromkeys([*decision.warnings, warning])),
        }
    )


def _looks_paper_scoped(question: str) -> bool:
    normalized = question.lower()
    paper_terms = (
        "论文",
        "文章",
        "本文",
        "作者",
        "这篇",
        "摘要",
        "方法",
        "实验",
        "数据集",
        "结果",
        "贡献",
        "结论",
        "paper",
        "article",
        "method",
        "experiment",
        "dataset",
        "result",
        "contribution",
        "conclusion",
    )
    return any(term in normalized for term in paper_terms)


def fallback_route(question: str, conversation_summary: str) -> RouteDecision:
    normalized = question.lower().strip()
    explain_tokens = ("为什么", "如何解释", "机制", "why", "explain", "mechanism")
    follow_up_tokens = (
        "这个",
        "上述",
        "那个",
        "那",
        "呢",
        "它",
        "that",
        "those",
        "what about",
    )
    out_tokens = ("天气", "股票", "彩票", "weather", "stock price", "lottery")
    evaluation_tokens = (
        "评分",
        "打分",
        "优缺点",
        "是否合理",
        "是否充分",
        "充分",
        "review",
        "score",
        "weakness",
    )

    initial = "fact"
    effective = "fact"
    warnings = ["deterministic route fallback"]
    if any(token in normalized for token in out_tokens):
        initial = effective = "out_of_scope"
    elif conversation_summary and any(token in normalized for token in follow_up_tokens):
        initial = "follow_up"
    elif any(token in normalized for token in evaluation_tokens):
        warnings.append("evaluation intent was handled as non-evaluative paper reading")
    elif any(token in normalized for token in explain_tokens):
        initial = effective = "explain"

    standalone = question
    if initial == "follow_up":
        standalone = f"基于会话上下文（{conversation_summary}），回答：{question}"
    return RouteDecision(
        initial_route_type=initial,
        effective_route_type=effective,
        standalone_question=standalone,
        review_dimensions=[],
        needs_public_kb=False,
        confidence=0.55,
        warnings=warnings,
    )


__all__ = [
    "IntentRouterAgent",
    "fallback_route",
    "normalize_reading_route",
]
