"""Controller agent: routing and evidence-grounded synthesis."""

from ..errors import AiWorkflowError
from ..llm import StructuredLlm
from ..prompts import PromptRepository
from ..schemas import (
    AgentResult,
    AnswerDraft,
    ConfigurationSnapshot,
    EvidenceItem,
    PaperUnderstanding,
    ReviewOpinions,
    RouteDecision,
)
from .common import agent_result, as_json


class ControllerAgent:
    def __init__(self, llm: StructuredLlm, prompts: PromptRepository) -> None:
        self._llm = llm
        self._prompts = prompts

    async def route(
        self,
        *,
        original_question: str,
        conversation_summary: str,
        configuration: ConfigurationSnapshot,
    ) -> tuple[RouteDecision, AgentResult]:
        messages = self._prompts.render(
            "controller_route",
            configuration.prompt_version,
            original_question=original_question,
            conversation_summary=conversation_summary,
        )
        try:
            result = await self._llm.invoke_structured(messages, RouteDecision, configuration.model)
            decision = result.output
            return decision, agent_result(
                name="controller",
                summary=f"routed as {decision.initial_route_type}",
                confidence=decision.confidence,
                metrics=result.metrics,
                warnings=decision.warnings,
            )
        except AiWorkflowError:
            decision = fallback_route(original_question, conversation_summary)
            warning = "controller model failed; deterministic routing fallback used"
            decision = decision.model_copy(update={"warnings": [*decision.warnings, warning]})
            return decision, agent_result(
                name="controller",
                summary=f"fallback routed as {decision.initial_route_type}",
                confidence=decision.confidence,
                metrics=None,
                warnings=decision.warnings,
            )

    async def synthesize(
        self,
        *,
        original_question: str,
        route: RouteDecision,
        paper_understanding: PaperUnderstanding,
        review_a: ReviewOpinions | None,
        review_b: ReviewOpinions | None,
        evidences: list[EvidenceItem],
        warnings: list[str],
        configuration: ConfigurationSnapshot,
        previous_draft: AnswerDraft | None = None,
        validation_errors: list[str] | None = None,
    ) -> tuple[AnswerDraft, AgentResult]:
        messages = self._prompts.render(
            "synthesis",
            configuration.prompt_version,
            original_question=original_question,
            standalone_question=route.standalone_question,
            initial_route_type=route.initial_route_type,
            effective_route_type=route.effective_route_type,
            paper_understanding_json=as_json(paper_understanding),
            review_a_json=as_json(review_a) if review_a else "null",
            review_b_json=as_json(review_b) if review_b else "null",
            evidence_json=as_json([item.model_dump(mode="json") for item in evidences]),
            standard_version=configuration.standard_version or "unavailable",
            warnings_json=as_json(warnings),
            previous_draft_json=as_json(previous_draft) if previous_draft else "null",
            validation_errors_json=as_json(validation_errors or []),
        )
        result = await self._llm.invoke_structured(messages, AnswerDraft, configuration.model)
        draft = result.output
        if draft.route_type != route.initial_route_type:
            draft = draft.model_copy(update={"route_type": route.initial_route_type})
        evidence_ids = [item for claim in draft.claims for item in claim.evidence_ids]
        return draft, agent_result(
            name="controller",
            summary="synthesized final candidate answer",
            confidence=draft.confidence,
            metrics=result.metrics,
            claims=draft.claims,
            evidence_ids=evidence_ids,
            warnings=draft.warnings,
        )


def fallback_route(question: str, conversation_summary: str) -> RouteDecision:
    normalized = question.lower().strip()
    score_tokens = ("评分", "打分", "几分", "score", "rating", "grade")
    review_tokens = (
        "是否合理",
        "是否充分",
        "优缺点",
        "问题",
        "评价",
        "review",
        "weakness",
        "sufficient",
    )
    explain_tokens = ("为什么", "如何解释", "机制", "why", "explain", "mechanism")
    follow_up_tokens = ("这个", "上述", "那", "它", "that", "those", "what about")
    out_tokens = ("天气", "股票", "彩票", "weather", "stock price", "lottery")

    initial = "fact"
    effective = "fact"
    dimensions: list[str] = []
    if any(token in normalized for token in out_tokens):
        initial = effective = "out_of_scope"
    elif conversation_summary and any(token in normalized for token in follow_up_tokens):
        initial = "follow_up"
        effective = "review" if any(token in normalized for token in review_tokens) else "fact"
    elif any(token in normalized for token in score_tokens):
        initial = effective = "score"
    elif any(token in normalized for token in review_tokens):
        initial = effective = "review"
    elif any(token in normalized for token in explain_tokens):
        initial = effective = "explain"
    if effective in {"review", "score"}:
        dimensions = ["实验充分性"]
    standalone = question
    if initial == "follow_up":
        standalone = f"基于会话上下文（{conversation_summary}），回答：{question}"
    return RouteDecision(
        initial_route_type=initial,
        effective_route_type=effective,
        standalone_question=standalone,
        review_dimensions=dimensions,
        needs_public_kb=effective in {"review", "score"},
        confidence=0.55,
        warnings=["deterministic route fallback"],
    )
