"""Controller agent: routing and evidence-grounded synthesis."""

from collections.abc import Awaitable, Callable

from ..errors import AiWorkflowError, ModelOutputInvalid
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
        on_answer_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[AnswerDraft, AgentResult, bool]:
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
        streamed = False
        stream = getattr(self._llm, "invoke_structured_stream", None)
        if on_answer_delta is not None and callable(stream):
            extractor = _JsonStringFieldExtractor("answer")

            async def relay(content: str) -> None:
                nonlocal streamed
                delta = extractor.feed(content)
                if delta:
                    streamed = True
                    await on_answer_delta(delta)

            result = await stream(messages, AnswerDraft, configuration.model, relay)
        else:
            result = await self._llm.invoke_structured(messages, AnswerDraft, configuration.model)
        draft = result.output
        # Reading answers are descriptive only.  Do not allow a malformed or
        # overly eager model response to turn this workflow into a score or a
        # peer-review conclusion.
        if route.effective_route_type not in {"fact", "explain"}:
            raise ModelOutputInvalid("reading route cannot be synthesized")
        if draft.route_type != route.effective_route_type or draft.score is not None:
            draft = draft.model_copy(
                update={"route_type": route.effective_route_type, "score": None}
            )
        evidence_ids = [item for claim in draft.claims for item in claim.evidence_ids]
        return draft, agent_result(
            name="controller",
            summary="synthesized final candidate answer",
            confidence=draft.confidence,
            metrics=result.metrics,
            claims=draft.claims,
            evidence_ids=evidence_ids,
            warnings=draft.warnings,
        ), streamed


class _JsonStringFieldExtractor:
    """Extract one JSON string field incrementally from a model token stream."""

    def __init__(self, field: str) -> None:
        self._marker = f'"{field}"'
        self._prefix = ""
        self._started = False
        self._finished = False
        self._escaped = False
        self._unicode: str | None = None

    def feed(self, content: str) -> str:
        if self._finished:
            return ""
        pending = content
        if not self._started:
            self._prefix += pending
            marker_index = self._prefix.find(self._marker)
            if marker_index < 0:
                self._prefix = self._prefix[-len(self._marker) - 16 :]
                return ""
            value_start = self._prefix.find(":", marker_index + len(self._marker))
            if value_start < 0:
                return ""
            quote_start = self._prefix.find('"', value_start + 1)
            if quote_start < 0:
                return ""
            self._started = True
            pending = self._prefix[quote_start + 1 :]
            self._prefix = ""

        output: list[str] = []
        for char in pending:
            if self._unicode is not None:
                self._unicode += char
                if len(self._unicode) == 4:
                    try:
                        output.append(chr(int(self._unicode, 16)))
                    except ValueError:
                        output.append("\\u" + self._unicode)
                    self._unicode = None
                    self._escaped = False
                continue
            if self._escaped:
                escaped = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}
                if char == "u":
                    self._unicode = ""
                else:
                    output.append(escaped.get(char, char))
                    self._escaped = False
                continue
            if char == "\\":
                self._escaped = True
            elif char == '"':
                self._finished = True
                break
            else:
                output.append(char)
        return "".join(output)


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
    warnings: list[str] = ["deterministic route fallback"]
    if any(token in normalized for token in out_tokens):
        initial = effective = "out_of_scope"
    elif conversation_summary and any(token in normalized for token in follow_up_tokens):
        initial = "follow_up"
        effective = "fact"
        if any(token in normalized for token in (*score_tokens, *review_tokens)):
            warnings.append("evaluation intent was handled as non-evaluative paper reading")
    elif any(token in normalized for token in (*score_tokens, *review_tokens)):
        initial = "fact"
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
