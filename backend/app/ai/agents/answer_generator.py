"""Direct evidence-to-answer agent for the V3 reading workflow."""

import re
from collections.abc import Awaitable, Callable
from typing import Any

from ..runner import AgentRunner
from ..schemas import (
    AgentResult,
    AnswerDraft,
    ConfigurationSnapshot,
    EvidenceItem,
    RouteDecision,
)
from .common import JsonStringFieldExtractor, agent_result, as_json


class AnswerGeneratorAgent:
    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner

    async def run(
        self,
        *,
        original_question: str,
        route: RouteDecision,
        evidences: list[EvidenceItem],
        paper_summary: str,
        warnings: list[str],
        configuration: ConfigurationSnapshot,
        previous_draft: AnswerDraft | None = None,
        validation_errors: list[str] | None = None,
        on_answer_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[AnswerDraft, AgentResult, bool]:
        streamed = False
        extractor = JsonStringFieldExtractor("answer")

        async def relay(content: str) -> None:
            nonlocal streamed
            delta = extractor.feed(content)
            if delta and on_answer_delta is not None:
                streamed = True
                await on_answer_delta(delta)

        result = await self._runner.run(
            prompt_name="answer_generator",
            prompt_version=configuration.prompt_version,
            output_model=AnswerDraft,
            model_config=configuration.model,
            context={
                "original_question": original_question,
                "standalone_question": route.standalone_question,
                "route_type": route.effective_route_type,
                "answer_language": (
                    "Simplified Chinese"
                    if any("\u3400" <= char <= "\u9fff" for char in original_question)
                    else "the language of the original question"
                ),
                "paper_summary": paper_summary or "unavailable",
                # Persistence keeps rich provenance and raw OCR metadata, but
                # those fields are not useful to the answer model and can
                # duplicate the source text.  Use a bounded citeable payload.
                "evidence_json": as_json(
                    _prompt_evidence(evidences, route.standalone_question)
                ),
                "warnings_json": as_json(warnings),
                "previous_draft_json": as_json(previous_draft) if previous_draft else "null",
                "validation_errors_json": as_json(validation_errors or []),
            },
            on_content=relay if on_answer_delta is not None else None,
        )
        draft = result.output
        if draft.route_type != route.effective_route_type or draft.score is not None:
            draft = draft.model_copy(
                update={"route_type": route.effective_route_type, "score": None}
            )
        # This top-level field is only a denormalized union of claim citations.
        # Derive it so forgetting the duplicate field cannot trigger a full
        # model regeneration after otherwise valid claim-level citations.
        claim_evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for claim in draft.claims
                for evidence_id in claim.evidence_ids
            )
        )
        if draft.evidence_ids != claim_evidence_ids:
            draft = draft.model_copy(update={"evidence_ids": claim_evidence_ids})
        cited = [item for claim in draft.claims for item in claim.evidence_ids]
        return draft, agent_result(
            name="answer_generator",
            summary="generated an evidence-grounded answer candidate",
            confidence=draft.confidence,
            metrics=result.metrics,
            claims=draft.claims,
            evidence_ids=cited,
            warnings=draft.warnings,
        ), streamed


_MAX_PROMPT_EVIDENCE_ITEMS = 10
_MAX_PROMPT_EVIDENCE_CHARS = 24_000
_MAX_PROMPT_QUOTE_CHARS = 3_500
_EXCERPT_CONTEXT_CHARS = 500


def _prompt_evidence(
    evidences: list[EvidenceItem], question: str
) -> list[dict[str, Any]]:
    """Build compact model input while retaining full evidence in application state."""

    payload: list[dict[str, Any]] = []
    remaining = _MAX_PROMPT_EVIDENCE_CHARS
    for item in evidences[:_MAX_PROMPT_EVIDENCE_ITEMS]:
        if remaining <= 0:
            break
        quote = _relevant_excerpt(
            item.quote,
            question,
            limit=min(_MAX_PROMPT_QUOTE_CHARS, remaining),
        )
        if not quote:
            continue
        payload.append(
            {
                "evidence_id": item.evidence_id,
                "content_type": item.content_type,
                "content_role": item.content_role,
                "section_title": item.section_title,
                "page_number": item.page_number,
                "quote": quote,
            }
        )
        remaining -= len(quote)
    return payload


def _relevant_excerpt(text: str, question: str, *, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized

    terms = sorted(
        {
            *re.findall(r"[a-z][a-z0-9_-]{2,}", question.lower()),
            *re.findall(r"\d+(?:\.\d+)?%?", question),
            *(
                sequence[index : index + size]
                for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", question)
                for size in (2, 3)
                for index in range(len(sequence) - size + 1)
            ),
        },
        key=len,
        reverse=True,
    )
    lowered = normalized.lower()
    positions = [lowered.find(term.lower()) for term in terms]
    positions = [position for position in positions if position >= 0]
    centre = min(positions) if positions else 0
    start = max(0, centre - _EXCERPT_CONTEXT_CHARS)
    end = min(len(normalized), start + limit)
    start = max(0, end - limit)
    prefix = "... " if start else ""
    suffix = " ..." if end < len(normalized) else ""
    body_budget = max(0, limit - len(prefix) - len(suffix))
    if end == len(normalized):
        body = normalized[max(0, end - body_budget) : end]
    else:
        body = normalized[start : start + body_budget]
    return f"{prefix}{body.strip()}{suffix}"
