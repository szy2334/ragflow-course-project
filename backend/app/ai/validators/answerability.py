"""Answerability gate."""

from ..schemas import AnswerDraft, EvidenceItem, PaperUnderstanding, RouteDecision


def validate_answerability(
    draft: AnswerDraft,
    route: RouteDecision,
    understanding: PaperUnderstanding | None,
    evidences: list[EvidenceItem],
) -> list[str]:
    if draft.is_refusal:
        return []
    if route.effective_route_type == "out_of_scope":
        return ["out_of_scope route must return a refusal"]
    if understanding is None or not understanding.answerable:
        return ["paper evidence does not make the question answerable"]
    if not any(item.source_type == "paper" for item in evidences):
        return ["non-refusal answer requires paper evidence"]
    if not draft.claims:
        return ["non-refusal answer requires at least one verified claim"]
    return []
