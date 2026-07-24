"""Ordered answer-validation pipeline."""

from ..schemas import (
    AnswerDraft,
    EvidenceItem,
    PaperUnderstanding,
    RouteDecision,
    ValidationResult,
)
from .answerability import validate_answerability
from .citation import validate_citations
from .numeric import validate_numbers
from .output_sanity import validate_output_sanity


class AnswerValidationPipeline:
    def validate(
        self,
        *,
        draft: AnswerDraft,
        route: RouteDecision,
        understanding: PaperUnderstanding | None,
        evidences: list[EvidenceItem],
        original_question: str,
    ) -> ValidationResult:
        evidence_errors = (
            []
            if route.effective_route_type == "general_chat"
            else [
                *validate_citations(draft, route, evidences),
                *validate_numbers(draft, evidences),
            ]
        )
        errors = [
            *validate_answerability(draft, route, understanding, evidences),
            *evidence_errors,
            *validate_output_sanity(draft, original_question),
        ]
        errors = list(dict.fromkeys(errors))
        return ValidationResult(valid=not errors, errors=errors, warnings=[])
