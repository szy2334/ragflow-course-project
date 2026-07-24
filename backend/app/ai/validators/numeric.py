"""Conservative numeric consistency checks."""

import re
from decimal import Decimal, InvalidOperation

from ..schemas import AnswerDraft, EvidenceItem

NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?(?![\w.])")


def validate_numbers(draft: AnswerDraft, evidences: list[EvidenceItem]) -> list[str]:
    if draft.is_refusal:
        return []
    evidence_by_id = {item.evidence_id: item.quote for item in evidences}
    all_evidence_numbers = _numbers(" ".join(evidence_by_id.values()))
    derived_numbers: set[str] = set()
    if draft.score is not None:
        derived_numbers.update(
            {_normalize(str(draft.score.value)), _normalize(str(draft.score.scale))}
        )

    errors: list[str] = []
    answer_text = _without_evidence_ids(draft.answer, evidence_by_id)
    for number in _numbers(answer_text):
        if number not in all_evidence_numbers and number not in derived_numbers:
            errors.append(f"answer number {number} is not present in cited evidence")

    for claim in draft.claims:
        if claim.type == "negative":
            continue
        cited_text = " ".join(evidence_by_id.get(item, "") for item in claim.evidence_ids)
        cited_numbers = _numbers(cited_text)
        claim_text = _without_evidence_ids(claim.text, evidence_by_id)
        for number in _numbers(claim_text):
            if number not in cited_numbers and number not in derived_numbers:
                errors.append(
                    f"claim {claim.claim_id} number {number} is not present in its citations"
                )
    return errors


def _without_evidence_ids(text: str, evidence_by_id: dict[str, str]) -> str:
    """Remove structured citation identifiers before extracting factual numbers."""

    for evidence_id in sorted(evidence_by_id, key=len, reverse=True):
        text = text.replace(evidence_id, "")
    return text


def _numbers(text: str) -> set[str]:
    return {_normalize(match.group(0)) for match in NUMBER_PATTERN.finditer(text)}


def _normalize(value: str) -> str:
    suffix = "%" if value.endswith("%") else ""
    raw = value[:-1] if suffix else value
    try:
        normalized = format(Decimal(raw).normalize(), "f")
    except InvalidOperation:
        normalized = raw
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return f"{normalized}{suffix}"
