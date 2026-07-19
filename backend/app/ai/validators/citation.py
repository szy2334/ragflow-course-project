"""Citation membership and coverage checks."""

from ..schemas import AnswerDraft, EvidenceItem, RouteDecision


def validate_citations(
    draft: AnswerDraft,
    route: RouteDecision,
    evidences: list[EvidenceItem],
) -> list[str]:
    errors: list[str] = []
    source_by_id = {item.evidence_id: item.source_type for item in evidences}
    allowed = set(source_by_id)
    standards_available = any(value == "standard" for value in source_by_id.values())
    declared = set(draft.evidence_ids)
    if unknown := declared - allowed:
        errors.append(f"answer declares unknown evidence IDs: {sorted(unknown)}")

    for claim in draft.claims:
        ids = set(claim.evidence_ids)
        if unknown := ids - allowed:
            errors.append(f"claim {claim.claim_id} cites unknown evidence IDs: {sorted(unknown)}")
            continue
        if not ids:
            errors.append(f"claim {claim.claim_id} has no citations")
            continue
        if not any(source_by_id[item] == "paper" for item in ids):
            errors.append(f"claim {claim.claim_id} has no paper evidence")
        requires_standard = route.effective_route_type == "score" or (
            route.effective_route_type == "review" and standards_available
        )
        if requires_standard and not any(
            source_by_id[item] == "standard" for item in ids
        ):
            errors.append(f"review claim {claim.claim_id} has no standard evidence")
        if not ids.issubset(declared):
            errors.append(f"claim {claim.claim_id} uses evidence omitted from answer.evidence_ids")

    if draft.score is not None:
        score_ids = set(draft.score.standard_evidence_ids)
        if unknown := score_ids - allowed:
            errors.append(f"score cites unknown standard IDs: {sorted(unknown)}")
        elif not all(source_by_id[item] == "standard" for item in score_ids):
            errors.append("score may cite only standard evidence")
    if route.effective_route_type == "score" and not draft.is_refusal and draft.score is None:
        errors.append("score route requires a score or an explicit refusal")
    return errors
