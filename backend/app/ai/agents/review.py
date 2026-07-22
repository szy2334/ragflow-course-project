"""Strict and balanced reviewer adapters."""

from ..llm import ModelCallMetrics, StructuredLlm
from ..prompts import PromptRepository
from ..schemas import (
    AgentResult,
    Claim,
    ConfigurationSnapshot,
    EvidenceItem,
    PaperUnderstanding,
    ReviewOpinions,
)
from .common import agent_result, as_json


class ReviewAgentA:
    def __init__(self, llm: StructuredLlm, prompts: PromptRepository) -> None:
        self._llm = llm
        self._prompts = prompts

    async def run(
        self,
        *,
        standalone_question: str,
        dimensions: list[str],
        paper_understanding: PaperUnderstanding,
        paper_evidences: list[EvidenceItem],
        standard_evidences: list[EvidenceItem],
        configuration: ConfigurationSnapshot,
    ) -> tuple[ReviewOpinions, AgentResult]:
        messages = self._prompts.render(
            "review_a",
            configuration.prompt_version,
            standalone_question=standalone_question,
            dimensions_json=as_json(dimensions),
            paper_understanding_json=as_json(paper_understanding),
            paper_evidence_json=as_json([item.model_dump(mode="json") for item in paper_evidences]),
            standard_evidence_json=as_json(
                [item.model_dump(mode="json") for item in standard_evidences]
            ),
        )
        result = await self._llm.invoke_structured(messages, ReviewOpinions, configuration.model)
        return result.output, _review_result("review_a", result.output, result.metrics)


class ReviewAgentB:
    def __init__(self, llm: StructuredLlm, prompts: PromptRepository) -> None:
        self._llm = llm
        self._prompts = prompts

    async def run(
        self,
        *,
        standalone_question: str,
        dimensions: list[str],
        paper_understanding: PaperUnderstanding,
        review_a: ReviewOpinions | None,
        paper_evidences: list[EvidenceItem],
        standard_evidences: list[EvidenceItem],
        configuration: ConfigurationSnapshot,
    ) -> tuple[ReviewOpinions, AgentResult]:
        messages = self._prompts.render(
            "review_b",
            configuration.prompt_version,
            standalone_question=standalone_question,
            dimensions_json=as_json(dimensions),
            paper_understanding_json=as_json(paper_understanding),
            review_a_json=as_json(review_a) if review_a else "null",
            paper_evidence_json=as_json([item.model_dump(mode="json") for item in paper_evidences]),
            standard_evidence_json=as_json(
                [item.model_dump(mode="json") for item in standard_evidences]
            ),
        )
        result = await self._llm.invoke_structured(messages, ReviewOpinions, configuration.model)
        return result.output, _review_result("review_b", result.output, result.metrics)


def _review_result(name: str, reviews: ReviewOpinions, metrics: ModelCallMetrics) -> AgentResult:
    claims: list[Claim] = []
    evidence_ids: list[str] = []
    warnings: list[str] = []
    confidences: list[float] = []
    for opinion_index, opinion in enumerate(reviews.opinions, start=1):
        confidences.append(opinion.confidence)
        warnings.extend(opinion.warnings)
        for claim_index, review_claim in enumerate(opinion.claims, start=1):
            ids = [*review_claim.paper_evidence_ids, *review_claim.standard_evidence_ids]
            evidence_ids.extend(ids)
            claims.append(
                Claim(
                    claim_id=f"{name}-{opinion_index}-{claim_index}",
                    text=review_claim.statement,
                    verdict=(
                        "refuted" if review_claim.review_a_verdict == "unsupported" else "supported"
                    ),
                    confidence=opinion.confidence,
                    evidence_ids=ids,
                    reason=review_claim.reasoning_summary,
                )
            )
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return agent_result(
        name=name,
        summary=f"produced {len(reviews.opinions)} review opinions",
        confidence=confidence,
        metrics=metrics,
        claims=claims,
        evidence_ids=evidence_ids,
        warnings=warnings,
    )
