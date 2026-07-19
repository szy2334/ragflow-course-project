"""Paper-understanding agent adapter."""

from ..llm import StructuredLlm
from ..prompts import PromptRepository
from ..schemas import (
    AgentResult,
    Claim,
    ConfigurationSnapshot,
    EvidenceItem,
    PaperUnderstanding,
)
from .common import agent_result, as_json


class PaperUnderstandingAgent:
    def __init__(self, llm: StructuredLlm, prompts: PromptRepository) -> None:
        self._llm = llm
        self._prompts = prompts

    async def run(
        self,
        *,
        standalone_question: str,
        evidences: list[EvidenceItem],
        configuration: ConfigurationSnapshot,
    ) -> tuple[PaperUnderstanding, AgentResult]:
        messages = self._prompts.render(
            "paper_understanding",
            configuration.prompt_version,
            standalone_question=standalone_question,
            evidence_json=as_json([item.model_dump(mode="json") for item in evidences]),
        )
        result = await self._llm.invoke_structured(
            messages, PaperUnderstanding, configuration.model
        )
        understanding = result.output
        claims = [
            Claim(
                claim_id=f"paper-fact-{index}",
                text=fact.claim,
                verdict=(
                    "insufficient_evidence" if fact.evidence_status == "missing" else "supported"
                ),
                confidence=fact.confidence,
                evidence_ids=fact.evidence_ids,
                reason=f"paper evidence status: {fact.evidence_status}",
            )
            for index, fact in enumerate(understanding.facts, start=1)
        ]
        evidence_ids = [item for fact in understanding.facts for item in fact.evidence_ids]
        confidence = (
            sum(fact.confidence for fact in understanding.facts) / len(understanding.facts)
            if understanding.facts
            else 0.0
        )
        return understanding, agent_result(
            name="paper_understanding",
            summary=understanding.paper_summary,
            confidence=confidence,
            metrics=result.metrics,
            claims=claims,
            evidence_ids=evidence_ids,
            warnings=understanding.missing_information,
        )
