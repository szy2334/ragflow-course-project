"""Paper-understanding agent adapter."""

from ..llm import StructuredLlm
from ..prompts import PromptRepository
from ..runner import AgentRunner
from ..schemas import (
    AgentResult,
    Claim,
    ConfigurationSnapshot,
    EvidenceItem,
    PaperSummary,
    PaperUnderstanding,
)
from .common import agent_result, as_json


class PaperUnderstandingAgent:
    def __init__(
        self,
        runner_or_llm: AgentRunner | StructuredLlm,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._runner = (
            runner_or_llm
            if isinstance(runner_or_llm, AgentRunner)
            else AgentRunner(runner_or_llm, prompts or PromptRepository())
        )

    async def run(
        self,
        *,
        standalone_question: str,
        evidences: list[EvidenceItem],
        configuration: ConfigurationSnapshot,
    ) -> tuple[PaperUnderstanding, AgentResult]:
        result = await self._runner.run(
            prompt_name="paper_understanding",
            prompt_version=configuration.prompt_version,
            output_model=PaperUnderstanding,
            model_config=configuration.model,
            context={
                "standalone_question": standalone_question,
                "evidence_json": as_json(
                    [item.model_dump(mode="json") for item in evidences]
                ),
            },
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

    async def run_summary(
        self,
        *,
        evidences: list[EvidenceItem],
        configuration: ConfigurationSnapshot,
    ) -> tuple[PaperSummary, AgentResult]:
        result = await self._runner.run(
            prompt_name="paper_summary",
            prompt_version=configuration.prompt_version,
            output_model=PaperSummary,
            model_config=configuration.model,
            context={
                "evidence_json": as_json(
                    [item.model_dump(mode="json") for item in evidences]
                )
            },
        )
        summary = result.output
        return summary, agent_result(
            name="paper_understanding",
            summary="generated upload-time paper summary",
            confidence=1.0,
            metrics=result.metrics,
            evidence_ids=[item.evidence_id for item in evidences],
        )
