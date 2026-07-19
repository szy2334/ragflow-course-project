"""Shared helpers for agent adapters."""

import json
from collections.abc import Iterable

from ..llm import ModelCallMetrics
from ..schemas import AgentMetrics, AgentResult, Claim


def as_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def agent_result(
    *,
    name: str,
    summary: str,
    confidence: float,
    metrics: ModelCallMetrics | None,
    claims: Iterable[Claim] = (),
    evidence_ids: Iterable[str] = (),
    warnings: Iterable[str] = (),
) -> AgentResult:
    normalized_metrics = AgentMetrics(
        latency_ms=metrics.latency_ms if metrics else 0,
        input_tokens=metrics.input_tokens if metrics else 0,
        output_tokens=metrics.output_tokens if metrics else 0,
        model=metrics.model if metrics else None,
        model_config_version=metrics.model_config_version if metrics else None,
        retry_count=metrics.retry_count if metrics else 0,
    )
    return AgentResult(
        agent_name=name,
        status="succeeded",
        summary=summary,
        claims=list(claims),
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        confidence=confidence,
        warnings=list(warnings),
        metrics=normalized_metrics,
    )

