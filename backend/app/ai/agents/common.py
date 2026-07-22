"""Shared helpers for agent adapters."""

import json
from collections.abc import Iterable

from ..llm import ModelCallMetrics
from ..schemas import AgentMetrics, AgentResult, Claim


class JsonStringFieldExtractor:
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
