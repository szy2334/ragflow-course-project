"""公共评审库检索结果的“无有效证据”判定。

该模块不负责调用 RAGFlow，只负责把适配器返回的候选片段转换为稳定的
业务状态，供后端的 public_retrieve/synthesize 节点使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PUBLIC_KB_NO_EVIDENCE = "PUBLIC_KB_NO_EVIDENCE"


@dataclass(frozen=True)
class PublicEvidenceDecision:
    status: str
    items: tuple[dict[str, Any], ...]
    warning_code: str | None
    message: str
    score_allowed: bool
    degraded: bool


def _metadata(chunk: Mapping[str, Any]) -> Mapping[str, Any]:
    value = chunk.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _field(chunk: Mapping[str, Any], name: str) -> Any:
    metadata = _metadata(chunk)
    value = metadata.get(name)
    return chunk.get(name) if value is None else value


def _score(chunk: Mapping[str, Any]) -> float | None:
    value = _field(chunk, "rerank_score")
    if value is None:
        value = _field(chunk, "similarity")
    if value is None:
        value = _field(chunk, "score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matches_scope(chunk: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for name, expected in filters.items():
        if expected is None:
            continue
        actual = _field(chunk, name)
        if actual != expected:
            return False
    return True


def evaluate_public_evidence(
    chunks: Sequence[Mapping[str, Any]],
    *,
    min_score: float,
    filters: Mapping[str, Any] | None = None,
) -> PublicEvidenceDecision:
    """筛选公共标准证据，并显式返回 found/no_evidence。

    缺少标准 ID、版本、正文、来源类型或相关性分数的候选片段不会被视为
    有效证据。`filters` 的字段通常包括 paper_type、rule_type、venue_code
    和 dimension。
    """

    effective: list[dict[str, Any]] = []
    requested_filters = filters or {}
    for raw in chunks:
        if not isinstance(raw, Mapping):
            continue
        source_type = _field(raw, "source_type")
        if source_type is not None and source_type not in {"standard", "public_standard"}:
            continue
        if not _field(raw, "standard_id") or not _field(raw, "standard_version"):
            continue
        text = raw.get("text") or raw.get("content") or _field(raw, "rule_text")
        if not isinstance(text, str) or not text.strip():
            continue
        if not _matches_scope(raw, requested_filters):
            continue
        score = _score(raw)
        if score is None or score < min_score:
            continue
        effective.append(dict(raw))

    effective.sort(key=lambda item: _score(item) or 0.0, reverse=True)

    if effective:
        return PublicEvidenceDecision(
            status="found",
            items=tuple(effective),
            warning_code=None,
            message="已检索到匹配的公共评审标准证据。",
            score_allowed=True,
            degraded=False,
        )

    return PublicEvidenceDecision(
        status="no_evidence",
        items=(),
        warning_code=PUBLIC_KB_NO_EVIDENCE,
        message="公共评审库未检索到与当前问题匹配的有效标准证据。",
        score_allowed=False,
        degraded=True,
    )
