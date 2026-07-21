"""Rule-level manuscript-format review backed by a server-controlled RAGFlow profile."""

# ruff: noqa: E501

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.llm import ChatMessage, OpenAICompatibleClient
from app.ai.schemas.base import StrictModel
from app.core.config import Settings
from app.db.models import FormatReview, FormatReviewItem, Paper, PaperChunk, TaskRecord
from app.runtime.adapters import task_view
from app.runtime.executor import snapshot_from_settings
from app.runtime.redis_store import RedisRuntime


class FormatReviewFailure(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True)
class FrozenFormatProfile:
    """Profile fields frozen when the user submitted the review task.

    A profile can be superseded or deactivated after a task is queued.  The
    queued task must still use exactly the rules and RAGFlow dataset selected
    at submission time so that the eventual report remains reproducible.
    """

    format_profile_id: str
    profile_key: str
    name: str
    version: str
    ragflow_dataset_id: str
    retrieval_query: str


class FormatReviewItemOutput(StrictModel):
    rule_id: str = Field(min_length=1)
    result: Literal["compliant", "non_compliant", "needs_manual_check", "not_applicable"]
    severity: Literal["info", "low", "medium", "high"]
    finding: str = Field(min_length=1, max_length=4000)
    suggestion: str | None = Field(default=None, max_length=4000)
    paper_evidence_ids: list[str] = Field(default_factory=list)
    standard_evidence_ids: list[str] = Field(default_factory=list)


class FormatReviewOutput(StrictModel):
    summary_markdown: str = Field(min_length=1, max_length=16_000)
    items: list[FormatReviewItemOutput] = Field(default_factory=list)


async def execute_format_review(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    redis: RedisRuntime,
    task_id: str,
) -> None:
    """Retrieve only the selected profile's standards and persist one result per rule."""

    async with sessions() as session:
        task, review, profile, paper, rules = await _load_review(session, task_id)
        task.stage, task.progress = "retrieving_format_rules", 0.2
        review.status = "running"
        await session.commit()
        await redis.set_task_state(task.task_id, task_view(task))

    if not settings.ragflow_base_url or not settings.ragflow_api_key:
        raise FormatReviewFailure(
            "FORMAT_KB_UNAVAILABLE", "格式规范知识库服务尚未配置。"
        )
    if not settings.llm_base_url or not settings.llm_api_key or not settings.llm_model:
        raise FormatReviewFailure("MODEL_NOT_CONFIGURED", "格式审查模型服务尚未配置。")

    standard_evidences = await _retrieve_profile_standards(settings, profile, rules)
    if not standard_evidences:
        raise FormatReviewFailure("FORMAT_KB_NO_EVIDENCE", "未从所选格式规范库检索到规则条文。")

    async with sessions() as session:
        task = await _task(session, task_id)
        task.stage, task.progress = "collecting_paper_structure", 0.45
        chunks = list(
            (
                await session.scalars(
                    select(PaperChunk)
                    .where(
                        PaperChunk.paper_id == paper.paper_id,
                        PaperChunk.paper_version_id == paper.paper_version_id,
                        PaperChunk.indexable.is_(True),
                    )
                    .order_by(PaperChunk.page_number, PaperChunk.chunk_id)
                )
            ).all()
        )
        await session.commit()
        await redis.set_task_state(task.task_id, task_view(task))

    paper_evidences = _paper_evidences(chunks)
    if not paper_evidences:
        raise FormatReviewFailure("FORMAT_REVIEW_NO_PAPER_EVIDENCE", "论文没有可用于格式审查的结构化内容。")

    output, metrics = await _review_with_model(
        settings, profile, rules, paper_evidences, standard_evidences
    )
    await _persist_result(
        sessions,
        redis,
        task_id,
        output,
        metrics,
        rules,
        paper_evidences,
        standard_evidences,
    )


async def _load_review(
    session: AsyncSession, task_id: str
) -> tuple[TaskRecord, FormatReview, FrozenFormatProfile, Paper, list[dict[str, str]]]:
    task = await _task(session, task_id)
    review = await session.get(FormatReview, task.resource_id) if task.resource_id else None
    if review is None:
        raise FormatReviewFailure("FORMAT_REVIEW_NOT_FOUND", "格式审查任务不存在。")
    paper = await session.get(Paper, review.paper_id)
    if paper is None or paper.owner_id != review.user_id or paper.status != "ready":
        raise FormatReviewFailure("PAPER_NOT_READY", "论文尚未完成解析和理解。")

    snapshot = review.profile_snapshot_json if isinstance(review.profile_snapshot_json, dict) else {}
    raw_rules = snapshot.get("rules")
    rules = [item for item in raw_rules if isinstance(item, dict)] if isinstance(raw_rules, list) else []
    profile = FrozenFormatProfile(
        format_profile_id=review.format_profile_id,
        profile_key=str(snapshot.get("profile_key") or ""),
        name=str(snapshot.get("name") or ""),
        version=str(snapshot.get("version") or ""),
        ragflow_dataset_id=str(snapshot.get("ragflow_dataset_id") or ""),
        retrieval_query=str(snapshot.get("retrieval_query") or ""),
    )
    if not all(
        (
            profile.profile_key,
            profile.name,
            profile.version,
            profile.ragflow_dataset_id,
            profile.retrieval_query,
        )
    ):
        raise FormatReviewFailure("FORMAT_PROFILE_SNAPSHOT_INVALID", "格式规范快照不完整，无法继续审查。")
    selected = set(review.selected_rule_ids)
    rules = [item for item in rules if str(item.get("rule_id")) in selected]
    if not rules:
        raise FormatReviewFailure("FORMAT_RULES_UNAVAILABLE", "所选格式规范没有可执行规则。")
    return task, review, profile, paper, [
        {"rule_id": str(item["rule_id"]), "title": str(item.get("title") or item["rule_id"]), "description": str(item.get("description") or "")}
        for item in rules
    ]


async def _retrieve_profile_standards(
    settings: Settings, profile: FrozenFormatProfile, rules: list[dict[str, str]]
) -> list[dict[str, Any]]:
    query = "\n".join(
        [profile.retrieval_query, *[f"{item['rule_id']}: {item['description']}" for item in rules]]
    )
    payload = {
        "question": query,
        "dataset_ids": [profile.ragflow_dataset_id],
        "document_ids": [],
        "top_k": min(24, max(8, len(rules) * 3)),
    }
    headers = {"Authorization": f"Bearer {settings.ragflow_api_key.get_secret_value()}"}
    url = _ragflow_endpoint(settings.ragflow_base_url, "retrieval")
    try:
        # RAGFlow is local in the desktop runtime.  Do not route localhost
        # requests through a machine-level HTTP proxy.
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise FormatReviewFailure("FORMAT_KB_UNAVAILABLE", "格式规范知识库检索失败。", retryable=True) from exc
    data = body.get("data", body) if isinstance(body, dict) else {}
    raw = data.get("chunks", data.get("items", [])) if isinstance(data, dict) else []
    evidences: list[dict[str, Any]] = []
    for index, item in enumerate(raw if isinstance(raw, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        content = str(item.get("content") or item.get("text") or "").strip()
        if not content:
            continue
        evidences.append(
            {
                "evidence_id": f"S{index}",
                "quote": content[:4000],
                "document_id": str(item.get("document_id") or metadata.get("document_id") or ""),
                "chunk_id": str(item.get("chunk_id") or item.get("id") or index),
                "section_title": metadata.get("section_title") or metadata.get("section"),
                "source_uri": metadata.get("source_uri"),
                "retrieval_score": item.get("score", item.get("similarity", 0.0)),
            }
        )
    return evidences


def _paper_evidences(chunks: list[PaperChunk]) -> list[dict[str, Any]]:
    """Sample structural evidence by section without turning a review into full-text prompting."""

    selected: list[PaperChunk] = []
    seen_sections: set[tuple[str, ...]] = set()
    for chunk in chunks:
        path = tuple(chunk.section_path_json or [])
        if path not in seen_sections:
            selected.append(chunk)
            seen_sections.add(path)
    selected_ids = {chunk.chunk_id for chunk in selected}
    selected.extend(chunk for chunk in chunks if chunk.chunk_id not in selected_ids)
    selected = selected[:32]
    return [
        {
            "evidence_id": f"P{index}",
            "quote": chunk.content[:1800],
            "chunk_id": chunk.chunk_id,
            "section_title": chunk.section_title,
            "section_path": chunk.section_path_json,
            "page_number": chunk.page_number,
            "page_end": chunk.page_end,
            "content_type": chunk.content_type,
            "content_role": chunk.content_role,
            "source_uri": f"paper://{chunk.paper_id}/{chunk.chunk_id}",
        }
        for index, chunk in enumerate(selected, start=1)
    ]


async def _review_with_model(
    settings: Settings,
    profile: FrozenFormatProfile,
    rules: list[dict[str, str]],
    paper_evidences: list[dict[str, Any]],
    standard_evidences: list[dict[str, Any]],
) -> tuple[FormatReviewOutput, dict[str, Any]]:
    messages = [
        ChatMessage(
            role="system",
            content=(
                "You review manuscript FORMAT compliance only. Do not evaluate research quality, "
                "novelty, experiments, statistics, or conclusions. Assess only the supplied rule IDs. "
                "Use only P* manuscript evidence and S* standard evidence. When a rule needs visual "
                "properties unavailable in the evidence, return needs_manual_check rather than guessing."
            ),
        ),
        ChatMessage(
            role="user",
            content=json.dumps(
                {
                    "format_profile": {
                        "name": profile.name,
                        "version": profile.version,
                        "rules": rules,
                    },
                    "paper_evidence": paper_evidences,
                    "standard_evidence": standard_evidences,
                    "required_output": "one item for every requested rule_id",
                },
                ensure_ascii=False,
            ),
        ),
    ]
    result = await OpenAICompatibleClient(settings.llm_api_key).invoke_structured(
        messages,
        FormatReviewOutput,
        snapshot_from_settings(settings).model,
    )
    return result.output, asdict(result.metrics)


async def _persist_result(
    sessions: async_sessionmaker[AsyncSession],
    redis: RedisRuntime,
    task_id: str,
    output: FormatReviewOutput,
    metrics: dict[str, Any],
    rules: list[dict[str, str]],
    paper_evidences: list[dict[str, Any]],
    standard_evidences: list[dict[str, Any]],
) -> None:
    paper_by_id = {item["evidence_id"]: item for item in paper_evidences}
    standard_by_id = {item["evidence_id"]: item for item in standard_evidences}
    rule_by_id = {item["rule_id"]: item for item in rules}
    output_by_rule = {item.rule_id: item for item in output.items if item.rule_id in rule_by_id}
    async with sessions() as session:
        task = await _task(session, task_id)
        review = await session.get(FormatReview, task.resource_id) if task.resource_id else None
        if review is None:
            raise FormatReviewFailure("FORMAT_REVIEW_NOT_FOUND", "格式审查任务不存在。")
        await session.execute(
            delete(FormatReviewItem).where(FormatReviewItem.format_review_id == review.format_review_id)
        )
        items: list[FormatReviewItem] = []
        for rule_id, rule in rule_by_id.items():
            output_item = output_by_rule.get(rule_id)
            if output_item is None:
                output_item = FormatReviewItemOutput(
                    rule_id=rule_id,
                    result="needs_manual_check",
                    severity="info",
                    finding="模型未返回该规则的可核验判定。",
                    suggestion="请人工核对该规则。",
                )
            paper_refs = [paper_by_id[item] for item in output_item.paper_evidence_ids if item in paper_by_id]
            standard_refs = [
                standard_by_id[item] for item in output_item.standard_evidence_ids if item in standard_by_id
            ]
            pages = sorted(
                {int(item["page_number"]) for item in paper_refs if item.get("page_number") is not None}
            )
            items.append(
                FormatReviewItem(
                    format_review_id=review.format_review_id,
                    rule_id=rule_id,
                    rule_title=rule["title"],
                    result=output_item.result,
                    severity=output_item.severity,
                    finding=output_item.finding,
                    suggestion=output_item.suggestion,
                    page_numbers=pages,
                    paper_evidence_json=paper_refs,
                    standard_evidence_json=standard_refs,
                )
            )
        session.add_all(items)
        now = datetime.now(UTC)
        review.status, review.summary_markdown, review.metrics_json, review.completed_at = (
            "succeeded",
            output.summary_markdown,
            metrics,
            now,
        )
        review.error_json = None
        task.status, task.stage, task.progress, task.completed_at = "succeeded", "completed", 1.0, now
        task.result_json = {"format_review_id": review.format_review_id, "item_count": len(items)}
        await session.commit()
        await redis.set_task_state(task.task_id, task_view(task))


async def _task(session: AsyncSession, task_id: str) -> TaskRecord:
    task = await session.get(TaskRecord, task_id)
    if task is None:
        raise FormatReviewFailure("TASK_NOT_FOUND", "格式审查任务不存在。")
    return task


def _ragflow_endpoint(base_url: str, resource: str) -> str:
    """Accept either a RAGFlow host root or an already versioned API root."""

    root = base_url.rstrip("/")
    if not root.endswith("/api/v1"):
        root = f"{root}/api/v1"
    return f"{root}/{resource.lstrip('/')}"
