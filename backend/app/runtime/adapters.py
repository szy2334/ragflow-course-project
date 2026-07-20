"""Concrete adapters that connect the pure AI graph to runtime infrastructure."""

import re
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.ports import AnswerPersistencePort, CancellationPort, ContextPort, EventSink, TraceSink
from app.ai.schemas import (
    EvidenceItem,
    EvidenceSet,
    PersistAnswerCommand,
    RetrieveEvidenceRequest,
    RetrieveStandardsRequest,
    StreamEvent,
)
from app.core.config import Settings
from app.db.models import (
    ChatMessage,
    Citation,
    Paper,
    PaperChunk,
    ReviewResult,
    TaskRecord,
    TraceRecord,
    WorkflowRun,
)

from .redis_store import RedisRuntime


class SqlAlchemyContextPort(ContextPort):
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load_session_summary(self, *, user_id: str, session_id: str) -> str:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(8)
            )
            messages = list(reversed(rows.all()))
        turns: list[str] = []
        for item in messages:
            turns.append(f"user: {item.content}")
            if item.answer_text:
                turns.append(f"assistant: {item.answer_text}")
        summary = "\n".join(turns)
        return summary[-6000:]


class RedisCancellationPort(CancellationPort):
    def __init__(self, redis: RedisRuntime) -> None:
        self._redis = redis

    async def is_cancelled(self, task_id: str) -> bool:
        return await self._redis.is_cancelled(task_id)


class RedisEventSink(EventSink):
    def __init__(self, redis: RedisRuntime, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._redis = redis
        self._sessions = sessions

    async def emit(self, event: StreamEvent) -> None:
        await self._redis.append_event(event)
        if event.event_type != "status":
            return
        async with self._sessions() as session:
            task = await session.get(TaskRecord, event.task_id)
            if task is None:
                return
            task.stage = str(event.data.get("stage", task.stage))
            task.progress = _stage_progress(task.stage)
            await session.commit()
            await self._redis.set_task_state(event.task_id, task_view(task))


class SqlAlchemyTraceSink(TraceSink):
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record(self, trace) -> None:
        async with self._sessions() as session:
            session.add(
                TraceRecord(
                    task_id=trace.task_id,
                    message_id=trace.message_id,
                    request_id=trace.request_id,
                    correlation_id=trace.correlation_id,
                    node_name=trace.node_name,
                    duration_ms=trace.duration_ms,
                    status=trace.status,
                    error_code=trace.error_code,
                    metrics_json=trace.metrics,
                )
            )
            await session.commit()


class SqlAlchemyAnswerPersistence(AnswerPersistencePort):
    """One transaction persists a validated answer before terminal SSE events."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession], redis: RedisRuntime) -> None:
        self._sessions = sessions
        self._redis = redis

    async def persist(self, command: PersistAnswerCommand) -> None:
        async with self._sessions() as session:
            message = await session.get(ChatMessage, command.message_id)
            task = await session.get(TaskRecord, command.task_id)
            if message is None or task is None:
                raise RuntimeError("workflow resources disappeared")
            if message.user_id != command.user_id or message.session_id != command.session_id:
                raise RuntimeError("workflow ownership mismatch")
            if message.answer_json is not None and task.status == "succeeded":
                return

            answer_json = command.answer.model_dump(mode="json")
            message.answer_json = answer_json
            message.answer_text = command.answer.answer
            message.status = "succeeded"
            message.confidence = command.answer.confidence
            message.route_type = command.answer.route_type
            message.model_version = command.configuration.model.config_version
            message.prompt_version = command.configuration.prompt_version
            message.retrieval_config_json = {
                "schema_version": command.configuration.schema_version,
                "standard_version": command.configuration.standard_version,
            }
            task.status = "succeeded"
            task.progress = 1.0
            task.stage = "completed"
            task.result_json = {"message_id": command.message_id}
            task.completed_at = datetime.now(UTC)

            run = await session.scalar(
                select(WorkflowRun).where(WorkflowRun.task_id == task.task_id)
            )
            if run is not None:
                run.status = "succeeded"
                run.completed_at = task.completed_at
                run.summary_json = {
                    "configuration": command.configuration.model_dump(mode="json"),
                    "agent_results": [
                        item.model_dump(mode="json") for item in command.agent_results
                    ],
                }

            for evidence in command.answer.evidences:
                session.add(
                    Citation(
                        message_id=message.message_id,
                        evidence_id=evidence.evidence_id,
                        source_type=evidence.source_type,
                        paper_id=evidence.paper_id,
                        document_id=evidence.document_id,
                        chunk_id=evidence.chunk_id,
                        content_type=evidence.content_type,
                        source_text=evidence.quote,
                        section_title=evidence.section_title,
                        page_start=evidence.page_number,
                        page_end=int(evidence.metadata.get("page_end") or evidence.page_number)
                        if evidence.page_number is not None
                        else None,
                        similarity=evidence.retrieval_score,
                        source_uri=evidence.source_uri,
                        content_sha256=evidence.metadata.get("content_sha256"),
                        evidence_json=evidence.model_dump(mode="json"),
                    )
                )
            for opinion in command.answer.review_opinions:
                session.add(
                    ReviewResult(
                        message_id=message.message_id,
                        reviewer=opinion.reviewer,
                        position=opinion.position,
                        confidence=opinion.confidence,
                        summary=opinion.summary,
                        opinion_json=opinion.model_dump(mode="json"),
                    )
                )
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))


class RagFlowRetrievalPort:
    """Local user-paper evidence plus on-demand fixed RAGFlow reference retrieval."""

    def __init__(self, settings: Settings, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._settings = settings
        self._sessions = sessions

    async def retrieve_paper(self, request: RetrieveEvidenceRequest) -> EvidenceSet:
        async with self._sessions() as session:
            papers = list(
                (
                    await session.scalars(
                        select(Paper).where(
                            Paper.paper_id.in_(request.paper_ids),
                            Paper.owner_id == request.user_id,
                            Paper.status == "ready",
                            Paper.deleted_at.is_(None),
                        )
                    )
                ).all()
            )
            if {item.paper_id for item in papers} != set(request.paper_ids):
                return EvidenceSet(
                    items=[],
                    query=request.standalone_question,
                    relaxed=request.relaxed,
                    warnings=["authorized paper scope is unavailable"],
                )
            current_versions = [
                item.paper_version_id for item in papers if item.paper_version_id is not None
            ]
            chunks = list(
                (
                    await session.scalars(
                        select(PaperChunk)
                        .where(
                            PaperChunk.paper_id.in_(request.paper_ids),
                            PaperChunk.paper_version_id.in_(current_versions),
                            PaperChunk.indexable.is_(True),
                        )
                        .order_by(PaperChunk.paper_id, PaperChunk.page_number)
                    )
                ).all()
            )
        if request.content_preferences:
            preferred = [
                item for item in chunks if item.content_type in request.content_preferences
            ]
            chunks = preferred or chunks
        ranked = sorted(
            chunks,
            key=lambda item: _local_relevance(item.content, request.standalone_question),
            reverse=True,
        )
        limit = 12 if request.relaxed else 6
        items = [
            EvidenceItem(
                evidence_id=f"P{index}",
                source_type="paper",
                paper_id=item.paper_id,
                document_id=f"local:{item.paper_id}",
                chunk_id=item.chunk_id,
                content_type=item.content_type,
                quote=item.content,
                section_title=item.section_title or None,
                page_number=item.page_number,
                source_uri=f"paper://{item.paper_id}/{item.chunk_id}",
                retrieval_score=_local_relevance(item.content, request.standalone_question),
                content_role=item.content_role,
                object_id=item.object_id,
                parent_chunk_id=item.parent_chunk_id,
                metadata=item.metadata_json,
            )
            for index, item in enumerate(ranked[:limit], start=1)
        ]
        return EvidenceSet(items=items, query=request.standalone_question, relaxed=request.relaxed)

    async def retrieve_standards(self, request: RetrieveStandardsRequest) -> EvidenceSet:
        dataset_id = self._settings.ragflow_reference_dataset
        if not dataset_id:
            return EvidenceSet(
                items=[],
                query=request.standalone_question,
                warnings=["reference paper knowledge base is not configured"],
            )
        raw = await self._retrieve(request.standalone_question, [dataset_id], [], False)
        items = self._normalize(raw, source_type="standard", paper_by_document={})
        for item in items:
            item.metadata.update(
                {
                    "knowledge_base": "user_paper",
                    "evidence_role": "reference_paper",
                    "name": item.metadata.get("name") or item.section_title or "参考论文",
                }
            )
        return EvidenceSet(
            items=items,
            query=request.standalone_question,
        )

    async def _retrieve(
        self, question: str, dataset_ids: list[str], document_ids: list[str], relaxed: bool
    ) -> list[dict[str, Any]]:
        if not self._settings.ragflow_base_url or not self._settings.ragflow_api_key:
            return []
        payload: dict[str, Any] = {
            "question": question,
            "dataset_ids": dataset_ids,
            "document_ids": document_ids,
            "top_k": 12 if relaxed else 6,
        }
        headers = {"Authorization": f"Bearer {self._settings.ragflow_api_key.get_secret_value()}"}
        url = self._settings.ragflow_base_url.rstrip("/") + "/api/v1/retrieval"
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError):
            return []
        data = body.get("data", body) if isinstance(body, dict) else {}
        candidates = data.get("chunks", data.get("items", [])) if isinstance(data, dict) else []
        return [item for item in candidates if isinstance(item, dict)]

    @staticmethod
    def _normalize(
        chunks: list[dict[str, Any]], *, source_type: str, paper_by_document: dict[str, str]
    ) -> list[EvidenceItem]:
        normalized: list[EvidenceItem] = []
        for position, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            document_id = str(chunk.get("document_id") or metadata.get("document_id") or "")
            content = str(chunk.get("content") or chunk.get("text") or "").strip()
            if not document_id or not content:
                continue
            content_type = str(metadata.get("content_type") or "text")
            if content_type not in {
                "text",
                "figure",
                "figure_caption",
                "table",
                "formula",
                "metadata",
                "reference",
            }:
                content_type = "text"
            score = chunk.get("score", chunk.get("similarity", 0.0))
            try:
                score = max(float(score), 0.0)
            except (TypeError, ValueError):
                score = 0.0
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or f"{document_id}:{position}")
            paper_id = paper_by_document.get(document_id)
            if source_type == "paper" and not paper_id:
                continue
            normalized.append(
                EvidenceItem(
                    evidence_id=("P" if source_type == "paper" else "S") + str(position),
                    source_type=source_type,
                    paper_id=paper_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    content_type=content_type,
                    quote=content,
                    section_title=metadata.get("section_title") or metadata.get("section"),
                    page_number=_positive_int(metadata.get("page_number") or metadata.get("page")),
                    source_uri=str(
                        metadata.get("source_uri") or f"ragflow://{document_id}/{chunk_id}"
                    ),
                    retrieval_score=score,
                    content_role=metadata.get("content_role"),
                    object_id=metadata.get("object_id"),
                    parent_chunk_id=metadata.get("parent_chunk_id"),
                    metadata=metadata,
                )
            )
        return normalized


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
        return number if number >= 1 else None
    except (TypeError, ValueError):
        return None


def _local_relevance(content: str, question: str) -> float:
    latin_terms = set(re.findall(r"[a-z0-9_]{2,}", question.lower()))
    chinese_terms = set(re.findall(r"[\u4e00-\u9fff]", question))
    terms = latin_terms | chinese_terms
    if not terms:
        return 0.01
    normalized = content.lower()
    matches = sum(term in normalized for term in terms)
    return matches / len(terms) if matches else 0.01


def _stage_progress(stage: str) -> float:
    stages = {
        "queued": 0.0,
        "loading_context": 0.05,
        "routing": 0.12,
        "retrieving_paper": 0.28,
        "understanding": 0.48,
        "retrieving_references": 0.58,
        "review_a": 0.7,
        "review_b": 0.8,
        "synthesizing": 0.9,
        "finalizing": 0.96,
        "completed": 1.0,
    }
    return stages.get(stage, 0.0)


def task_view(task: TaskRecord) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": task.status,
        "progress": task.progress,
        "stage": task.stage,
        "resource_id": task.resource_id,
        "error": task.error_json,
        "result": task.result_json,
        "created_at": task.created_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }
