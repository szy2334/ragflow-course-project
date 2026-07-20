"""PDF ingestion: parse -> OCR -> chunks -> quality -> AI understanding."""

# ruff: noqa: E501

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.agents import PaperUnderstandingAgent
from app.ai.llm import OpenAICompatibleClient
from app.ai.prompts import PromptRepository
from app.ai.schemas import EvidenceItem
from app.core.config import Settings
from app.db.models import (
    IngestionQualityReport,
    MediaObjectRecord,
    Paper,
    PaperChunk,
    PaperIngestionRun,
    ParsedBlockRecord,
    RagMapping,
    TaskRecord,
)
from app.runtime.adapters import task_view
from app.runtime.executor import snapshot_from_settings
from app.runtime.redis_store import RedisRuntime


class IngestionFailure(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    block_id: str
    content: str
    page_number: int
    section_title: str
    content_type: str = "text"
    source_ref: str = ""


@dataclass(frozen=True, slots=True)
class MediaObject:
    object_id: str
    kind: str
    page_number: int
    source_ref: str
    image_url: str | None
    caption: str | None
    required: bool = True


@dataclass(frozen=True, slots=True)
class ParsedPaper:
    blocks: list[ParsedBlock]
    media: list[MediaObject]


@dataclass(frozen=True, slots=True)
class BuiltChunk:
    chunk_id: str
    content: str
    content_type: str
    section_title: str
    page_number: int
    source_ref: str
    content_sha256: str
    object_id: str | None = None
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] | None = None


class MinerUClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def parse(self, file_path: Path) -> ParsedPaper:
        if not self._settings.mineru_base_url:
            raise IngestionFailure("MINERU_UNAVAILABLE", "论文解析服务尚未配置。")
        headers = _bearer(self._settings.mineru_api_key)
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                with file_path.open("rb") as file_handle:
                    response = await client.post(
                        self._settings.mineru_base_url.rstrip("/") + "/api/v1/parse",
                        headers=headers,
                        files={"file": (file_path.name, file_handle, "application/pdf")},
                    )
                response.raise_for_status()
                body = response.json()
        except (OSError, httpx.HTTPError, ValueError) as exc:
            raise IngestionFailure("MINERU_PARSE_FAILED", "论文解析失败，请稍后重试。") from exc
        data = body.get("data", body) if isinstance(body, dict) else {}
        raw_blocks = data.get("blocks", []) if isinstance(data, dict) else []
        raw_media = data.get("media", data.get("objects", [])) if isinstance(data, dict) else []
        blocks = [
            _block_from_raw(item, index)
            for index, item in enumerate(raw_blocks, start=1)
            if isinstance(item, dict)
        ]
        media = [
            _media_from_raw(item, index)
            for index, item in enumerate(raw_media, start=1)
            if isinstance(item, dict)
        ]
        if not blocks:
            raise IngestionFailure("MINERU_PARSE_FAILED", "论文解析结果为空。")
        return ParsedPaper(blocks=blocks, media=media)


class BaiduSpecializedOcrClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def recognize(self, item: MediaObject) -> str:
        if not item.required:
            return ""
        if not self._settings.baidu_ocr_base_url or not self._settings.baidu_ocr_api_key:
            raise IngestionFailure("BAIDU_OCR_SPECIALIZED_FAILED", "专项 OCR 服务尚未配置。")
        endpoint = "/table-recognition-v2" if item.kind == "table" else "/paddleocr-vl"
        payload = {
            "image_url": item.image_url,
            "object_id": item.object_id,
            "page_number": item.page_number,
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self._settings.baidu_ocr_base_url.rstrip("/") + endpoint,
                    headers=_bearer(self._settings.baidu_ocr_api_key),
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise IngestionFailure("BAIDU_OCR_SPECIALIZED_FAILED", "专项 OCR 识别失败。") from exc
        data = body.get("data", body) if isinstance(body, dict) else {}
        text = _ocr_text(data)
        if not text:
            raise IngestionFailure("BAIDU_OCR_SPECIALIZED_FAILED", "专项 OCR 未返回可用内容。")
        return text


class RagFlowManualImporter:
    """Imports already structured chunks and requires a complete mapping response."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def import_chunks(
        self, paper: Paper, chunks: list[BuiltChunk]
    ) -> tuple[str, dict[str, str]]:
        if (
            not self._settings.ragflow_base_url
            or not self._settings.ragflow_api_key
            or not self._settings.ragflow_user_dataset_id
        ):
            raise IngestionFailure("RAGFLOW_IMPORT_FAILED", "论文知识库服务尚未配置。")
        payload = {
            "name": f"paper-{paper.paper_id}",
            "chunk_method": "manual",
            "metadata": {
                "paper_id": paper.paper_id,
                "paper_version_id": paper.paper_version_id,
                "user_id": paper.owner_id,
                "quality_status": "ready",
                "file_hash": paper.content_sha256,
            },
            "chunks": [
                {
                    "id": item.chunk_id,
                    "content": item.content,
                    "metadata": {
                        "paper_id": paper.paper_id,
                        "paper_version_id": paper.paper_version_id,
                        "source_chunk_id": item.chunk_id,
                        "content_type": item.content_type,
                        "section_title": item.section_title,
                        "page_number": item.page_number,
                        "source_ref": item.source_ref,
                        "object_id": item.object_id,
                        "parent_chunk_id": item.parent_chunk_id,
                        "content_sha256": item.content_sha256,
                        **(item.metadata or {}),
                    },
                }
                for item in chunks
            ],
        }
        url = (
            self._settings.ragflow_base_url.rstrip("/")
            + f"/api/v1/datasets/{self._settings.ragflow_user_dataset_id}/documents"
        )
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    url, headers=_bearer(self._settings.ragflow_api_key), json=payload
                )
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise IngestionFailure("RAGFLOW_IMPORT_FAILED", "论文知识库导入失败。") from exc
        data = body.get("data", body) if isinstance(body, dict) else {}
        document_id = (
            str(data.get("document_id") or data.get("id") or "") if isinstance(data, dict) else ""
        )
        raw_mappings = data.get("chunk_mappings", []) if isinstance(data, dict) else []
        mapping: dict[str, str] = {}
        for item in raw_mappings:
            if (
                isinstance(item, dict)
                and item.get("source_chunk_id")
                and item.get("ragflow_chunk_id")
            ):
                mapping[str(item["source_chunk_id"])] = str(item["ragflow_chunk_id"])
        if not document_id or set(mapping) != {chunk.chunk_id for chunk in chunks}:
            raise IngestionFailure("RAGFLOW_IMPORT_INCOMPLETE", "论文知识库映射不完整。")
        return document_id, mapping


class IngestionTaskExecutor:
    """Persists every stage artifact so retries restart at the requested boundary."""

    def __init__(
        self, settings: Settings, sessions: async_sessionmaker[AsyncSession], redis: RedisRuntime
    ) -> None:
        self._settings = settings
        self._sessions = sessions
        self._redis = redis
        self._running: set[asyncio.Task[None]] = set()

    def submit(self, task_id: str, *, start_stage: str | None = None) -> None:
        task = asyncio.create_task(
            self.run(task_id, start_stage=start_stage), name=f"ingestion:{task_id}"
        )
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def run(self, task_id: str, *, start_stage: str | None = None) -> None:
        try:
            await self._run(task_id, start_stage=start_stage)
        except IngestionFailure as exc:
            await self._fail(task_id, exc.code, exc.message, exc.retryable)
        except Exception:
            await self._fail(task_id, "PAPER_INGEST_FAILED", "论文入库失败，请稍后重试。", True)

    async def _run(self, task_id: str, *, start_stage: str | None) -> None:
        async with self._sessions() as session:
            task = await session.get(TaskRecord, task_id)
            if task is None or not task.resource_id:
                return
            paper = await session.get(Paper, task.resource_id)
            if paper is None:
                return
            if paper.paper_version_id is None:
                raise IngestionFailure(
                    "PAPER_VERSION_MISSING",
                    "论文缺少可处理的版本记录。",
                    retryable=False,
                )
            stage = _ingestion_stage(start_stage or task.stage)
            task.status, task.stage, task.started_at = "running", stage, datetime.now(UTC)
            paper.status, paper.parse_progress, paper.index_status, paper.failure, paper.understanding_json = (
                stage,
                _ingestion_progress(stage),
                "pending",
                None,
                None,
            )
            run = await session.scalar(
                select(PaperIngestionRun).where(PaperIngestionRun.task_id == task.task_id)
            )
            if run is None:
                run = PaperIngestionRun(
                    task_id=task.task_id,
                    paper_id=paper.paper_id,
                    paper_version_id=paper.paper_version_id,
                    stage=stage,
                    status="running",
                    started_at=task.started_at,
                )
                session.add(run)
            else:
                run.stage, run.status, run.started_at, run.error_json = stage, "running", task.started_at, None
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))
            paper_id, version_id, file_path = paper.paper_id, paper.paper_version_id, Path(paper.file_path)

        if stage == "mineru_parsing" or not await self._has_parse_artifacts(paper_id, version_id):
            parsed = await MinerUClient(self._settings).parse(file_path)
            await self._store_parse_artifacts(task_id, paper_id, version_id, parsed)
        parsed = await self._load_parsed_artifacts(paper_id, version_id)

        await self._set_stage(task_id, paper_id, "ocr_processing")
        ocr_results = await self._run_ocr(task_id, paper_id, version_id, parsed.media)
        parsed = ParsedPaper(blocks=parsed.blocks, media=parsed.media)

        await self._set_stage(task_id, paper_id, "cleaning")
        chunks = _build_chunks(paper_id, parsed, ocr_results)
        await self._store_chunks(paper_id, version_id, chunks)

        await self._set_stage(task_id, paper_id, "quality_check")
        errors = _quality_report(parsed, chunks, ocr_results)
        await self._store_quality_report(task_id, paper_id, version_id, errors, len(chunks))
        if errors:
            raise IngestionFailure(
                "CHUNK_QUALITY_FAILED", "论文结构化质量检查未通过。", retryable=False
            )

        await self._set_stage(task_id, paper_id, "understanding")
        understanding = await self._understand(paper_id, chunks)
        await self._store_understanding(paper_id, understanding)

        await self._complete(task_id, paper_id, len(chunks))

    async def _understand(self, paper_id: str, chunks: list[BuiltChunk]) -> dict[str, Any]:
        if not self._settings.llm_base_url or not self._settings.llm_api_key or not self._settings.llm_model:
            raise IngestionFailure("MODEL_NOT_CONFIGURED", "模型服务尚未配置。", retryable=False)
        evidences = _understanding_evidences(paper_id, chunks)
        if not evidences:
            raise IngestionFailure("PAPER_UNDERSTANDING_FAILED", "论文缺少可供理解的正文内容。")
        try:
            understanding, _ = await PaperUnderstandingAgent(
                OpenAICompatibleClient(self._settings.llm_api_key), PromptRepository()
            ).run(
                standalone_question="请概括这篇论文的研究问题、方法、实验设置、主要发现和局限。",
                evidences=evidences,
                configuration=snapshot_from_settings(self._settings),
            )
        except Exception as exc:
            raise IngestionFailure("PAPER_UNDERSTANDING_FAILED", "论文智能理解失败，请稍后重试。") from exc
        evidence_ids = {item.evidence_id for item in evidences}
        cited_ids = {item for fact in understanding.facts for item in fact.evidence_ids}
        if not cited_ids.issubset(evidence_ids):
            raise IngestionFailure("PAPER_UNDERSTANDING_FAILED", "论文理解结果包含无效证据引用。")
        return understanding.model_dump(mode="json")

    async def _store_understanding(self, paper_id: str, understanding: dict[str, Any]) -> None:
        async with self._sessions() as session:
            paper = await session.get(Paper, paper_id)
            if paper is None:
                raise IngestionFailure("PAPER_NOT_FOUND", "论文不存在。", retryable=False)
            paper.understanding_json = understanding
            await session.commit()

    async def _has_parse_artifacts(self, paper_id: str, version_id: str) -> bool:
        async with self._sessions() as session:
            item = await session.scalar(
                select(ParsedBlockRecord.parsed_block_id).where(
                    ParsedBlockRecord.paper_id == paper_id,
                    ParsedBlockRecord.paper_version_id == version_id,
                )
            )
            return item is not None

    async def _store_parse_artifacts(
        self, task_id: str, paper_id: str, version_id: str, parsed: ParsedPaper
    ) -> None:
        async with self._sessions() as session:
            await session.execute(
                delete(ParsedBlockRecord).where(
                    ParsedBlockRecord.paper_id == paper_id,
                    ParsedBlockRecord.paper_version_id == version_id,
                )
            )
            await session.execute(
                delete(MediaObjectRecord).where(
                    MediaObjectRecord.paper_id == paper_id,
                    MediaObjectRecord.paper_version_id == version_id,
                )
            )
            session.add_all(
                [
                    ParsedBlockRecord(
                        paper_id=paper_id,
                        paper_version_id=version_id,
                        block_id=block.block_id,
                        content=block.content,
                        content_type=block.content_type,
                        section_title=block.section_title,
                        page_number=block.page_number,
                        source_ref=block.source_ref,
                    )
                    for block in parsed.blocks
                ]
            )
            session.add_all(
                [
                    MediaObjectRecord(
                        paper_id=paper_id,
                        paper_version_id=version_id,
                        object_id=item.object_id,
                        object_type=item.kind,
                        page_number=item.page_number,
                        source_ref=item.source_ref,
                        image_url=item.image_url,
                        image_sha256=_image_hash(item.image_url),
                        caption=item.caption,
                        required=item.required,
                    )
                    for item in parsed.media
                ]
            )
            run = await session.scalar(
                select(PaperIngestionRun).where(PaperIngestionRun.task_id == task_id)
            )
            if run is not None:
                run.stage = "ocr_processing"
            await session.commit()

    async def _load_parsed_artifacts(self, paper_id: str, version_id: str) -> ParsedPaper:
        async with self._sessions() as session:
            blocks = list(
                (
                    await session.scalars(
                        select(ParsedBlockRecord)
                        .where(
                            ParsedBlockRecord.paper_id == paper_id,
                            ParsedBlockRecord.paper_version_id == version_id,
                        )
                        .order_by(ParsedBlockRecord.page_number, ParsedBlockRecord.block_id)
                    )
                ).all()
            )
            media = list(
                (
                    await session.scalars(
                        select(MediaObjectRecord)
                        .where(
                            MediaObjectRecord.paper_id == paper_id,
                            MediaObjectRecord.paper_version_id == version_id,
                        )
                        .order_by(MediaObjectRecord.page_number, MediaObjectRecord.object_id)
                    )
                ).all()
            )
        if not blocks:
            raise IngestionFailure("MINERU_PARSE_FAILED", "论文解析产物缺失。")
        return ParsedPaper(
            blocks=[
                ParsedBlock(
                    item.block_id,
                    item.content,
                    item.page_number,
                    item.section_title,
                    item.content_type,
                    item.source_ref,
                )
                for item in blocks
            ],
            media=[
                MediaObject(
                    item.object_id,
                    item.object_type,
                    item.page_number,
                    item.source_ref,
                    item.image_url,
                    item.caption,
                    item.required,
                )
                for item in media
            ],
        )

    async def _run_ocr(
        self, task_id: str, paper_id: str, version_id: str, media: list[MediaObject]
    ) -> dict[str, str]:
        async with self._sessions() as session:
            rows = {
                item.object_id: item
                for item in (
                    await session.scalars(
                        select(MediaObjectRecord).where(
                            MediaObjectRecord.paper_id == paper_id,
                            MediaObjectRecord.paper_version_id == version_id,
                        )
                    )
                ).all()
            }
        client = BaiduSpecializedOcrClient(self._settings)
        result: dict[str, str] = {}
        for item in media:
            row = rows[item.object_id]
            if not item.required:
                result[item.object_id] = row.ocr_text or ""
                continue
            if row.ocr_status != "success":
                try:
                    text = await client.recognize(item)
                except IngestionFailure:
                    await self._record_ocr_failure(paper_id, version_id, item.object_id)
                    raise
                async with self._sessions() as session:
                    current = await session.scalar(
                        select(MediaObjectRecord).where(
                            MediaObjectRecord.paper_id == paper_id,
                            MediaObjectRecord.paper_version_id == version_id,
                            MediaObjectRecord.object_id == item.object_id,
                        )
                    )
                    assert current is not None
                    current.ocr_status, current.ocr_text = "success", text
                    current.ocr_engine = (
                        "baidu-table-v2" if item.kind == "table" else "baidu-paddleocr-vl"
                    )
                    current.engines_json = [current.ocr_engine]
                    current.failure_json = None
                    await session.commit()
                row.ocr_status, row.ocr_text = "success", text
            result[item.object_id] = row.ocr_text or ""
        return result

    async def _record_ocr_failure(self, paper_id: str, version_id: str, object_id: str) -> None:
        async with self._sessions() as session:
            item = await session.scalar(
                select(MediaObjectRecord).where(
                    MediaObjectRecord.paper_id == paper_id,
                    MediaObjectRecord.paper_version_id == version_id,
                    MediaObjectRecord.object_id == object_id,
                )
            )
            if item is not None:
                item.ocr_status = "failed"
                item.retry_count += 1
                item.failure_json = {"code": "BAIDU_OCR_SPECIALIZED_FAILED"}
                await session.commit()

    async def _store_chunks(
        self, paper_id: str, version_id: str, chunks: list[BuiltChunk]
    ) -> None:
        async with self._sessions() as session:
            await session.execute(
                delete(PaperChunk).where(
                    PaperChunk.paper_id == paper_id,
                    PaperChunk.paper_version_id == version_id,
                )
            )
            session.add_all([_chunk_row(paper_id, version_id, item) for item in chunks])
            await session.commit()

    async def _store_quality_report(
        self,
        task_id: str,
        paper_id: str,
        version_id: str,
        errors: list[str],
        indexable_chunk_count: int,
    ) -> None:
        async with self._sessions() as session:
            item = await session.scalar(
                select(IngestionQualityReport).where(IngestionQualityReport.task_id == task_id)
            )
            status = "failed" if errors else "ready"
            payload = {
                "status": status,
                "blocking_errors": errors,
                "indexable_chunks": indexable_chunk_count,
            }
            if item is None:
                session.add(
                    IngestionQualityReport(
                        task_id=task_id,
                        paper_id=paper_id,
                        paper_version_id=version_id,
                        status=status,
                        indexable_chunk_count=indexable_chunk_count,
                        blocking_error_count=len(errors),
                        expected_mapping_count=indexable_chunk_count,
                        report_json=payload,
                    )
                )
            else:
                item.status = status
                item.indexable_chunk_count = indexable_chunk_count
                item.blocking_error_count = len(errors)
                item.expected_mapping_count = indexable_chunk_count
                item.report_json = payload
            run = await session.scalar(
                select(PaperIngestionRun).where(PaperIngestionRun.task_id == task_id)
            )
            if run is not None:
                run.stage, run.quality_status = "quality_check", status
            await session.commit()

    async def _set_stage(self, task_id: str, paper_id: str, stage: str) -> None:
        async with self._sessions() as session:
            task, paper = await _task_and_paper(session, task_id, paper_id)
            task.stage, task.progress = stage, _ingestion_progress(stage)
            paper.status, paper.parse_progress = stage, task.progress
            if stage == "indexing":
                paper.index_status = "running"
            run = await session.scalar(
                select(PaperIngestionRun).where(PaperIngestionRun.task_id == task_id)
            )
            if run is not None:
                run.stage = stage
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))

    async def _store_mappings(
        self,
        paper_id: str,
        version_id: str,
        document_id: str,
        mappings: dict[str, str],
        chunks: list[BuiltChunk],
    ) -> None:
        async with self._sessions() as session:
            await session.execute(
                delete(RagMapping).where(
                    RagMapping.paper_id == paper_id,
                    RagMapping.paper_version_id == version_id,
                )
            )
            session.add_all(
                [
                    RagMapping(
                        paper_id=paper_id,
                        paper_version_id=version_id,
                        source_chunk_id=chunk.chunk_id,
                        dataset_id=self._settings.ragflow_user_dataset_id or "",
                        document_id=document_id,
                        ragflow_chunk_id=mappings[chunk.chunk_id],
                        content_sha256=chunk.content_sha256,
                        status="ready",
                    )
                    for chunk in chunks
                ]
            )
            await session.commit()

    async def _complete(
        self,
        task_id: str,
        paper_id: str,
        count: int,
    ) -> None:
        async with self._sessions() as session:
            task, paper = await _task_and_paper(session, task_id, paper_id)
            task.status, task.stage, task.progress, task.completed_at = (
                "succeeded",
                "completed",
                1.0,
                datetime.now(UTC),
            )
            task.result_json = {
                "paper_id": paper_id,
                "local_chunks": count,
                "understanding": paper.understanding_json,
            }
            paper.status, paper.parse_progress, paper.index_status, paper.quality_status = (
                "ready",
                1.0,
                "not_indexed",
                "ready",
            )
            run = await session.scalar(
                select(PaperIngestionRun).where(PaperIngestionRun.task_id == task_id)
            )
            if run is not None:
                run.stage, run.status, run.quality_status, run.completed_at = (
                    "completed",
                    "succeeded",
                    "ready",
                    task.completed_at,
                )
            report = await session.scalar(
                select(IngestionQualityReport).where(IngestionQualityReport.task_id == task_id)
            )
            if report is not None:
                report.expected_mapping_count = 0
                report.mapped_chunk_count = 0
                report.mapping_failure_count = 0
                report.report_json = {
                    **report.report_json,
                    "local_chunks": count,
                    "knowledge_base_import": "not_required",
                }
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))

    async def _fail(self, task_id: str, code: str, message: str, retryable: bool) -> None:
        async with self._sessions() as session:
            task = await session.get(TaskRecord, task_id)
            if task is None:
                return
            paper = await session.get(Paper, task.resource_id) if task.resource_id else None
            failed_stage = task.stage
            task.status, task.stage, task.completed_at = "failed", "failed", datetime.now(UTC)
            task.error_json = {"code": code, "message": message, "retryable": retryable}
            if paper is not None:
                paper.status, paper.index_status, paper.quality_status = (
                    "failed",
                    "failed",
                    "failed",
                )
                paper.failure = {
                    "stage": failed_stage,
                    "error_code": code,
                    "message": message,
                    "retryable": retryable,
                }
            run = await session.scalar(
                select(PaperIngestionRun).where(PaperIngestionRun.task_id == task_id)
            )
            if run is not None:
                run.stage, run.status, run.completed_at, run.error_json = (
                    failed_stage,
                    "failed",
                    task.completed_at,
                    task.error_json,
                )
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))


async def _task_and_paper(
    session: AsyncSession, task_id: str, paper_id: str
) -> tuple[TaskRecord, Paper]:
    task = await session.get(TaskRecord, task_id)
    paper = await session.get(Paper, paper_id)
    if task is None or paper is None:
        raise IngestionFailure("PAPER_INGEST_FAILED", "论文入库任务不存在。", retryable=False)
    return task, paper


async def _complete_mapping(
    session: AsyncSession, paper_id: str, version_id: str, chunks: list[BuiltChunk]
) -> tuple[str, dict[str, str]] | None:
    """Reuse only a complete, content-identical mapping from a prior retry."""
    rows = list(
        (
            await session.scalars(
                select(RagMapping).where(
                    RagMapping.paper_id == paper_id,
                    RagMapping.paper_version_id == version_id,
                    RagMapping.status == "ready",
                )
            )
        ).all()
    )
    expected = {item.chunk_id: item.content_sha256 for item in chunks}
    by_chunk = {item.source_chunk_id: item for item in rows if item.source_chunk_id}
    if set(by_chunk) != set(expected):
        return None
    if any(by_chunk[key].content_sha256 != digest for key, digest in expected.items()):
        return None
    documents = {item.document_id for item in by_chunk.values()}
    if len(documents) != 1 or not all(item.ragflow_chunk_id for item in by_chunk.values()):
        return None
    return documents.pop(), {key: str(item.ragflow_chunk_id) for key, item in by_chunk.items()}


def _ingestion_stage(value: str) -> str:
    stages = {
        "mineru_parsing",
        "ocr_processing",
        "cleaning",
        "quality_check",
        "understanding",
        "indexing",
    }
    return value if value in stages else "mineru_parsing"


def _ingestion_progress(stage: str) -> float:
    return {
        "mineru_parsing": 0.05,
        "ocr_processing": 0.3,
        "cleaning": 0.5,
        "quality_check": 0.65,
        "understanding": 0.75,
        "indexing": 0.8,
    }.get(stage, 0.0)


def _image_hash(image_url: str | None) -> str | None:
    if not image_url:
        return None
    return hashlib.sha256(image_url.encode("utf-8")).hexdigest()


def _block_from_raw(raw: dict[str, Any], index: int) -> ParsedBlock:
    content = str(raw.get("content") or raw.get("text") or "").strip()
    page = _page(raw.get("page_number") or raw.get("page") or 1)
    source_ref = str(raw.get("source_ref") or f"page:{page}:block:{index}")
    content_type = str(raw.get("content_type") or "text")
    return ParsedBlock(
        str(raw.get("id") or f"block-{index}"),
        content,
        page,
        str(raw.get("section_title") or raw.get("section") or ""),
        content_type if content_type in {"text", "formula", "reference", "metadata"} else "text",
        source_ref,
    )


def _media_from_raw(raw: dict[str, Any], index: int) -> MediaObject:
    kind = str(raw.get("kind") or raw.get("type") or "figure")
    if kind not in {"figure", "table"}:
        kind = "figure"
    page = _page(raw.get("page_number") or raw.get("page") or 1)
    return MediaObject(
        str(raw.get("id") or f"media-{index}"),
        kind,
        page,
        str(raw.get("source_ref") or f"page:{page}:media:{index}"),
        raw.get("image_url") or raw.get("url"),
        raw.get("caption"),
        bool(raw.get("required", True)),
    )


def _build_chunks(
    paper_id: str, parsed: ParsedPaper, ocr_results: dict[str, str]
) -> list[BuiltChunk]:
    chunks: list[BuiltChunk] = []
    text_blocks = [block for block in parsed.blocks if block.content]
    text_ids = [f"{paper_id}:text:{index}" for index in range(1, len(text_blocks) + 1)]
    for index, block in enumerate(text_blocks, start=1):
        chunk_id = f"{paper_id}:text:{index}"
        content_role = {
            "formula": "formula",
            "reference": "reference_entry",
            "metadata": "metadata",
        }.get(block.content_type, "paragraph")
        chunks.append(
            _built(
                chunk_id,
                block.content,
                block.content_type,
                block.section_title,
                block.page_number,
                block.source_ref,
                metadata={
                    "content_role": content_role,
                    "section_path": [block.section_title] if block.section_title else [],
                    "prev_chunk_id": text_ids[index - 2] if index > 1 else None,
                    "next_chunk_id": text_ids[index] if index < len(text_ids) else None,
                    "retrieval_weight": 0.35 if content_role == "reference_entry" else 1.0,
                },
            )
        )
    for item in parsed.media:
        text = ocr_results.get(item.object_id, "")
        content = "\n".join(part for part in [item.caption or "", text] if part).strip()
        if not content:
            continue
        parent_id = f"{paper_id}:{item.kind}:{item.object_id}"
        chunks.append(
            _built(
                parent_id,
                content,
                item.kind,
                item.caption or item.kind,
                item.page_number,
                item.source_ref,
                object_id=item.object_id,
                metadata={
                    "content_role": f"{item.kind}_overview",
                    "section_path": [item.caption] if item.caption else [],
                },
            )
        )
        if text:
            child_type = "table" if item.kind == "table" else "figure"
            chunks.append(
                _built(
                    f"{parent_id}:ocr",
                    text,
                    child_type,
                    item.caption or item.kind,
                    item.page_number,
                    item.source_ref,
                    object_id=item.object_id,
                    parent_chunk_id=parent_id,
                    metadata={
                        "content_role": "table_rows"
                        if item.kind == "table"
                        else "figure_ocr",
                        "section_path": [item.caption] if item.caption else [],
                    },
                )
            )
    return chunks


def _built(
    chunk_id: str,
    content: str,
    content_type: str,
    section_title: str,
    page: int,
    source_ref: str,
    *,
    object_id: str | None = None,
    parent_chunk_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> BuiltChunk:
    return BuiltChunk(
        chunk_id,
        content,
        content_type,
        section_title,
        page,
        source_ref,
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
        object_id,
        parent_chunk_id,
        metadata,
    )


def _quality_report(
    parsed: ParsedPaper, chunks: list[BuiltChunk], ocr_results: dict[str, str]
) -> list[str]:
    errors: list[str] = []
    if not chunks:
        errors.append("no_indexable_chunks")
    for item in parsed.media:
        if item.required and not ocr_results.get(item.object_id):
            errors.append(f"ocr_missing:{item.object_id}")
    ids = {item.chunk_id for item in chunks}
    for chunk in chunks:
        if not chunk.content.strip() or not chunk.source_ref or chunk.page_number < 1:
            errors.append(f"invalid_chunk:{chunk.chunk_id}")
        if chunk.parent_chunk_id and chunk.parent_chunk_id not in ids:
            errors.append(f"orphan_chunk:{chunk.chunk_id}")
    return errors


def _understanding_evidences(paper_id: str, chunks: list[BuiltChunk]) -> list[EvidenceItem]:
    """Build a bounded, local evidence set for upload-time paper understanding."""
    candidates = [
        chunk
        for chunk in chunks
        if chunk.content.strip() and chunk.content_type != "reference" and (chunk.metadata or {}).get("indexable", True)
    ]
    if len(candidates) > 24:
        positions = sorted({round(index * (len(candidates) - 1) / 23) for index in range(24)})
        candidates = [candidates[index] for index in positions]
    return [
        EvidenceItem(
            evidence_id=f"U{index}",
            source_type="paper",
            paper_id=paper_id,
            document_id=f"local:{paper_id}",
            chunk_id=chunk.chunk_id,
            content_type=chunk.content_type,
            quote=chunk.content[:2_000],
            section_title=chunk.section_title or None,
            page_number=chunk.page_number,
            source_uri=f"paper://{paper_id}/{chunk.chunk_id}",
            retrieval_score=1.0,
        )
        for index, chunk in enumerate(candidates, start=1)
    ]


def _chunk_row(paper_id: str, version_id: str, item: BuiltChunk) -> PaperChunk:
    metadata = item.metadata or {}
    return PaperChunk(
        chunk_id=item.chunk_id,
        paper_id=paper_id,
        paper_version_id=version_id,
        content=item.content,
        content_type=item.content_type,
        content_role=str(metadata.get("content_role") or "paragraph"),
        section_title=item.section_title,
        section_path_json=list(metadata.get("section_path") or []),
        page_number=item.page_number,
        page_end=int(metadata.get("page_end") or item.page_number),
        source_ref=item.source_ref,
        object_id=item.object_id,
        parent_chunk_id=item.parent_chunk_id,
        prev_chunk_id=metadata.get("prev_chunk_id"),
        next_chunk_id=metadata.get("next_chunk_id"),
        retrieval_weight=float(metadata.get("retrieval_weight") or 1.0),
        quality_flags=list(metadata.get("quality_flags") or []),
        indexable=bool(metadata.get("indexable", True)),
        parser_version=str(metadata.get("parser_version") or "mineru-v1"),
        cleaning_version=str(metadata.get("cleaning_version") or "cleaning-v1"),
        content_sha256=item.content_sha256,
        metadata_json=metadata,
    )


def _ocr_text(data: Any) -> str:
    if isinstance(data, str):
        return data.strip()
    if not isinstance(data, dict):
        return ""
    if isinstance(data.get("text"), str):
        return data["text"].strip()
    words = data.get("words_result") or data.get("items") or data.get("cells") or []
    if isinstance(words, list):
        return "\n".join(
            str(item.get("words") or item.get("text") or item)
            for item in words
            if isinstance(item, (dict, str))
        ).strip()
    return ""


def _page(value: Any) -> int:
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 1


def _bearer(secret: Any) -> dict[str, str]:
    if not secret:
        return {}
    return {"Authorization": f"Bearer {secret.get_secret_value()}"}
