"""PDF ingestion: parse -> OCR -> chunks -> quality -> AI understanding."""

# ruff: noqa: E501

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.agents import PaperUnderstandingAgent
from app.ai.errors import ModelTransportError
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
    PaperVersion,
    ParsedBlockRecord,
    TaskRecord,
)
from app.runtime.adapters import task_view
from app.runtime.executor import snapshot_from_settings
from app.runtime.redis_store import RedisRuntime
from app.workers.external_pipeline import (
    ExternalPipelineError,
    parse_mineru_pdf,
    recognize_baidu_media,
)
from app.workers.second_clean_adapter import build_chunks as build_second_clean_chunks


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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MediaObject:
    object_id: str
    kind: str
    page_number: int
    source_ref: str
    image_url: str | None
    caption: str | None
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


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

    async def parse(self, file_path: Path, *, paper_id: str, paper_version_id: str) -> ParsedPaper:
        if not self._settings.mineru_base_url or not self._settings.mineru_api_key:
            raise IngestionFailure("MINERU_UNAVAILABLE", "论文解析服务尚未配置。")
        artifact_root = (
            self._settings.object_storage_path.resolve()
            / paper_id
            / paper_version_id
            / "mineru"
        )
        try:
            raw_blocks, raw_media = await asyncio.to_thread(
                parse_mineru_pdf,
                self._settings,
                pdf_path=file_path,
                paper_id=paper_id,
                paper_version_id=paper_version_id,
                artifact_root=artifact_root,
            )
        except (OSError, ExternalPipelineError) as exc:
            raise IngestionFailure("MINERU_PARSE_FAILED", "论文解析失败，请稍后重试。") from exc
        blocks = [
            _block_from_pipeline(item, index)
            for index, item in enumerate(raw_blocks, start=1)
            if isinstance(item, dict)
        ]
        media = [
            _media_from_pipeline(item, index)
            for index, item in enumerate(raw_media, start=1)
            if isinstance(item, dict)
        ]
        if not blocks:
            raise IngestionFailure("MINERU_PARSE_FAILED", "论文解析结果为空。")
        return ParsedPaper(blocks=blocks, media=media)


class BaiduSpecializedOcrClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def recognize(self, item: MediaObject) -> dict[str, Any]:
        if not item.required:
            return {"status": "skipped", "ocr_text": "", "processors": []}
        if not self._settings.baidu_ocr_api_key or not self._settings.baidu_ocr_secret_key:
            raise IngestionFailure("BAIDU_OCR_SPECIALIZED_FAILED", "专项 OCR 服务尚未配置。")
        media = dict(item.metadata.get("pipeline_media") or {})
        media.setdefault("object_id", item.object_id)
        media.setdefault("object_type", item.kind)
        media.setdefault("image_path", item.image_url)
        media.setdefault("page_start", item.page_number)
        raw_root = self._settings.object_storage_path.resolve() / "ocr" / item.object_id
        try:
            return await asyncio.to_thread(
                recognize_baidu_media, self._settings, media, raw_root=raw_root
            )
        except ExternalPipelineError as exc:
            raise IngestionFailure("BAIDU_OCR_SPECIALIZED_FAILED", "专项 OCR 识别失败。") from exc


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
            document = {
                "paper_id": paper.paper_id,
                "paper_version_id": paper.paper_version_id,
                "title": paper.title,
                "file_name": paper.file_name,
                "file_sha256": paper.content_sha256,
                "parser_name": "mineru",
                "parser_version": "mineru-v1",
            }

        if stage == "mineru_parsing" or not await self._has_parse_artifacts(paper_id, version_id):
            parsed = await MinerUClient(self._settings).parse(
                file_path, paper_id=paper_id, paper_version_id=version_id
            )
            await self._store_parse_artifacts(task_id, paper_id, version_id, parsed)
        parsed = await self._load_parsed_artifacts(paper_id, version_id)

        await self._set_stage(task_id, paper_id, "ocr_processing")
        ocr_results = await self._run_ocr(task_id, paper_id, version_id, parsed.media)
        parsed = ParsedPaper(blocks=parsed.blocks, media=parsed.media)

        await self._set_stage(task_id, paper_id, "cleaning")
        second_clean = build_second_clean_chunks(
            document=document,
            blocks=_second_clean_blocks(parsed.blocks),
            media_objects=_second_clean_media(parsed.media),
            ocr_by_id=_second_clean_ocr(ocr_results, parsed.media),
        )
        chunks = [_chunk_from_second_clean(item) for item in second_clean.chunks]
        await self._store_chunks(paper_id, version_id, chunks)

        await self._set_stage(task_id, paper_id, "quality_check")
        await self._store_quality_report(
            task_id, paper_id, version_id, second_clean.quality_report
        )
        if second_clean.blocking_errors:
            raise IngestionFailure(
                "CHUNK_QUALITY_FAILED", "论文结构化质量检查未通过。", retryable=False
            )

        await self._set_stage(task_id, paper_id, "understanding")
        # Parsing, OCR, second-clean and local retrieval remain useful without
        # a generation model.  Record the missing capability explicitly and
        # mark the document ready for evidence retrieval instead of discarding
        # already verified chunks.  Answer generation and format judgment keep
        # their separate MODEL_NOT_CONFIGURED guardrails.
        if not self._settings.paper_summary_enabled:
            await self._store_understanding(
                paper_id,
                {
                    "status": "disabled",
                    "reason": "PAPER_SUMMARY_DISABLED",
                    "message": "已完成解析、清洗与本地检索；上传时论文摘要功能已关闭。",
                },
            )
            await self._complete(
                task_id,
                paper_id,
                len(chunks),
                quality_status=str(second_clean.quality_report["status"]),
            )
            return
        if not self._settings.llm_base_url or not self._settings.llm_api_key or not self._settings.llm_model:
            await self._store_understanding(
                paper_id,
                {
                    "status": "unavailable",
                    "reason": "MODEL_NOT_CONFIGURED",
                    "message": "已完成解析、清洗与本地检索；配置生成模型后可生成理解与总结。",
                },
            )
            await self._complete(
                task_id,
                paper_id,
                len(chunks),
                quality_status=str(second_clean.quality_report["status"]),
            )
            return
        try:
            understanding = await self._understand(paper_id, chunks)
        except IngestionFailure as exc:
            # Summary generation is optional enrichment.  Model and schema
            # failures must not discard validated chunks or block reading.
            await self._store_understanding(
                paper_id,
                {
                    "status": (
                        "unavailable"
                        if exc.code in {"MODEL_ENDPOINT_UNAVAILABLE", "MODEL_NOT_CONFIGURED"}
                        else "failed"
                    ),
                    "reason": exc.code,
                    "message": "论文已完成结构化入库；论文理解模型服务暂不可用，可稍后重试生成摘要。",
                },
            )
            await self._complete(
                task_id,
                paper_id,
                len(chunks),
                quality_status=str(second_clean.quality_report["status"]),
            )
            return
        await self._store_understanding(paper_id, understanding)

        await self._complete(
            task_id,
            paper_id,
            len(chunks),
            quality_status=str(second_clean.quality_report["status"]),
        )

    async def _understand(self, paper_id: str, chunks: list[BuiltChunk]) -> dict[str, Any]:
        if not self._settings.llm_base_url or not self._settings.llm_api_key or not self._settings.llm_model:
            raise IngestionFailure("MODEL_NOT_CONFIGURED", "模型服务尚未配置。", retryable=False)
        evidences = _understanding_evidences(paper_id, chunks)
        if not evidences:
            raise IngestionFailure("PAPER_UNDERSTANDING_FAILED", "论文缺少可供理解的正文内容。")
        try:
            summary, _ = await PaperUnderstandingAgent(
                OpenAICompatibleClient(self._settings.llm_api_key), PromptRepository()
            ).run_summary(
                evidences=evidences,
                configuration=snapshot_from_settings(self._settings),
            )
        except ModelTransportError as exc:
            raise IngestionFailure(
                "MODEL_ENDPOINT_UNAVAILABLE",
                "论文理解模型服务暂不可用，请检查模型服务连接。",
            ) from exc
        except Exception as exc:
            raise IngestionFailure("PAPER_UNDERSTANDING_FAILED", "论文智能理解失败，请稍后重试。") from exc
        return {
            "status": "ready",
            "summary_markdown": summary.summary_markdown,
            "paper_summary": summary.summary_markdown,
            "model_version": self._settings.model_config_version,
            "prompt_version": self._settings.prompt_version,
        }

    async def _store_understanding(self, paper_id: str, understanding: dict[str, Any]) -> None:
        async with self._sessions() as session:
            paper = await session.get(Paper, paper_id)
            if paper is None:
                raise IngestionFailure("PAPER_NOT_FOUND", "论文不存在。", retryable=False)
            paper.understanding_json = understanding
            paper.summary_markdown = understanding.get("summary_markdown")
            paper.summary_status = str(understanding.get("status") or "failed")
            paper.summary_model_version = understanding.get("model_version")
            paper.summary_prompt_version = understanding.get("prompt_version")
            paper.summary_generated_at = datetime.now(UTC) if paper.summary_markdown else None
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
                        bbox_json=block.metadata.get("bbox"),
                        source_ref=block.source_ref,
                        metadata_json=block.metadata,
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
                        bbox_json=item.metadata.get("pdf_bbox"),
                        source_ref=item.source_ref,
                        image_url=item.image_url,
                        image_sha256=_image_hash(item.image_url),
                        caption=item.caption,
                        required=item.required,
                        raw_response_json={"parser_metadata": item.metadata},
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
                    item.metadata_json or {},
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
                    (
                        item.raw_response_json.get("parser_metadata", {})
                        if isinstance(item.raw_response_json, dict)
                        else {}
                    ),
                )
                for item in media
            ],
        )

    async def _run_ocr(
        self, task_id: str, paper_id: str, version_id: str, media: list[MediaObject]
    ) -> dict[str, dict[str, Any]]:
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
        result: dict[str, dict[str, Any]] = {}
        for item in media:
            row = rows[item.object_id]
            if not item.required:
                result[item.object_id] = {"status": "skipped", "ocr_text": row.ocr_text or ""}
                continue
            if row.ocr_status != "success":
                try:
                    ocr_result = await client.recognize(item)
                except IngestionFailure:
                    await self._record_ocr_failure(paper_id, version_id, item.object_id)
                    raise
                text = _ocr_result_text(ocr_result)
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
                    processors = ocr_result.get("processors")
                    current.engines_json = [str(value) for value in processors] if isinstance(processors, list) else []
                    current.ocr_engine = current.engines_json[-1] if current.engines_json else None
                    current.raw_response_json = {
                        "parser_metadata": item.metadata,
                        "normalized_ocr": ocr_result,
                    }
                    current.failure_json = None
                    await session.commit()
                row.ocr_status, row.ocr_text = "success", text
                result[item.object_id] = ocr_result
                continue
            result[item.object_id] = {
                "status": "success",
                "ocr_text": row.ocr_text or "",
                "processors": row.engines_json or [],
            }
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
            version = await session.get(PaperVersion, version_id)
            if version is not None and chunks:
                version.cleaning_version = str(
                    (chunks[0].metadata or {}).get("cleaning_version")
                    or version.cleaning_version
                )
            await session.commit()

    async def _store_quality_report(
        self, task_id: str, paper_id: str, version_id: str, report: dict[str, Any]
    ) -> None:
        async with self._sessions() as session:
            item = await session.scalar(
                select(IngestionQualityReport).where(IngestionQualityReport.task_id == task_id)
            )
            status = str(report["status"])
            errors = list(report.get("critical_errors") or [])
            indexable_chunk_count = int(report.get("indexable_chunks") or 0)
            if item is None:
                session.add(
                    IngestionQualityReport(
                        task_id=task_id,
                        paper_id=paper_id,
                        paper_version_id=version_id,
                        status=status,
                        indexable_chunk_count=indexable_chunk_count,
                        blocking_error_count=len(errors),
                        expected_mapping_count=0,
                        mapped_chunk_count=0,
                        mapping_failure_count=0,
                        report_json=report,
                    )
                )
            else:
                item.status = status
                item.indexable_chunk_count = indexable_chunk_count
                item.blocking_error_count = len(errors)
                item.expected_mapping_count = 0
                item.mapped_chunk_count = 0
                item.mapping_failure_count = 0
                item.report_json = report
            run = await session.scalar(
                select(PaperIngestionRun).where(PaperIngestionRun.task_id == task_id)
            )
            if run is not None:
                run.stage, run.quality_status = "quality_check", status
                run.cleaning_version = str(
                    report.get("cleaning_version") or run.cleaning_version
                )
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

    async def _complete(
        self,
        task_id: str,
        paper_id: str,
        count: int,
        *,
        quality_status: str,
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
                quality_status,
            )
            run = await session.scalar(
                select(PaperIngestionRun).where(PaperIngestionRun.task_id == task_id)
            )
            if run is not None:
                run.stage, run.status, run.quality_status, run.completed_at = (
                    "completed",
                    "succeeded",
                    quality_status,
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


def _block_from_pipeline(raw: dict[str, Any], index: int) -> ParsedBlock:
    """Keep the complete MinerU record while exposing its common fields to SQL."""

    content = str(raw.get("content") or raw.get("text") or "").strip()
    content = str(raw.get("normalized_text") or content).strip()
    page = _page(raw.get("page_start") or raw.get("page_number") or raw.get("page") or 1)
    source_ref = str(raw.get("source_ref") or f"page:{page}:block:{index}")
    section_path = raw.get("section_path") if isinstance(raw.get("section_path"), list) else []
    section_title = str(section_path[-1]) if section_path else ""
    return ParsedBlock(
        str(raw.get("block_id") or raw.get("id") or f"block-{index}"),
        content,
        page,
        section_title or str(raw.get("section_title") or raw.get("section") or ""),
        str(raw.get("content_type") or "text"),
        source_ref,
        {"second_clean_block": raw, "bbox": raw.get("bbox")},
    )


def _media_from_pipeline(raw: dict[str, Any], index: int) -> MediaObject:
    kind = str(raw.get("object_type") or raw.get("kind") or raw.get("type") or "image")
    page = _page(raw.get("page_start") or raw.get("page_number") or raw.get("page") or 1)
    caption = raw.get("caption")
    if isinstance(caption, list):
        caption = " ".join(str(value) for value in caption if str(value).strip()) or None
    return MediaObject(
        str(raw.get("object_id") or raw.get("id") or f"media-{index}"),
        kind,
        page,
        str(raw.get("source_ref") or f"page:{page}:media:{index}"),
        raw.get("image_path") or raw.get("image_url") or raw.get("url"),
        str(caption) if caption else None,
        bool(raw.get("required", True)),
        {"pipeline_media": raw, "pdf_bbox": raw.get("pdf_bbox")},
    )


def _second_clean_blocks(blocks: list[ParsedBlock]) -> list[dict[str, Any]]:
    """Translate API parser records to second_clean's stable block contract."""

    converted: list[dict[str, Any]] = []
    for block in blocks:
        source = block.metadata.get("second_clean_block")
        if isinstance(source, dict):
            converted.append(dict(source))
            continue
        if block.content_type == "formula":
            role, indexable = "display_formula", True
        elif block.content_type == "reference":
            role, indexable = "reference_entry", True
        elif block.content_type == "metadata":
            role, indexable = "metadata", False
        else:
            role, indexable = "paragraph", True
        section_path = [block.section_title] if block.section_title else []
        converted.append(
            {
                "block_id": block.block_id,
                "raw_text": block.content,
                "normalized_text": block.content,
                "content_type": block.content_type,
                "content_role": role,
                "section_path": section_path,
                "page_start": block.page_number,
                "page_end": block.page_number,
                "bbox": None,
                "source_ref": block.source_ref,
                "indexable": indexable,
                "quality_flags": [],
            }
        )
    return converted


def _second_clean_media(media: list[MediaObject]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in media:
        source = item.metadata.get("pipeline_media")
        if isinstance(source, dict):
            converted.append(dict(source))
            continue
        converted.append({
            "object_id": item.object_id,
            "object_type": item.kind,
            "caption": [item.caption] if item.caption else [],
            "nearby_text": [],
            "section_path": [item.caption] if item.caption else [],
            "page_start": item.page_number,
            "page_end": item.page_number,
            "pdf_bbox": None,
            "source_ref": item.source_ref,
            "block_id": item.object_id,
            "quality_flags": [],
        })
    return converted


def _second_clean_ocr(
    ocr_results: dict[str, dict[str, Any]], media: list[MediaObject]
) -> dict[str, dict[str, Any]]:
    media_by_id = {item.object_id: item for item in media}
    results: dict[str, dict[str, Any]] = {}
    for object_id, ocr_result in ocr_results.items():
        if not isinstance(ocr_result, dict) or ocr_result.get("status") != "success":
            continue
        item = media_by_id.get(object_id)
        value = dict(ocr_result)
        text = _ocr_result_text(value)
        if text:
            value["ocr_text"] = text
        if item is not None and item.kind == "table" and text and not value.get("table_markdown_candidates"):
            # The API OCR adapter currently returns normalized table text rather
            # than a matrix.  Keep it as a table candidate instead of discarding
            # it merely because it is not HTML.
            value["table_markdown_candidates"] = [text]
        results[object_id] = value
    return results


def _ocr_result_text(result: dict[str, Any]) -> str:
    direct = result.get("ocr_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for key in ("table_markdown_candidates", "table_html_candidates", "visual_descriptions"):
        value = result.get(key)
        if isinstance(value, list):
            text = "\n".join(str(item).strip() for item in value if str(item).strip())
            if text:
                return text
    return ""


def _chunk_from_second_clean(item: dict[str, Any]) -> BuiltChunk:
    metadata = {
        "content_role": item["content_role"],
        "section_path": list(item.get("section_path") or []),
        "section_role": item.get("section_role") or "body",
        "prev_chunk_id": item.get("prev_chunk_id"),
        "next_chunk_id": item.get("next_chunk_id"),
        "retrieval_weight": float(item.get("retrieval_weight") or 1.0),
        "quality_flags": list(item.get("quality_flags") or []),
        "indexable": bool(item.get("indexable")),
        "parser_version": item.get("parser_version") or "mineru-v1",
        "cleaning_version": item.get("cleaning_version") or "paper_second_clean_v2",
        "raw_content": item.get("raw_content") or "",
        "source_refs": list(item.get("source_refs") or []),
        "source_block_ids": list(item.get("source_block_ids") or []),
        "bbox": item.get("bbox"),
        "provenance": item.get("provenance") or {},
    }
    page_start = int(item.get("page_start") or 1)
    return _built(
        str(item["source_chunk_id"]),
        str(item["content"]),
        str(item["content_type"]),
        str(item.get("section") or ""),
        page_start,
        str(item.get("source_ref") or f"paper://{item['paper_id']}/{item['source_chunk_id']}"),
        object_id=item.get("object_id"),
        parent_chunk_id=item.get("parent_chunk_id"),
        metadata={**metadata, "page_end": int(item.get("page_end") or page_start)},
    )


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


def _understanding_evidences(paper_id: str, chunks: list[BuiltChunk]) -> list[EvidenceItem]:
    """Build a bounded, chapter-balanced evidence set for paper understanding."""
    candidates = [
        chunk
        for chunk in chunks
        if (
            chunk.content.strip()
            and chunk.content_type != "reference"
            and (chunk.metadata or {}).get("indexable", True)
            and (chunk.metadata or {}).get("content_role") != "paper_metadata"
        )
    ]
    candidates = _sample_by_section(candidates, limit=24)
    return [
        EvidenceItem(
            evidence_id=f"U{index}",
            source_type="paper",
            paper_id=paper_id,
            document_id=f"local:{paper_id}",
            chunk_id=chunk.chunk_id,
            content_type=_understanding_content_type(chunk.content_type),
            quote=chunk.content[:2_000],
            section_title=chunk.section_title or None,
            page_number=chunk.page_number,
            source_uri=f"paper://{paper_id}/{chunk.chunk_id}",
            retrieval_score=1.0,
        )
        for index, chunk in enumerate(candidates, start=1)
    ]


def _understanding_content_type(value: str) -> str:
    """Map parser-specific text labels to the public evidence contract.

    The second-clean pipeline preserves source labels such as ``abstract`` and
    ``heading``.  The AI ``EvidenceItem`` schema intentionally exposes a
    smaller set of evidence media types, so unsupported textual labels must be
    represented as ``text`` rather than making the entire ingestion task fail
    during Pydantic validation.
    """

    normalized = value.strip().lower()
    accepted = {
        "text",
        "figure",
        "figure_caption",
        "table",
        "formula",
        "metadata",
        "reference",
    }
    if normalized in accepted:
        return normalized
    if normalized in {"caption", "image_caption", "table_caption"}:
        return "figure_caption"
    if normalized in {"equation", "math"}:
        return "formula"
    if normalized in {"title", "heading", "section", "keyword"}:
        return "metadata"
    return "text"


def _sample_by_section(chunks: list[BuiltChunk], *, limit: int) -> list[BuiltChunk]:
    """Allocate the bounded understanding context across sections, not positions.

    Every represented section receives one slot before the remaining slots are
    apportioned to longer sections.  Selected chunks within a section are
    evenly spaced, preserving coverage without privileging the document's
    opening pages.
    """

    if len(chunks) <= limit:
        return chunks
    sections: dict[tuple[str, ...], list[BuiltChunk]] = {}
    for chunk in chunks:
        metadata = chunk.metadata or {}
        path = tuple(str(part) for part in metadata.get("section_path") or [] if part)
        key = path or (chunk.section_title or "Document",)
        sections.setdefault(key, []).append(chunk)

    ordered = sorted(
        sections.items(),
        key=lambda item: (min(chunk.page_number for chunk in item[1]), item[0]),
    )
    if len(ordered) > limit:
        # Extremely fragmented parser output: retain the first chunk from the
        # earliest sections rather than reverting to global positional sampling.
        return [chunks[0] for _, chunks in ordered[:limit]]

    quotas = {key: 1 for key, _ in ordered}
    remaining = limit - len(ordered)
    while remaining:
        key, section_chunks = max(
            ordered,
            key=lambda item: (len(item[1]) / quotas[item[0]], -item[1][0].page_number),
        )
        quotas[key] += 1
        remaining -= 1

    selected: list[BuiltChunk] = []
    for key, section_chunks in ordered:
        count = min(quotas[key], len(section_chunks))
        if count == 1:
            selected.append(section_chunks[len(section_chunks) // 2])
            continue
        positions = {
            round(index * (len(section_chunks) - 1) / (count - 1))
            for index in range(count)
        }
        selected.extend(section_chunks[index] for index in sorted(positions))
    return sorted(selected, key=lambda chunk: (chunk.page_number, chunk.chunk_id))[:limit]


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


def _page(value: Any) -> int:
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 1
