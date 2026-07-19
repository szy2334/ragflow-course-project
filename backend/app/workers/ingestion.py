"""Strict PDF ingestion: MinerU -> Baidu OCR -> chunks -> quality gate -> RAGFlow."""

# ruff: noqa: E501

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Paper, PaperChunk, RagMapping, TaskRecord
from app.runtime.adapters import task_view
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
                "user_id": paper.owner_id,
                "quality_status": "ready",
            },
            "chunks": [
                {
                    "id": item.chunk_id,
                    "content": item.content,
                    "metadata": {
                        "paper_id": paper.paper_id,
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
            raise IngestionFailure("RAGFLOW_MAPPING_INCOMPLETE", "论文知识库映射不完整。")
        return document_id, mapping


class IngestionTaskExecutor:
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
            await self._run(task_id)
        except IngestionFailure as exc:
            await self._fail(task_id, exc.code, exc.message, exc.retryable)
        except Exception:
            await self._fail(task_id, "PAPER_INGEST_FAILED", "论文入库失败，请稍后重试。", True)

    async def _run(self, task_id: str) -> None:
        async with self._sessions() as session:
            task = await session.get(TaskRecord, task_id)
            if task is None or not task.resource_id:
                return
            paper = await session.get(Paper, task.resource_id)
            if paper is None:
                return
            task.status, task.stage, task.started_at = (
                "running",
                "mineru_parsing",
                datetime.now(UTC),
            )
            paper.status, paper.parse_progress, paper.index_status, paper.failure = (
                "mineru_parsing",
                0.05,
                "pending",
                None,
            )
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))
            file_path = Path(paper.file_path)
            paper_id = paper.paper_id

        parsed = await MinerUClient(self._settings).parse(file_path)
        async with self._sessions() as session:
            task, paper = await _task_and_paper(session, task_id, paper_id)
            task.stage, task.progress = "ocr_processing", 0.3
            paper.status, paper.parse_progress = "ocr_processing", 0.3
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))

        ocr_results: dict[str, str] = {}
        ocr = BaiduSpecializedOcrClient(self._settings)
        for item in parsed.media:
            ocr_results[item.object_id] = await ocr.recognize(item)

        chunks = _build_chunks(paper_id, parsed, ocr_results)
        report = _quality_report(parsed, chunks, ocr_results)
        if report:
            raise IngestionFailure(
                "CHUNK_QUALITY_FAILED", "论文结构化质量检查未通过。", retryable=False
            )
        async with self._sessions() as session:
            task, paper = await _task_and_paper(session, task_id, paper_id)
            task.stage, task.progress = "quality_check", 0.65
            paper.status, paper.parse_progress, paper.quality_status = (
                "quality_check",
                0.65,
                "ready",
            )
            await session.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper_id))
            session.add_all([_chunk_row(paper_id, item) for item in chunks])
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))

        async with self._sessions() as session:
            task, paper = await _task_and_paper(session, task_id, paper_id)
            task.stage, task.progress = "indexing", 0.8
            paper.status, paper.parse_progress, paper.index_status = "indexing", 0.8, "running"
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))
            paper_for_import = paper
        document_id, mappings = await RagFlowManualImporter(self._settings).import_chunks(
            paper_for_import, chunks
        )

        async with self._sessions() as session:
            task, paper = await _task_and_paper(session, task_id, paper_id)
            await session.execute(delete(RagMapping).where(RagMapping.paper_id == paper_id))
            session.add_all(
                [
                    RagMapping(
                        paper_id=paper_id,
                        source_chunk_id=chunk.chunk_id,
                        dataset_id=self._settings.ragflow_user_dataset_id or "",
                        document_id=document_id,
                        ragflow_chunk_id=mappings[chunk.chunk_id],
                        content_sha256=chunk.content_sha256,
                    )
                    for chunk in chunks
                ]
            )
            task.status, task.stage, task.progress, task.completed_at = (
                "succeeded",
                "completed",
                1.0,
                datetime.now(UTC),
            )
            task.result_json = {
                "paper_id": paper_id,
                "document_id": document_id,
                "mapped_chunks": len(chunks),
            }
            paper.status, paper.parse_progress, paper.index_status, paper.quality_status = (
                "ready",
                1.0,
                "succeeded",
                "ready",
            )
            paper.active_index_version = (paper.active_index_version or 0) + 1
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
    for index, block in enumerate(parsed.blocks, start=1):
        if not block.content:
            continue
        chunk_id = f"{paper_id}:text:{index}"
        chunks.append(
            _built(
                chunk_id,
                block.content,
                block.content_type,
                block.section_title,
                block.page_number,
                block.source_ref,
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
                metadata={"content_role": f"{item.kind}_overview"},
            )
        )
        child_type = "table" if item.kind == "table" else "figure_caption"
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
                metadata={"content_role": "table_rows" if item.kind == "table" else "figure_ocr"},
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


def _chunk_row(paper_id: str, item: BuiltChunk) -> PaperChunk:
    return PaperChunk(
        chunk_id=item.chunk_id,
        paper_id=paper_id,
        content=item.content,
        content_type=item.content_type,
        section_title=item.section_title,
        page_number=item.page_number,
        source_ref=item.source_ref,
        object_id=item.object_id,
        parent_chunk_id=item.parent_chunk_id,
        content_sha256=item.content_sha256,
        metadata_json=item.metadata or {},
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
