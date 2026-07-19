"""Durable non-workflow task handlers for cleanup, reports, exports and evaluation."""

# ruff: noqa: E501

import asyncio
import hashlib
import html
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import (
    IngestionQualityReport,
    MediaObjectRecord,
    Paper,
    PaperChunk,
    ParsedBlockRecord,
    RagMapping,
    ReadingReport,
    ReportExport,
    TaskRecord,
)
from app.runtime.adapters import task_view
from app.runtime.redis_store import RedisRuntime


class OperationFailure(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(code)


class OperationsTaskExecutor:
    def __init__(
        self, settings: Settings, sessions: async_sessionmaker[AsyncSession], redis: RedisRuntime
    ) -> None:
        self._settings = settings
        self._sessions = sessions
        self._redis = redis
        self._running: set[asyncio.Task[None]] = set()

    def submit(self, task_id: str) -> None:
        task = asyncio.create_task(self.run(task_id), name=f"operation:{task_id}")
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def run(self, task_id: str) -> None:
        try:
            async with self._sessions() as session:
                task = await session.get(TaskRecord, task_id)
                if task is None:
                    return
                task.status, task.stage, task.started_at = "running", "starting", datetime.now(UTC)
                await session.commit()
                await self._redis.set_task_state(task.task_id, task_view(task))
                task_type = task.task_type
            if task_type == "paper_cleanup":
                await self._cleanup(task_id)
            elif task_type == "reading_report":
                await self._build_report(task_id)
            elif task_type == "report_export":
                await self._export_report(task_id)
            elif task_type == "evaluation":
                await self._evaluate(task_id)
            else:
                raise OperationFailure("TASK_TYPE_UNSUPPORTED", "不支持的后台任务类型。", retryable=False)
        except OperationFailure as exc:
            await self._fail(task_id, exc.code, exc.message, exc.retryable)
        except Exception:
            await self._fail(task_id, "OPERATION_FAILED", "后台任务未能完成，请稍后重试。", True)

    async def _cleanup(self, task_id: str) -> None:
        async with self._sessions() as session:
            task = await _task(session, task_id)
            paper = await _paper(session, task.resource_id)
            task.stage, task.progress = "cleaning_ragflow", 0.2
            paper.deletion_requested_at = paper.deletion_requested_at or datetime.now(UTC)
            mappings = list(
                (await session.scalars(select(RagMapping).where(RagMapping.paper_id == paper.paper_id))).all()
            )
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))
            file_path = Path(paper.file_path)
        await self._delete_ragflow_documents(mappings)
        async with self._sessions() as session:
            task = await _task(session, task_id)
            paper = await _paper(session, task.resource_id)
            task.stage, task.progress = "cleaning_storage", 0.65
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))
            try:
                file_path.unlink(missing_ok=True)
            except OSError as exc:
                raise OperationFailure("OBJECT_STORAGE_CLEANUP_FAILED", "论文原文件清理失败。") from exc
            await session.execute(delete(RagMapping).where(RagMapping.paper_id == paper.paper_id))
            await session.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper.paper_id))
            await session.execute(delete(MediaObjectRecord).where(MediaObjectRecord.paper_id == paper.paper_id))
            await session.execute(delete(ParsedBlockRecord).where(ParsedBlockRecord.paper_id == paper.paper_id))
            await session.execute(
                delete(IngestionQualityReport).where(IngestionQualityReport.paper_id == paper.paper_id)
            )
            paper.deleted_at, paper.status, paper.index_status = datetime.now(UTC), "deleted", "cancelled"
            task.status, task.stage, task.progress, task.completed_at = (
                "succeeded",
                "completed",
                1.0,
                datetime.now(UTC),
            )
            task.result_json = {"paper_id": paper.paper_id, "deleted_at": paper.deleted_at.isoformat()}
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))

    async def _delete_ragflow_documents(self, mappings: list[RagMapping]) -> None:
        documents = {(item.dataset_id, item.document_id) for item in mappings}
        if not documents:
            return
        if not self._settings.ragflow_base_url or not self._settings.ragflow_api_key:
            raise OperationFailure("RAGFLOW_UNAVAILABLE", "论文知识库服务不可用，暂不能删除论文。")
        headers = {"Authorization": f"Bearer {self._settings.ragflow_api_key.get_secret_value()}"}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                for dataset_id, document_id in documents:
                    response = await client.delete(
                        self._settings.ragflow_base_url.rstrip("/")
                        + f"/api/v1/datasets/{dataset_id}/documents/{document_id}",
                        headers=headers,
                    )
                    if response.status_code not in {200, 202, 204, 404}:
                        response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OperationFailure("RAGFLOW_CLEANUP_FAILED", "论文知识库清理失败。") from exc

    async def _build_report(self, task_id: str) -> None:
        async with self._sessions() as session:
            task = await _task(session, task_id)
            report = await _report(session, task.resource_id)
            task.stage, task.progress = "collecting_evidence", 0.25
            papers = list(
                (
                    await session.scalars(
                        select(Paper).where(
                            Paper.paper_id.in_(report.paper_ids),
                            Paper.owner_id == report.user_id,
                            Paper.status == "ready",
                            Paper.deleted_at.is_(None),
                        )
                    )
                ).all()
            )
            if len(papers) != len(set(report.paper_ids)):
                raise OperationFailure("PAPER_NOT_READY", "报告包含尚未就绪的论文。", retryable=False)
            chunks = list(
                (
                    await session.scalars(
                        select(PaperChunk)
                        .where(PaperChunk.paper_id.in_(report.paper_ids))
                        .order_by(PaperChunk.paper_id, PaperChunk.page_number)
                    )
                ).all()
            )
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))
        markdown, evidence_ids = _report_markdown(report.title, papers, chunks)
        async with self._sessions() as session:
            task = await _task(session, task_id)
            report = await _report(session, task.resource_id)
            task.status, task.stage, task.progress, task.completed_at = (
                "succeeded",
                "completed",
                1.0,
                datetime.now(UTC),
            )
            task.result_json = {"report_id": report.report_id}
            report.status, report.content_markdown, report.evidence_ids, report.completed_at = (
                "succeeded",
                markdown,
                evidence_ids,
                task.completed_at,
            )
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))

    async def _export_report(self, task_id: str) -> None:
        async with self._sessions() as session:
            task = await _task(session, task_id)
            export = await session.scalar(select(ReportExport).where(ReportExport.task_id == task_id))
            if export is None:
                raise OperationFailure("REPORT_EXPORT_NOT_FOUND", "报告导出任务不存在。", retryable=False)
            report = await _report(session, export.report_id)
            if report.status != "succeeded" or not report.content_markdown:
                raise OperationFailure("REPORT_NOT_READY", "阅读报告尚未生成完成。", retryable=False)
            task.stage, task.progress = "rendering", 0.5
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))
            content, suffix = _render_export(report.content_markdown, export.format)
            path = self._settings.object_storage_path / "exports" / f"{export.export_id}.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            checksum = hashlib.sha256(content).hexdigest()
        async with self._sessions() as session:
            task = await _task(session, task_id)
            export = await session.scalar(select(ReportExport).where(ReportExport.task_id == task_id))
            assert export is not None
            export.status, export.file_path, export.checksum_sha256, export.completed_at = (
                "succeeded",
                str(path),
                checksum,
                datetime.now(UTC),
            )
            task.status, task.stage, task.progress, task.completed_at = (
                "succeeded",
                "completed",
                1.0,
                export.completed_at,
            )
            task.result_json = {
                "export_id": export.export_id,
                "download_url": f"/api/v1/reading-reports/{export.report_id}/exports/{export.export_id}/file",
                "checksum_sha256": checksum,
            }
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))

    async def _evaluate(self, task_id: str) -> None:
        async with self._sessions() as session:
            task = await _task(session, task_id)
            task.stage, task.progress = "evaluating", 0.5
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))
            payload = task.payload_json
            result = {
                "dataset_id": payload.get("dataset_id"),
                "split": payload.get("split", "default"),
                "experiment_type": payload.get("experiment_type"),
                "status": "queued_for_dataset_runner",
            }
            task.status, task.stage, task.progress, task.completed_at = (
                "succeeded",
                "completed",
                1.0,
                datetime.now(UTC),
            )
            task.result_json = result
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))

    async def _fail(self, task_id: str, code: str, message: str, retryable: bool) -> None:
        async with self._sessions() as session:
            task = await session.get(TaskRecord, task_id)
            if task is None:
                return
            task.status, task.stage, task.completed_at = "failed", "failed", datetime.now(UTC)
            task.error_json = {"code": code, "message": message, "retryable": retryable}
            if task.task_type == "reading_report" and task.resource_id:
                report = await session.get(ReadingReport, task.resource_id)
                if report is not None:
                    report.status = "failed"
            if task.task_type == "report_export":
                export = await session.scalar(select(ReportExport).where(ReportExport.task_id == task_id))
                if export is not None:
                    export.status = "failed"
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))


async def _task(session: AsyncSession, task_id: str) -> TaskRecord:
    task = await session.get(TaskRecord, task_id)
    if task is None:
        raise OperationFailure("TASK_NOT_FOUND", "任务不存在。", retryable=False)
    return task


async def _paper(session: AsyncSession, paper_id: str | None) -> Paper:
    paper = await session.get(Paper, paper_id) if paper_id else None
    if paper is None:
        raise OperationFailure("PAPER_NOT_FOUND", "论文不存在。", retryable=False)
    return paper


async def _report(session: AsyncSession, report_id: str | None) -> ReadingReport:
    report = await session.get(ReadingReport, report_id) if report_id else None
    if report is None:
        raise OperationFailure("REPORT_NOT_FOUND", "阅读报告不存在。", retryable=False)
    return report


def _report_markdown(title: str, papers: list[Paper], chunks: list[PaperChunk]) -> tuple[str, list[str]]:
    paper_by_id = {item.paper_id: item for item in papers}
    grouped: dict[str, list[PaperChunk]] = {item.paper_id: [] for item in papers}
    for item in chunks:
        if item.content_type != "reference" and len(grouped[item.paper_id]) < 6:
            grouped[item.paper_id].append(item)
    lines = [f"# {title}", "", "## 论文要点"]
    evidence_ids: list[str] = []
    for paper_id, items in grouped.items():
        paper = paper_by_id[paper_id]
        lines.extend(["", f"### {paper.title}"])
        for index, item in enumerate(items, start=1):
            evidence_id = f"R-{paper_id[:8]}-{index}"
            evidence_ids.append(evidence_id)
            lines.extend(
                [
                    f"- {item.section_title or '正文'}（第 {item.page_number} 页）：{item.content}",
                    f"  [{evidence_id}] {item.source_ref}",
                ]
            )
    if not evidence_ids:
        raise OperationFailure("RAG_NO_EVIDENCE", "论文中没有可用于报告的结构化证据。", retryable=False)
    return "\n".join(lines) + "\n", evidence_ids


def _render_export(markdown: str, format_name: str) -> tuple[bytes, str]:
    if format_name == "markdown":
        return markdown.encode("utf-8"), "md"
    if format_name == "pdf":
        return _minimal_pdf(markdown), "pdf"
    if format_name == "docx":
        return _minimal_docx(markdown), "docx"
    raise OperationFailure("EXPORT_FORMAT_UNSUPPORTED", "不支持的报告导出格式。", retryable=False)


def _minimal_pdf(markdown: str) -> bytes:
    lines = [line[:90] for line in markdown.splitlines() if line.strip()][:60]
    content = "BT\n/F1 10 Tf\n50 760 Td\n"
    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index:
            content += "0 -13 Td\n"
        content += f"({escaped}) Tj\n"
    content += "ET\n"
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        f"4 0 obj\n<< /Length {len(content.encode('latin-1', 'replace'))} >>\nstream\n{content}endstream\nendobj\n",
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    header = "%PDF-1.4\n"
    offsets: list[int] = []
    cursor = len(header)
    for item in objects:
        offsets.append(cursor)
        cursor += len(item.encode("latin-1", "replace"))
    xref = "xref\n0 6\n0000000000 65535 f \n" + "".join(
        f"{offset:010d} 00000 n \n" for offset in offsets
    )
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{cursor}\n%%EOF\n"
    return (header + "".join(objects) + xref + trailer).encode("latin-1", "replace")


def _minimal_docx(markdown: str) -> bytes:
    paragraphs = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">{html.escape(line)}</w:t></w:r></w:p>"
        for line in markdown.splitlines()
        if line
    )
    document = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        f"<w:body>{paragraphs}<w:sectPr/></w:body></w:document>"
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
            "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
            "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>"
            "</Relationships>",
        )
        archive.writestr("word/document.xml", document)
    return output.getvalue()
