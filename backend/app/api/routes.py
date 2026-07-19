"""The /api/v1 REST and SSE contract."""

# ruff: noqa: B008, E501

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, Request, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.schemas import StartQaWorkflowCommand
from app.api.dependencies import (
    current_user,
    get_session,
    require_admin,
    require_idempotency_key,
    require_request_id,
)
from app.api.schemas import (
    AnalysisInput,
    CancelInput,
    ComparisonInput,
    ConfigUpdateInput,
    EvaluationInput,
    ExportInput,
    FeedbackInput,
    LoginInput,
    PaperRetryInput,
    QuestionInput,
    ReadingReportInput,
    RegisterInput,
    SessionCreateInput,
    SessionUpdateInput,
)
from app.api.serialization import message_view, paper_view, session_view, task_view, user_view
from app.core.errors import ApiError, envelope
from app.core.security import (
    hash_password,
    hash_refresh_token,
    issue_access_token,
    new_refresh_token,
    verify_password,
)
from app.db.models import (
    ChatMessage,
    ChatSession,
    ConfigurationVersion,
    Feedback,
    Paper,
    PaperChunk,
    RagMapping,
    ReadingReport,
    RefreshToken,
    ReportExport,
    TaskRecord,
    TraceRecord,
    User,
    WorkflowRun,
    new_id,
)
from app.runtime.executor import snapshot_from_settings
from app.services.idempotency import replay_or_raise, request_fingerprint, save_response

router = APIRouter()


def _page(page: int, page_size: int) -> tuple[int, int]:
    if page < 1 or page_size < 1 or page_size > 100:
        raise ApiError(422, "VALIDATION_ERROR", "page 和 page_size 超出允许范围。")
    return page, page_size


def _accepted(task: TaskRecord, *, message_id: str | None = None) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "message_id": message_id,
        "status": task.status,
        "status_url": f"/api/v1/tasks/{task.task_id}",
        "stream_url": f"/api/v1/messages/{message_id}/events" if message_id else None,
        "resource_id": task.resource_id,
    }


def _json(
    status_code: int,
    data: object,
    request_id: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(envelope(data, request_id)),
        headers=headers,
    )


def _report_view(report: ReadingReport) -> dict[str, object]:
    return {
        "report_id": report.report_id,
        "user_id": report.user_id,
        "title": report.title,
        "paper_ids": report.paper_ids,
        "status": report.status,
        "content_markdown": report.content_markdown,
        "claims": report.claims_json,
        "evidence_ids": report.evidence_ids,
        "created_at": report.created_at,
        "completed_at": report.completed_at,
    }


_CONFIG_ID_FIELDS = {
    "model": "model_config_id",
    "prompt": "prompt_template_id",
    "retrieval": "retrieval_config_id",
}
_SECRET_FIELD_MARKERS = (
    "api_key",
    "secret",
    "password",
    "authorization",
    "access_token",
    "refresh_token",
)


def _contains_secret(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            any(marker in str(key).lower() for marker in _SECRET_FIELD_MARKERS)
            or _contains_secret(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    return False


def _public_config_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _public_config_value(item)
            for key, item in value.items()
            if not any(marker in str(key).lower() for marker in _SECRET_FIELD_MARKERS)
        }
    if isinstance(value, list):
        return [_public_config_value(item) for item in value]
    return value


def _config_view(config: ConfigurationVersion) -> dict[str, object]:
    value = _public_config_value(config.payload_json)
    data = dict(value) if isinstance(value, dict) else {"value": value}
    data.update(
        {
            _CONFIG_ID_FIELDS[config.kind]: config.configuration_id,
            "kind": config.kind,
            "version": config.version,
            "status": "active",
            "updated_at": config.created_at,
        }
    )
    return data


def _next_config_version(config: ConfigurationVersion) -> str:
    prefix, separator, suffix = config.version.rpartition(":")
    if separator and suffix.isdecimal():
        return f"{prefix}:{int(suffix) + 1}"
    return f"{config.version}:2"


async def _configuration(
    session: AsyncSession, *, kind: str, configuration_id: str
) -> ConfigurationVersion:
    item = await session.scalar(
        select(ConfigurationVersion).where(
            ConfigurationVersion.configuration_id == configuration_id,
            ConfigurationVersion.kind == kind,
        )
    )
    if item is None:
        raise ApiError(404, "CONFIG_NOT_FOUND", "配置不存在。")
    return item


async def _ready_papers(
    session: AsyncSession, *, user_id: str, paper_ids: list[str]
) -> list[Paper]:
    papers = list(
        (
            await session.scalars(
                select(Paper).where(
                    Paper.paper_id.in_(paper_ids),
                    Paper.owner_id == user_id,
                    Paper.status == "ready",
                    Paper.deleted_at.is_(None),
                )
            )
        ).all()
    )
    if len(papers) != len(set(paper_ids)):
        raise ApiError(409, "PAPER_NOT_READY", "论文尚未完成解析和索引。", {"paper_ids": paper_ids})
    return papers


async def _create_internal_workflow(
    session: AsyncSession,
    *,
    request: Request,
    user: User,
    paper_ids: list[str],
    question: str,
    task_type: str,
    resource_id: str,
    request_id: str,
) -> tuple[TaskRecord, ChatMessage, StartQaWorkflowCommand]:
    task_id, message_id = new_id(), new_id()
    chat_session = ChatSession(
        user_id=user.user_id,
        title=f"系统任务：{task_type}",
        paper_ids=paper_ids,
        is_internal=True,
    )
    session.add(chat_session)
    await session.flush()
    snapshot = snapshot_from_settings(request.app.state.settings)
    message = ChatMessage(
        message_id=message_id,
        session_id=chat_session.session_id,
        user_id=user.user_id,
        role="user",
        content=question,
        task_id=task_id,
        status="pending",
    )
    task = TaskRecord(
        task_id=task_id,
        user_id=user.user_id,
        task_type=task_type,
        status="pending",
        stage="queued",
        message_id=message_id,
        resource_id=resource_id,
        request_id=request_id,
        correlation_id=task_id,
    )
    session.add_all(
        [
            message,
            task,
            WorkflowRun(
                task_id=task_id,
                session_id=chat_session.session_id,
                user_id=user.user_id,
                configuration_json=snapshot.model_dump(mode="json"),
                status="pending",
            ),
        ]
    )
    command = StartQaWorkflowCommand(
        request_id=request_id,
        correlation_id=task_id,
        task_id=task_id,
        message_id=message_id,
        user_id=user.user_id,
        session_id=chat_session.session_id,
        paper_ids=paper_ids,
        original_question=question,
        configuration=snapshot,
    )
    return task, message, command


async def _token_view(request: Request, session: AsyncSession, user: User) -> dict[str, object]:
    settings = request.app.state.settings
    access, access_expires_at = issue_access_token(
        user_id=user.user_id,
        role=user.role,
        secret=request.app.state.access_token_secret,
        ttl_seconds=settings.access_token_ttl_seconds,
    )
    refresh = new_refresh_token()
    refresh_expires_at = datetime.now(UTC) + timedelta(seconds=settings.refresh_token_ttl_seconds)
    session.add(
        RefreshToken(
            user_id=user.user_id,
            token_hash=hash_refresh_token(refresh),
            expires_at=refresh_expires_at,
        )
    )
    await session.commit()
    request.state.refresh_token = refresh
    return {
        "access_token": access,
        "token_type": "bearer",
        "access_expires_at": access_expires_at,
        "refresh_expires_at": refresh_expires_at,
    }


def _set_refresh_cookie(response: Response, request: Request) -> None:
    token = getattr(request.state, "refresh_token", None)
    if token:
        response.set_cookie(
            "refresh_token",
            token,
            max_age=request.app.state.settings.refresh_token_ttl_seconds,
            httponly=True,
            secure=request.app.state.settings.is_production,
            samesite="lax",
            path="/api/v1/auth",
        )


@router.post("/auth/register")
async def register(
    body: RegisterInput,
    request: Request,
    response: Response,
    request_id: str = Depends(require_request_id),
    session: AsyncSession = Depends(get_session),
):
    if await session.scalar(select(User).where(User.email == body.email)):
        raise ApiError(409, "EMAIL_ALREADY_REGISTERED", "该邮箱已注册。")
    user = User(
        email=body.email, password_hash=hash_password(body.password), display_name=body.display_name
    )
    session.add(user)
    await session.flush()
    token = await _token_view(request, session, user)
    _set_refresh_cookie(response, request)
    return envelope({"user": user_view(user), "token": token}, request_id)


@router.post("/auth/login")
async def login(
    body: LoginInput,
    request: Request,
    response: Response,
    request_id: str = Depends(require_request_id),
    session: AsyncSession = Depends(get_session),
):
    user = await session.scalar(select(User).where(User.email == body.email.lower()))
    if user is None or not user.is_active or not verify_password(body.password, user.password_hash):
        raise ApiError(401, "INVALID_CREDENTIALS", "邮箱或密码不正确。")
    token = await _token_view(request, session, user)
    _set_refresh_cookie(response, request)
    return envelope({"user": user_view(user), "token": token}, request_id)


@router.post("/auth/refresh")
async def refresh(
    request: Request,
    response: Response,
    request_id: str = Depends(require_request_id),
    session: AsyncSession = Depends(get_session),
):
    raw = request.cookies.get("refresh_token")
    if not raw:
        raise ApiError(401, "UNAUTHENTICATED", "登录状态已失效。")
    record = await session.scalar(
        select(RefreshToken).where(
            RefreshToken.token_hash == hash_refresh_token(raw),
            RefreshToken.revoked_at.is_(None),
            RefreshToken.expires_at > datetime.now(UTC),
        )
    )
    if record is None:
        raise ApiError(401, "UNAUTHENTICATED", "登录状态已失效。")
    user = await session.get(User, record.user_id)
    if user is None or not user.is_active:
        raise ApiError(401, "UNAUTHENTICATED", "登录状态已失效。")
    record.revoked_at = datetime.now(UTC)
    token = await _token_view(request, session, user)
    _set_refresh_cookie(response, request)
    return envelope(token, request_id)


@router.post("/auth/logout")
async def logout(
    request: Request,
    response: Response,
    request_id: str = Depends(require_request_id),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    raw = request.cookies.get("refresh_token")
    if raw:
        token = await session.scalar(
            select(RefreshToken).where(
                RefreshToken.user_id == user.user_id,
                RefreshToken.token_hash == hash_refresh_token(raw),
                RefreshToken.revoked_at.is_(None),
            )
        )
        if token is not None:
            token.revoked_at = datetime.now(UTC)
            await session.commit()
    response.delete_cookie("refresh_token", path="/api/v1/auth")
    return envelope({}, request_id)


@router.get("/auth/me")
async def me(request: Request, user: User = Depends(current_user)):
    return envelope(user_view(user), request.state.request_id)


@router.get("/health")
async def health(request: Request, session: AsyncSession = Depends(get_session)):
    try:
        await session.execute(select(1))
        database = "ok"
    except Exception:
        database = "unavailable"
    return envelope(
        {
            "api": "ok",
            "database": database,
            "redis": "ok" if request.app.state.redis.persistent else "development_fallback",
            "ragflow": "configured"
            if request.app.state.settings.ragflow_base_url
            else "not_configured",
        },
        request.state.request_id,
    )


@router.post("/papers")
async def upload_papers(
    request: Request,
    files: list[UploadFile] = File(...),
    knowledge_base_id: str | None = Form(default=None),
    auto_index: bool = Form(default=True),
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    if not files:
        raise ApiError(422, "VALIDATION_ERROR", "至少上传一个 PDF 文件。")
    uploads: list[tuple[UploadFile, bytes, str]] = []
    max_bytes = request.app.state.settings.max_upload_mb * 1024 * 1024
    for upload in files:
        content = await upload.read()
        name = upload.filename or "paper.pdf"
        if not name.lower().endswith(".pdf") or not content.startswith(b"%PDF-"):
            raise ApiError(422, "INVALID_PAPER_FILE", "仅支持有效的 PDF 文件。")
        if len(content) > max_bytes:
            raise ApiError(413, "FILE_TOO_LARGE", "论文文件超过大小限制。")
        uploads.append((upload, content, hashlib.sha256(content).hexdigest()))
    fingerprint = request_fingerprint(
        {"files": [(item[0].filename, item[2]) for item in uploads], "auto_index": auto_index}
    )
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(202, replay, request_id)

    root = request.app.state.settings.object_storage_path / user.user_id
    root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, object]] = []
    scheduled: list[str] = []
    for upload, content, digest in uploads:
        existing = await session.scalar(
            select(Paper).where(
                Paper.owner_id == user.user_id,
                Paper.content_sha256 == digest,
                Paper.deleted_at.is_(None),
            )
        )
        if existing is not None:
            task = await session.scalar(
                select(TaskRecord)
                .where(TaskRecord.resource_id == existing.paper_id)
                .order_by(TaskRecord.created_at.desc())
            )
            items.append(
                {
                    "paper_id": existing.paper_id,
                    "file_name": existing.file_name,
                    "status": existing.status,
                    "task_id": task.task_id if task else None,
                }
            )
            continue
        suffix = Path(upload.filename or "paper.pdf").suffix.lower()
        object_path = root / f"{uuid4()}{suffix}"
        object_path.write_bytes(content)
        paper = Paper(
            owner_id=user.user_id,
            title=Path(upload.filename or "paper.pdf").stem[:500],
            file_name=upload.filename or "paper.pdf",
            file_path=str(object_path),
            content_sha256=digest,
            file_size_bytes=len(content),
            status="uploaded",
            index_status="pending" if auto_index else "not_indexed",
        )
        session.add(paper)
        await session.flush()
        task = TaskRecord(
            user_id=user.user_id,
            task_type="paper_ingest",
            resource_id=paper.paper_id,
            status="pending",
            stage="queued",
            request_id=request_id,
            correlation_id=new_id(),
        )
        session.add(task)
        await session.flush()
        items.append(
            {
                "paper_id": paper.paper_id,
                "file_name": paper.file_name,
                "status": paper.status,
                "task_id": task.task_id,
            }
        )
        if auto_index:
            scheduled.append(task.task_id)
    data = {"items": items}
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=202,
        response=data,
    )
    await session.commit()
    for task_id in scheduled:
        request.app.state.ingestion_executor.submit(task_id)
    return _json(202, data, request_id)


@router.get("/papers")
async def list_papers(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    status: str | None = None,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    page, page_size = _page(page, page_size)
    where = [Paper.owner_id == user.user_id, Paper.deleted_at.is_(None)]
    if status:
        where.append(Paper.status == status)
    total = await session.scalar(select(func.count()).select_from(Paper).where(*where)) or 0
    rows = await session.scalars(
        select(Paper)
        .where(*where)
        .order_by(Paper.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return envelope(
        {
            "page": page,
            "page_size": page_size,
            "total": total,
            "items": [paper_view(item) for item in rows],
        },
        request.state.request_id,
    )


async def _owned_paper(session: AsyncSession, user_id: str, paper_id: str) -> Paper:
    paper = await session.scalar(
        select(Paper).where(
            Paper.paper_id == paper_id, Paper.owner_id == user_id, Paper.deleted_at.is_(None)
        )
    )
    if paper is None:
        raise ApiError(403, "PAPER_ACCESS_DENIED", "无权访问该论文。")
    return paper


@router.get("/papers/{paper_id}")
async def get_paper(
    paper_id: str,
    request: Request,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    return envelope(
        paper_view(await _owned_paper(session, user.user_id, paper_id)), request.state.request_id
    )


@router.get("/papers/{paper_id}/file")
async def get_paper_file(
    paper_id: str,
    disposition: str = "inline",
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    paper = await _owned_paper(session, user.user_id, paper_id)
    path = Path(paper.file_path)
    if not path.is_file():
        raise ApiError(404, "PAPER_FILE_NOT_FOUND", "论文原文件不存在。")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=paper.file_name,
        content_disposition_type="attachment" if disposition == "attachment" else "inline",
    )


@router.get("/papers/{paper_id}/sections")
async def list_sections(
    paper_id: str,
    request: Request,
    page: int = 1,
    page_size: int = 50,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    page, page_size = _page(page, page_size)
    await _owned_paper(session, user.user_id, paper_id)
    total = (
        await session.scalar(
            select(func.count()).select_from(PaperChunk).where(PaperChunk.paper_id == paper_id)
        )
        or 0
    )
    chunks = await session.scalars(
        select(PaperChunk)
        .where(PaperChunk.paper_id == paper_id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = [
        {
            "section_id": item.chunk_id,
            "paper_id": paper_id,
            "parent_section_id": item.parent_chunk_id,
            "section_title": item.section_title,
            "section_level": 1,
            "section_order": index,
            "page_start": item.page_number,
            "page_end": item.page_number,
            "text": item.content,
        }
        for index, item in enumerate(chunks, start=(page - 1) * page_size + 1)
    ]
    return envelope(
        {"page": page, "page_size": page_size, "total": total, "items": items},
        request.state.request_id,
    )


@router.post("/papers/{paper_id}/retry")
async def retry_paper(
    paper_id: str,
    body: PaperRetryInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    paper = await _owned_paper(session, user.user_id, paper_id)
    fingerprint = request_fingerprint(body.model_dump(mode="json"))
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(202, replay, request_id)
    paper.status = body.stage
    paper.failure = None
    task = TaskRecord(
        user_id=user.user_id,
        task_type="paper_ingest",
        resource_id=paper.paper_id,
        status="pending",
        stage=body.stage,
        request_id=request_id,
        correlation_id=new_id(),
    )
    session.add(task)
    await session.flush()
    data = _accepted(task)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=202,
        response=data,
    )
    await session.commit()
    request.app.state.ingestion_executor.submit(task.task_id, start_stage=body.stage)
    return _json(202, data, request_id)


@router.delete("/papers/{paper_id}")
async def delete_paper(
    paper_id: str,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    paper = await _owned_paper(session, user.user_id, paper_id)
    fingerprint = request_fingerprint({"paper_id": paper_id})
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="DELETE",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(202, replay, request_id)
    paper.status = "deleting"
    paper.index_status = "cancelling"
    task = TaskRecord(
        user_id=user.user_id,
        task_type="paper_cleanup",
        resource_id=paper.paper_id,
        status="pending",
        stage="queued",
        request_id=request_id,
        correlation_id=new_id(),
    )
    session.add(task)
    await session.flush()
    data = _accepted(task)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="DELETE",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=202,
        response=data,
    )
    await session.commit()
    await request.app.state.redis.set_task_state(task.task_id, task_view(task))
    request.app.state.operations_executor.submit(task.task_id)
    return _json(202, data, request_id)


@router.post("/papers/{paper_id}/analyses/{kind}")
async def create_analysis(
    paper_id: str,
    kind: str,
    body: AnalysisInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    if kind not in {"summary", "method", "experiment", "critical-review"}:
        raise ApiError(422, "VALIDATION_ERROR", "不支持的论文分析类型。")
    await _ready_papers(session, user_id=user.user_id, paper_ids=[paper_id])
    fingerprint = request_fingerprint({"kind": kind, **body.model_dump(mode="json")})
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(202, replay, request_id)
    default_questions = {
        "summary": "请基于论文原文给出结构化阅读摘要，并标注关键证据。",
        "method": "请分析论文的方法设计、假设与局限，并标注原文证据。",
        "experiment": "请分析论文的实验设计、结果与证据充分性。",
        "critical-review": "请对论文进行平衡的同行评审，给出优点、问题和证据。",
    }
    task, message, command = await _create_internal_workflow(
        session,
        request=request,
        user=user,
        paper_ids=[paper_id],
        question=body.question or default_questions[kind],
        task_type=f"paper_analysis:{kind}",
        resource_id=paper_id,
        request_id=request_id,
    )
    await session.flush()
    data = _accepted(task, message_id=message.message_id)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=202,
        response=data,
    )
    await session.commit()
    await request.app.state.redis.set_task_state(task.task_id, task_view(task))
    request.app.state.workflow_executor.submit(command)
    return _json(202, data, request_id)


@router.post("/paper-comparisons")
async def compare_papers(
    body: ComparisonInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    await _ready_papers(session, user_id=user.user_id, paper_ids=body.paper_ids)
    fingerprint = request_fingerprint(body.model_dump(mode="json"))
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(202, replay, request_id)
    dimensions = "、".join(body.dimensions)
    question = body.question or f"请比较这些论文在以下维度的异同：{dimensions}。所有结论必须引用对应论文证据。"
    task, message, command = await _create_internal_workflow(
        session,
        request=request,
        user=user,
        paper_ids=body.paper_ids,
        question=question,
        task_type="paper_comparison",
        resource_id=new_id(),
        request_id=request_id,
    )
    await session.flush()
    data = _accepted(task, message_id=message.message_id)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=202,
        response=data,
    )
    await session.commit()
    await request.app.state.redis.set_task_state(task.task_id, task_view(task))
    request.app.state.workflow_executor.submit(command)
    return _json(202, data, request_id)


@router.post("/reading-reports")
async def create_reading_report(
    body: ReadingReportInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    await _ready_papers(session, user_id=user.user_id, paper_ids=body.paper_ids)
    if body.session_id:
        await _owned_session(session, user.user_id, body.session_id)
    fingerprint = request_fingerprint(body.model_dump(mode="json"))
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(202, replay, request_id)
    report = ReadingReport(
        user_id=user.user_id,
        session_id=body.session_id,
        paper_ids=body.paper_ids,
        title=body.title,
        template_key=body.template_key,
    )
    session.add(report)
    await session.flush()
    task = TaskRecord(
        user_id=user.user_id,
        task_type="reading_report",
        resource_id=report.report_id,
        status="pending",
        stage="queued",
        request_id=request_id,
        correlation_id=new_id(),
    )
    session.add(task)
    await session.flush()
    data = _accepted(task)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=202,
        response=data,
    )
    await session.commit()
    await request.app.state.redis.set_task_state(task.task_id, task_view(task))
    request.app.state.operations_executor.submit(task.task_id)
    return _json(202, data, request_id)


@router.get("/reading-reports/{report_id}")
async def get_reading_report(
    report_id: str,
    request: Request,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    report = await session.scalar(
        select(ReadingReport).where(
            ReadingReport.report_id == report_id, ReadingReport.user_id == user.user_id
        )
    )
    if report is None:
        raise ApiError(404, "REPORT_NOT_FOUND", "阅读报告不存在。")
    return envelope(_report_view(report), request.state.request_id)


@router.post("/reading-reports/{report_id}/exports")
async def export_reading_report(
    report_id: str,
    body: ExportInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    report = await session.scalar(
        select(ReadingReport).where(
            ReadingReport.report_id == report_id, ReadingReport.user_id == user.user_id
        )
    )
    if report is None:
        raise ApiError(404, "REPORT_NOT_FOUND", "阅读报告不存在。")
    if report.status != "succeeded" or not report.content_markdown:
        raise ApiError(409, "REPORT_NOT_READY", "阅读报告尚未生成完成。")
    fingerprint = request_fingerprint(body.model_dump(mode="json"))
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(202, replay, request_id)
    task = TaskRecord(
        user_id=user.user_id,
        task_type="report_export",
        resource_id=report_id,
        status="pending",
        stage="queued",
        request_id=request_id,
        correlation_id=new_id(),
        payload_json={"format": body.format},
    )
    session.add(task)
    await session.flush()
    export = ReportExport(report_id=report_id, task_id=task.task_id, format=body.format)
    session.add(export)
    data = _accepted(task)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=202,
        response=data,
    )
    await session.commit()
    await request.app.state.redis.set_task_state(task.task_id, task_view(task))
    request.app.state.operations_executor.submit(task.task_id)
    return _json(202, data, request_id)


@router.get("/reading-reports/{report_id}/exports/{export_id}/file")
async def download_report_export(
    report_id: str,
    export_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    export = await session.scalar(
        select(ReportExport)
        .join(ReadingReport, ReadingReport.report_id == ReportExport.report_id)
        .where(
            ReportExport.export_id == export_id,
            ReportExport.report_id == report_id,
            ReadingReport.user_id == user.user_id,
        )
    )
    if export is None or export.status != "succeeded" or not export.file_path:
        raise ApiError(404, "REPORT_EXPORT_NOT_FOUND", "报告导出文件不存在。")
    path = Path(export.file_path)
    if not path.is_file():
        raise ApiError(404, "REPORT_EXPORT_NOT_FOUND", "报告导出文件不存在。")
    media = {
        "markdown": "text/markdown; charset=utf-8",
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }[export.format]
    return FileResponse(path, media_type=media, filename=f"report-{report_id}.{export.format}")


@router.post("/sessions")
async def create_session(
    body: SessionCreateInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    fingerprint = request_fingerprint(body.model_dump(mode="json"))
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(201, replay, request_id)
    papers = list(
        (
            await session.scalars(
                select(Paper).where(
                    Paper.paper_id.in_(body.paper_ids),
                    Paper.owner_id == user.user_id,
                    Paper.status == "ready",
                    Paper.deleted_at.is_(None),
                )
            )
        ).all()
    )
    if len(papers) != len(set(body.paper_ids)):
        raise ApiError(409, "PAPER_NOT_READY", "所选论文尚未完成解析和索引。")
    item = ChatSession(
        user_id=user.user_id,
        title=body.title or "未命名会话",
        paper_ids=body.paper_ids,
        knowledge_base_id=body.knowledge_base_id,
    )
    session.add(item)
    await session.flush()
    data = session_view(item)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=201,
        response=data,
    )
    await session.commit()
    await session.refresh(item)
    return _json(201, data, request_id)


@router.get("/sessions")
async def list_sessions(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    page, page_size = _page(page, page_size)
    total = (
        await session.scalar(
            select(func.count())
            .select_from(ChatSession)
            .where(ChatSession.user_id == user.user_id, ChatSession.is_internal.is_(False))
        )
        or 0
    )
    rows = await session.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == user.user_id, ChatSession.is_internal.is_(False))
        .order_by(ChatSession.last_message_at.desc().nullslast(), ChatSession.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return envelope(
        {
            "page": page,
            "page_size": page_size,
            "total": total,
            "items": [session_view(item) for item in rows],
        },
        request.state.request_id,
    )


async def _owned_session(session: AsyncSession, user_id: str, session_id: str) -> ChatSession:
    item = await session.scalar(
        select(ChatSession).where(
            ChatSession.session_id == session_id, ChatSession.user_id == user_id
        )
    )
    if item is None:
        raise ApiError(404, "SESSION_NOT_FOUND", "会话不存在。")
    return item


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    body: SessionUpdateInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    fingerprint = request_fingerprint(body.model_dump(mode="json"))
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="PATCH",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(200, replay, request_id)
    item = await _owned_session(session, user.user_id, session_id)
    item.title = body.title
    data = session_view(item)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="PATCH",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=200,
        response=data,
    )
    await session.commit()
    return _json(200, data, request_id)


@router.get("/sessions/{session_id}/messages")
async def list_messages(
    session_id: str,
    request: Request,
    page: int = 1,
    page_size: int = 50,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    page, page_size = _page(page, page_size)
    await _owned_session(session, user.user_id, session_id)
    where = [ChatMessage.session_id == session_id, ChatMessage.user_id == user.user_id]
    total = await session.scalar(select(func.count()).select_from(ChatMessage).where(*where)) or 0
    rows = await session.scalars(
        select(ChatMessage)
        .where(*where)
        .order_by(ChatMessage.created_at)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return envelope(
        {
            "page": page,
            "page_size": page_size,
            "total": total,
            "items": [message_view(item) for item in rows],
        },
        request.state.request_id,
    )


@router.post("/sessions/{session_id}/messages")
async def ask_question(
    session_id: str,
    body: QuestionInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    chat_session = await _owned_session(session, user.user_id, session_id)
    fingerprint = request_fingerprint(body.model_dump(mode="json"))
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(202, replay, request_id)
    paper_ids = body.paper_ids or chat_session.paper_ids
    if not paper_ids:
        raise ApiError(422, "VALIDATION_ERROR", "会话必须关联至少一篇论文。")
    papers = list(
        (
            await session.scalars(
                select(Paper).where(
                    Paper.paper_id.in_(paper_ids),
                    Paper.owner_id == user.user_id,
                    Paper.status == "ready",
                    Paper.deleted_at.is_(None),
                )
            )
        ).all()
    )
    if len(papers) != len(set(paper_ids)):
        raise ApiError(409, "PAPER_NOT_READY", "论文尚未完成解析和索引。", {"paper_ids": paper_ids})
    task_id, message_id = new_id(), new_id()
    message = ChatMessage(
        message_id=message_id,
        session_id=session_id,
        user_id=user.user_id,
        role="user",
        content=body.question,
        task_id=task_id,
        status="pending",
    )
    task = TaskRecord(
        task_id=task_id,
        user_id=user.user_id,
        task_type="qa_workflow",
        status="pending",
        stage="queued",
        message_id=message_id,
        resource_id=session_id,
        request_id=request_id,
        correlation_id=task_id,
    )
    snapshot = snapshot_from_settings(request.app.state.settings)
    run = WorkflowRun(
        task_id=task_id,
        session_id=session_id,
        user_id=user.user_id,
        configuration_json=snapshot.model_dump(mode="json"),
        status="pending",
    )
    session.add_all([message, task, run])
    chat_session.last_message_at = datetime.now(UTC)
    data = _accepted(task, message_id=message_id)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=202,
        response=data,
    )
    await session.commit()
    await request.app.state.redis.set_task_state(task_id, task_view(task))
    request.app.state.workflow_executor.submit(
        StartQaWorkflowCommand(
            request_id=request_id,
            correlation_id=task_id,
            task_id=task_id,
            message_id=message_id,
            user_id=user.user_id,
            session_id=session_id,
            paper_ids=paper_ids,
            original_question=body.question,
            configuration=snapshot,
        )
    )
    return _json(202, data, request_id)


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    request: Request,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    task = await session.scalar(
        select(TaskRecord).where(TaskRecord.task_id == task_id, TaskRecord.user_id == user.user_id)
    )
    if task is None:
        raise ApiError(404, "TASK_NOT_FOUND", "任务不存在。")
    return envelope(task_view(task), request.state.request_id)


@router.get("/messages/{message_id}/details")
async def answer_detail(
    message_id: str,
    request: Request,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    message = await session.scalar(
        select(ChatMessage).where(
            ChatMessage.message_id == message_id, ChatMessage.user_id == user.user_id
        )
    )
    if message is None:
        raise ApiError(404, "MESSAGE_NOT_FOUND", "消息不存在。")
    if message.answer_json is None:
        raise ApiError(
            409,
            "TASK_NOT_COMPLETED",
            "回答仍在生成中。",
            {"task_id": message.task_id, "status": message.status},
        )
    task = await session.get(TaskRecord, message.task_id) if message.task_id else None
    run = (
        await session.scalar(select(WorkflowRun).where(WorkflowRun.task_id == message.task_id))
        if message.task_id
        else None
    )
    return envelope(
        {
            "answer": message.answer_json,
            "workflow_run": {
                "workflow_run_id": run.workflow_run_id if run else None,
                "task_id": message.task_id,
                "session_id": message.session_id,
                "task_type": task.task_type if task else "qa_workflow",
                "status": task.status if task else "succeeded",
                "planned_agents": ["controller", "paper_understanding", "review_a", "review_b"],
                "confidence": message.confidence,
                "started_at": run.started_at if run else None,
                "completed_at": run.completed_at if run else None,
            },
            "agent_results": (run.summary_json or {}).get("agent_results", []) if run else [],
        },
        request.state.request_id,
    )


@router.post("/messages/{message_id}/cancel")
async def cancel_message(
    message_id: str,
    body: CancelInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    message = await session.scalar(
        select(ChatMessage).where(
            ChatMessage.message_id == message_id, ChatMessage.user_id == user.user_id
        )
    )
    if message is None or not message.task_id:
        raise ApiError(404, "MESSAGE_NOT_FOUND", "消息不存在。")
    task = await session.get(TaskRecord, message.task_id)
    assert task is not None
    fingerprint = request_fingerprint(body.model_dump(mode="json"))
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return envelope(replay, request_id)
    if task.status not in {"succeeded", "failed", "cancelled"}:
        await request.app.state.redis.cancel(task.task_id)
        task.stage = "cancelling"
        await session.commit()
        await request.app.state.redis.set_task_state(task.task_id, task_view(task))
    data = task_view(task)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=200,
        response=data,
    )
    await session.commit()
    return envelope(data, request_id)


@router.post("/messages/{message_id}/feedback")
async def feedback(
    message_id: str,
    body: FeedbackInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    fingerprint = request_fingerprint(body.model_dump(mode="json"))
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(200, replay, request_id)
    message = await session.scalar(
        select(ChatMessage).where(
            ChatMessage.message_id == message_id, ChatMessage.user_id == user.user_id
        )
    )
    if message is None:
        raise ApiError(404, "MESSAGE_NOT_FOUND", "消息不存在。")
    item = Feedback(
        message_id=message_id,
        user_id=user.user_id,
        feedback_type=body.feedback_type,
        reason=body.reason,
        tags=body.tags,
    )
    session.add(item)
    await session.flush()
    data = {
        "feedback_id": item.feedback_id,
        "message_id": message_id,
        "feedback_type": item.feedback_type,
    }
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=200,
        response=data,
    )
    await session.commit()
    return _json(200, data, request_id)


@router.get("/messages/{message_id}/events")
async def stream_events(
    message_id: str,
    request: Request,
    after_sequence: int = 0,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    message = await session.scalar(
        select(ChatMessage).where(
            ChatMessage.message_id == message_id, ChatMessage.user_id == user.user_id
        )
    )
    if message is None:
        raise ApiError(404, "MESSAGE_NOT_FOUND", "消息不存在。")
    if after_sequence < 0:
        raise ApiError(422, "VALIDATION_ERROR", "after_sequence 不能为负数。")
    after_event = await request.app.state.redis.after_event_id(
        message_id, request.headers.get("Last-Event-ID")
    )
    sequence = max(after_sequence, after_event or 0)

    async def event_source():
        nonlocal sequence
        while True:
            events = await request.app.state.redis.events_after(message_id, sequence)
            for event in events:
                sequence = event.sequence
                payload = event.model_dump_json()
                yield f"id: {event.event_id}\nevent: {event.event_type}\ndata: {payload}\n\n"
                if event.event_type in {"final", "error"}:
                    return
            latest = await request.app.state.redis.latest_event(message_id)
            if latest is not None and latest.event_type in {"final", "error"}:
                return
            if await request.is_disconnected():
                return
            yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _list_configurations(
    *,
    kind: str,
    request: Request,
    session: AsyncSession,
    page: int,
    page_size: int,
) -> dict[str, object]:
    page, page_size = _page(page, page_size)
    where = ConfigurationVersion.kind == kind
    total = await session.scalar(select(func.count()).select_from(ConfigurationVersion).where(where))
    rows = await session.scalars(
        select(ConfigurationVersion)
        .where(where)
        .order_by(ConfigurationVersion.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return envelope(
        {
            "page": page,
            "page_size": page_size,
            "total": total or 0,
            "items": [_config_view(item) for item in rows],
        },
        request.state.request_id,
    )


async def _get_configuration_response(
    *,
    kind: str,
    configuration_id: str,
    request: Request,
    session: AsyncSession,
) -> JSONResponse:
    item = await _configuration(session, kind=kind, configuration_id=configuration_id)
    data = _config_view(item)
    return _json(200, data, request.state.request_id, headers={"ETag": item.version})


async def _put_configuration(
    *,
    kind: str,
    configuration_id: str,
    body: ConfigUpdateInput,
    request: Request,
    request_id: str,
    if_match: str | None,
    idempotency_key: str,
    user: User,
    session: AsyncSession,
) -> JSONResponse:
    if _contains_secret(body.value):
        raise ApiError(422, "CONFIG_SECRET_FORBIDDEN", "配置中不能保存密钥或令牌。")
    fingerprint = request_fingerprint(
        {"kind": kind, "configuration_id": configuration_id, **body.model_dump(mode="json")}
    )
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="PUT",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(200, replay, request_id, headers={"ETag": str(replay["version"])})

    item = await session.scalar(
        select(ConfigurationVersion).where(
            ConfigurationVersion.configuration_id == configuration_id,
            ConfigurationVersion.kind == kind,
        )
    )
    created = item is None
    if item is None:
        if if_match != "*":
            raise ApiError(
                409,
                "CONFIG_VERSION_CONFLICT",
                "创建配置时 If-Match 必须为 *。",
            )
        item = ConfigurationVersion(
            configuration_id=configuration_id,
            kind=kind,
            version=f"{configuration_id}:1",
            payload_json=body.value,
        )
        session.add(item)
    else:
        if if_match != item.version:
            raise ApiError(
                409,
                "CONFIG_VERSION_CONFLICT",
                "配置已经被其他管理员更新。",
                {"current_version": item.version},
            )
        item.version = _next_config_version(item)
        item.payload_json = body.value

    await session.flush()
    data = _config_view(item)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="PUT",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=201 if created else 200,
        response=data,
    )
    await session.commit()
    return _json(201 if created else 200, data, request_id, headers={"ETag": item.version})


@router.get("/admin/model-configs")
async def list_model_configs(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    return await _list_configurations(
        kind="model", request=request, session=session, page=page, page_size=page_size
    )


@router.get("/admin/prompt-templates")
async def list_prompt_templates(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    return await _list_configurations(
        kind="prompt", request=request, session=session, page=page, page_size=page_size
    )


@router.get("/admin/retrieval-configs")
async def list_retrieval_configs(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    return await _list_configurations(
        kind="retrieval", request=request, session=session, page=page, page_size=page_size
    )


@router.get("/admin/model-configs/{configuration_id}")
async def get_model_config(
    configuration_id: str,
    request: Request,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    return await _get_configuration_response(
        kind="model", configuration_id=configuration_id, request=request, session=session
    )


@router.get("/admin/prompt-templates/{configuration_id}")
async def get_prompt_template(
    configuration_id: str,
    request: Request,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    return await _get_configuration_response(
        kind="prompt", configuration_id=configuration_id, request=request, session=session
    )


@router.get("/admin/retrieval-configs/{configuration_id}")
async def get_retrieval_config(
    configuration_id: str,
    request: Request,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    return await _get_configuration_response(
        kind="retrieval", configuration_id=configuration_id, request=request, session=session
    )


@router.put("/admin/model-configs/{configuration_id}")
async def put_model_config(
    configuration_id: str,
    body: ConfigUpdateInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    if_match: str | None = Header(default=None, alias="If-Match"),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    return await _put_configuration(
        kind="model",
        configuration_id=configuration_id,
        body=body,
        request=request,
        request_id=request_id,
        if_match=if_match,
        idempotency_key=idempotency_key,
        user=user,
        session=session,
    )


@router.put("/admin/prompt-templates/{configuration_id}")
async def put_prompt_template(
    configuration_id: str,
    body: ConfigUpdateInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    if_match: str | None = Header(default=None, alias="If-Match"),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    return await _put_configuration(
        kind="prompt",
        configuration_id=configuration_id,
        body=body,
        request=request,
        request_id=request_id,
        if_match=if_match,
        idempotency_key=idempotency_key,
        user=user,
        session=session,
    )


@router.put("/admin/retrieval-configs/{configuration_id}")
async def put_retrieval_config(
    configuration_id: str,
    body: ConfigUpdateInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    if_match: str | None = Header(default=None, alias="If-Match"),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    return await _put_configuration(
        kind="retrieval",
        configuration_id=configuration_id,
        body=body,
        request=request,
        request_id=request_id,
        if_match=if_match,
        idempotency_key=idempotency_key,
        user=user,
        session=session,
    )


async def _admin_dataset_items(session: AsyncSession, request: Request) -> list[dict[str, object]]:
    rows = await session.execute(
        select(RagMapping.dataset_id, func.count(RagMapping.mapping_id))
        .group_by(RagMapping.dataset_id)
        .order_by(RagMapping.dataset_id)
    )
    counts = {str(dataset_id): int(count) for dataset_id, count in rows}
    settings = request.app.state.settings
    configured = {
        settings.ragflow_user_dataset_id: "用户论文库",
        settings.ragflow_public_dataset_id: "公共评审知识库",
    }
    for dataset_id in configured:
        if dataset_id:
            counts.setdefault(dataset_id, 0)
    return [
        {
            "dataset_id": dataset_id,
            "dataset_name": configured.get(dataset_id) or dataset_id,
            "dataset_version": "ragflow",
            "status": "ready" if count else "empty",
            "document_count": count,
            "chunk_mapping_count": count,
        }
        for dataset_id, count in counts.items()
    ]


@router.get("/admin/knowledge-bases")
async def list_knowledge_bases(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    page, page_size = _page(page, page_size)
    datasets = await _admin_dataset_items(session, request)
    items = [
        {
            "knowledge_base_id": item["dataset_id"],
            "name": item["dataset_name"],
            "status": item["status"],
            "paper_count": item["document_count"],
            "active_index_version": item["dataset_version"],
        }
        for item in datasets
    ]
    return envelope(
        {
            "page": page,
            "page_size": page_size,
            "total": len(items),
            "items": items[(page - 1) * page_size : page * page_size],
        },
        request.state.request_id,
    )


@router.get("/admin/datasets")
async def list_datasets(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    page, page_size = _page(page, page_size)
    items = await _admin_dataset_items(session, request)
    return envelope(
        {
            "page": page,
            "page_size": page_size,
            "total": len(items),
            "items": items[(page - 1) * page_size : page * page_size],
        },
        request.state.request_id,
    )


@router.get("/admin/workflow-runs/{workflow_run_id}")
async def get_workflow_run(
    workflow_run_id: str,
    request: Request,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    run = await session.get(WorkflowRun, workflow_run_id)
    if run is None:
        raise ApiError(404, "WORKFLOW_RUN_NOT_FOUND", "工作流运行记录不存在。")
    task = await session.get(TaskRecord, run.task_id)
    traces = await session.scalars(
        select(TraceRecord)
        .where(TraceRecord.task_id == run.task_id)
        .order_by(TraceRecord.created_at)
    )
    return envelope(
        {
            "workflow_run_id": run.workflow_run_id,
            "task_id": run.task_id,
            "session_id": run.session_id,
            "user_id": run.user_id,
            "status": run.status,
            "task": task_view(task) if task else None,
            "configuration": _public_config_value(run.configuration_json),
            "summary": run.summary_json,
            "traces": [
                {
                    "trace_id": trace.trace_id,
                    "node_name": trace.node_name,
                    "duration_ms": trace.duration_ms,
                    "status": trace.status,
                    "error_code": trace.error_code,
                    "metrics": trace.metrics_json,
                    "created_at": trace.created_at,
                }
                for trace in traces
            ],
        },
        request.state.request_id,
    )


@router.post("/admin/evaluation-runs")
async def create_evaluation_run(
    body: EvaluationInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    idempotency_key: str = Depends(require_idempotency_key),
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    fingerprint = request_fingerprint(body.model_dump(mode="json"))
    replay = await replay_or_raise(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
    )
    if replay is not None:
        return _json(202, replay, request_id)
    task = TaskRecord(
        user_id=user.user_id,
        task_type="evaluation",
        resource_id=body.dataset_id,
        status="pending",
        stage="queued",
        request_id=request_id,
        correlation_id=new_id(),
        payload_json=body.model_dump(mode="json"),
    )
    session.add(task)
    await session.flush()
    data = _accepted(task)
    save_response(
        session,
        user_id=user.user_id,
        key=idempotency_key,
        method="POST",
        path=request.url.path,
        fingerprint=fingerprint,
        status_code=202,
        response=data,
    )
    await session.commit()
    await request.app.state.redis.set_task_state(task.task_id, task_view(task))
    request.app.state.operations_executor.submit(task.task_id)
    return _json(202, data, request_id)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    index = max(0, min(len(values) - 1, int(len(values) * percentile + 0.9999) - 1))
    return sorted(values)[index]


@router.get("/admin/metrics/overview")
async def metrics_overview(
    request: Request,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    tasks = list((await session.scalars(select(TaskRecord))).all())
    workflow_tasks = [
        task
        for task in tasks
        if task.task_type == "qa_workflow"
        or task.task_type.startswith("paper_analysis:")
        or task.task_type == "paper_comparison"
    ]
    terminal = [task for task in tasks if task.status in {"succeeded", "failed", "cancelled"}]
    failures = [task for task in terminal if task.status in {"failed", "cancelled"}]
    latencies = [
        int((task.completed_at - task.started_at).total_seconds() * 1000)
        for task in terminal
        if task.started_at is not None and task.completed_at is not None
    ]
    no_evidence = [
        task
        for task in workflow_tasks
        if (task.error_json or {}).get("code") in {"RAG_NO_EVIDENCE", "QA_EVIDENCE_INVALID"}
    ]
    successful_workflows = [task for task in workflow_tasks if task.status == "succeeded"]
    created_at = [task.created_at for task in tasks]
    return envelope(
        {
            "request_count": len(tasks),
            "question_count": len(workflow_tasks),
            "token_input": 0,
            "token_output": 0,
            "estimated_cost": "0.00000000",
            "latency_p50_ms": _percentile(latencies, 0.5),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "error_rate": len(failures) / len(terminal) if terminal else 0.0,
            "retrieval_metrics": {
                "empty_rate": len(no_evidence) / len(workflow_tasks) if workflow_tasks else 0.0,
            },
            "workflow_metrics": {
                "success_rate": len(successful_workflows) / len(workflow_tasks)
                if workflow_tasks
                else 0.0,
                "refusal_rate": len(no_evidence) / len(workflow_tasks) if workflow_tasks else 0.0,
            },
            "time_range": {
                "start_time": min(created_at).isoformat() if created_at else None,
                "end_time": max(created_at).isoformat() if created_at else None,
                "interval": "all",
            },
        },
        request.state.request_id,
    )
