"""The /api/v1 REST and SSE contract."""

# ruff: noqa: B008, E501

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.schemas import StartQaWorkflowCommand
from app.api.dependencies import (
    current_user,
    get_session,
    require_idempotency_key,
    require_request_id,
)
from app.api.schemas import (
    CancelInput,
    FeedbackInput,
    LoginInput,
    PaperRetryInput,
    QuestionInput,
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
    Feedback,
    Paper,
    PaperChunk,
    RefreshToken,
    TaskRecord,
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


def _json(status_code: int, data: object, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content=jsonable_encoder(envelope(data, request_id))
    )


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


@router.post("/sessions")
async def create_session(
    body: SessionCreateInput,
    request: Request,
    request_id: str = Depends(require_request_id),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
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
    await session.commit()
    await session.refresh(item)
    return _json(201, session_view(item), request_id)


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
            select(func.count()).select_from(ChatSession).where(ChatSession.user_id == user.user_id)
        )
        or 0
    )
    rows = await session.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == user.user_id)
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
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    item = await _owned_session(session, user.user_id, session_id)
    item.title = body.title
    await session.commit()
    return envelope(session_view(item), request_id)


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
    item = Feedback(
        message_id=message_id,
        user_id=user.user_id,
        feedback_type=body.feedback_type,
        reason=body.reason,
        tags=body.tags,
    )
    session.add(item)
    await session.commit()
    return envelope(
        {
            "feedback_id": item.feedback_id,
            "message_id": message_id,
            "feedback_type": item.feedback_type,
        },
        request_id,
    )


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
