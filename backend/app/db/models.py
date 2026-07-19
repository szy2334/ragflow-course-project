"""Persistent ownership, workflow, ingestion and audit records."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .base import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid4())


JsonValue = JSON().with_variant(JSONB, "postgresql")


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    display_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(16), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    token_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Paper(Base):
    __tablename__ = "papers"
    paper_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    title: Mapped[str] = mapped_column(String(500))
    file_name: Mapped[str] = mapped_column(String(500))
    file_path: Mapped[str] = mapped_column(String(1024))
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="uploaded", index=True)
    parse_progress: Mapped[float] = mapped_column(Float, default=0.0)
    index_status: Mapped[str] = mapped_column(String(32), default="not_indexed")
    quality_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    failure: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    active_index_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class PaperChunk(Base):
    __tablename__ = "paper_chunks"
    chunk_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(32))
    section_title: Mapped[str] = mapped_column(String(500), default="")
    page_number: Mapped[int] = mapped_column(Integer, default=1)
    source_ref: Mapped[str] = mapped_column(String(1024))
    object_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    parent_chunk_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    content_sha256: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    __table_args__ = (UniqueConstraint("paper_id", "content_sha256", name="uq_chunk_content"),)


class RagMapping(Base):
    __tablename__ = "rag_mappings"
    mapping_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.paper_id"), nullable=True, index=True
    )
    source_chunk_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    dataset_id: Mapped[str] = mapped_column(String(128), index=True)
    document_id: Mapped[str] = mapped_column(String(128), index=True)
    ragflow_chunk_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        UniqueConstraint("source_chunk_id", "content_sha256", name="uq_rag_chunk_map"),
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    session_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    title: Mapped[str] = mapped_column(String(300), default="未命名会话")
    paper_ids: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    knowledge_base_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    message_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.session_id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    task_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TaskRecord(Base):
    __tablename__ = "tasks"
    task_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    task_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    resource_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    message_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    request_id: Mapped[str] = mapped_column(String(80))
    correlation_id: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    record_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    key: Mapped[str] = mapped_column(String(255))
    method: Mapped[str] = mapped_column(String(12))
    path: Mapped[str] = mapped_column(String(500))
    request_hash: Mapped[str] = mapped_column(String(64))
    status_code: Mapped[int] = mapped_column(Integer)
    response_json: Mapped[dict[str, Any]] = mapped_column(JsonValue)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("user_id", "key", "method", "path", name="uq_idempotency"),)


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"
    workflow_run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.session_id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    configuration_json: Mapped[dict[str, Any]] = mapped_column(JsonValue)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Citation(Base):
    __tablename__ = "citations"
    citation_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    message_id: Mapped[str] = mapped_column(ForeignKey("chat_messages.message_id"), index=True)
    evidence_id: Mapped[str] = mapped_column(String(128))
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JsonValue)


class ReviewResult(Base):
    __tablename__ = "review_results"
    review_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    message_id: Mapped[str] = mapped_column(ForeignKey("chat_messages.message_id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(32))
    opinion_json: Mapped[dict[str, Any]] = mapped_column(JsonValue)


class Feedback(Base):
    __tablename__ = "message_feedback"
    feedback_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    message_id: Mapped[str] = mapped_column(ForeignKey("chat_messages.message_id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    feedback_type: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ConfigurationVersion(Base):
    __tablename__ = "configuration_versions"
    configuration_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(128), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JsonValue)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("kind", "version", name="uq_config_version"),)


class TraceRecord(Base):
    __tablename__ = "trace_records"
    trace_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), index=True)
    message_id: Mapped[str] = mapped_column(String(36), index=True)
    request_id: Mapped[str] = mapped_column(String(80))
    correlation_id: Mapped[str] = mapped_column(String(80))
    node_name: Mapped[str] = mapped_column(String(128))
    duration_ms: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
