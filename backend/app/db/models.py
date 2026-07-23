"""Persistent ownership, ingestion, workflow and audit records.

The schema mirrors the requirements, detailed design V1.1 and unified API
contract. PostgreSQL is the production database; JSON uses JSONB there while
remaining SQLite-compatible for local contract tests.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .base import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid4())


JsonValue = JSON().with_variant(JSONB, "postgresql")

PAPER_STATUSES = (
    "uploaded",
    "mineru_parsing",
    "ocr_processing",
    "cleaning",
    "quality_check",
    "understanding",
    "indexing",
    "ready",
    "failed",
    "deleting",
    "deleted",
)
TASK_STATUSES = ("pending", "running", "succeeded", "failed", "cancelled")
MESSAGE_STATUSES = (*TASK_STATUSES, "partial")
TRACE_STATUSES = ("running", "succeeded", "retried", "failed", "skipped")
ROUTE_TYPES = ("fact", "explain", "review", "score", "follow_up", "out_of_scope")
SOURCE_TYPES = ("paper", "standard")
FORMAT_CHECK_RESULTS = ("compliant", "non_compliant", "unverifiable", "not_applicable")
FORMAT_SEVERITIES = ("info", "low", "medium", "high")


def _in_check(column: str, values: tuple[str, ...], name: str) -> CheckConstraint:
    choices = ", ".join(f"'{value}'" for value in values)
    return CheckConstraint(f"{column} IN ({choices})", name=name)


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    display_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(16), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        CheckConstraint("role IN ('user', 'admin')", name="ck_users_role"),
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    token_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Paper(Base):
    __tablename__ = "papers"

    paper_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    # Compatibility/current-version pointer. Immutable artifacts reference paper_versions directly.
    paper_version_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "paper_versions.paper_version_id",
            name="fk_papers_current_version",
            use_alter=True,
        ),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(500))
    file_name: Mapped[str] = mapped_column(String(500))
    file_path: Mapped[str] = mapped_column(String(1024))
    content_sha256: Mapped[str] = mapped_column(String(64))
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="uploaded", index=True)
    parse_progress: Mapped[float] = mapped_column(Float, default=0.0)
    index_status: Mapped[str] = mapped_column(String(32), default="not_indexed")
    quality_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    understanding_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    summary_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_status: Mapped[str] = mapped_column(
        String(32), default="pending", server_default="pending"
    )
    summary_model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    summary_prompt_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    summary_generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    failure: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    active_index_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ingestion_config_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    deletion_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        _in_check("status", PAPER_STATUSES, "ck_papers_status"),
        CheckConstraint(
            "parse_progress >= 0 AND parse_progress <= 1", name="ck_papers_parse_progress"
        ),
        Index("ix_papers_owner_paper", "owner_id", "paper_id"),
        Index("ix_papers_owner_content_hash", "owner_id", "content_sha256"),
        Index("ix_papers_owner_updated", "owner_id", "updated_at"),
    )


class PaperVersion(Base):
    __tablename__ = "paper_versions"

    paper_version_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.paper_id", ondelete="CASCADE"), index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    file_name: Mapped[str] = mapped_column(String(500))
    object_key: Mapped[str] = mapped_column(String(1024))
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    parser_version: Mapped[str] = mapped_column(String(128), default="mineru-v1")
    cleaning_version: Mapped[str] = mapped_column(String(128), default="cleaning-v1")
    chunk_schema_version: Mapped[str] = mapped_column(String(128), default="v1")
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("paper_id", "version_number", name="uq_paper_version_number"),
        CheckConstraint(
            "status IN ('active', 'superseded', 'failed')", name="ck_paper_versions_status"
        ),
    )


class TaskRecord(Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    task_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    resource_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "chat_messages.message_id",
            name="fk_tasks_message",
            use_alter=True,
            ondelete="SET NULL",
        ),
        nullable=True,
        unique=True,
        index=True,
    )
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    request_id: Mapped[str] = mapped_column(String(128), index=True)
    correlation_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        _in_check("status", TASK_STATUSES, "ck_tasks_status"),
        CheckConstraint("progress >= 0 AND progress <= 1", name="ck_tasks_progress"),
        Index("ix_tasks_user_created", "user_id", "created_at"),
    )


class PaperIngestionRun(Base):
    __tablename__ = "paper_ingestion_runs"

    ingestion_run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), unique=True, index=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    paper_version_id: Mapped[str] = mapped_column(
        ForeignKey("paper_versions.paper_version_id"), index=True
    )
    stage: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    parser_version: Mapped[str] = mapped_column(String(128), default="mineru-v1")
    cleaning_version: Mapped[str] = mapped_column(String(128), default="cleaning-v1")
    quality_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)

    __table_args__ = (
        _in_check("status", TASK_STATUSES, "ck_ingestion_runs_status"),
        Index(
            "ix_ingestion_version_config",
            "paper_version_id",
            "parser_version",
            "cleaning_version",
        ),
    )


class ParsedBlockRecord(Base):
    __tablename__ = "parsed_blocks"

    parsed_block_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    paper_version_id: Mapped[str] = mapped_column(
        ForeignKey("paper_versions.paper_version_id", ondelete="CASCADE"), index=True
    )
    block_id: Mapped[str] = mapped_column(String(128))
    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(32), default="text")
    section_title: Mapped[str] = mapped_column(String(500), default="")
    page_number: Mapped[int] = mapped_column(Integer)
    bbox_json: Mapped[list[float] | None] = mapped_column(JsonValue, nullable=True)
    source_ref: Mapped[str] = mapped_column(String(1024))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)

    __table_args__ = (
        UniqueConstraint("paper_version_id", "block_id", name="uq_parsed_block_version"),
        CheckConstraint("page_number >= 1", name="ck_parsed_blocks_page"),
    )


class PdfTextSpanRecord(Base):
    """Native PDF style facts retained for format review, not paper retrieval."""

    __tablename__ = "pdf_text_spans"

    pdf_text_span_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    paper_version_id: Mapped[str] = mapped_column(
        ForeignKey("paper_versions.paper_version_id", ondelete="CASCADE"), index=True
    )
    span_index: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int] = mapped_column(Integer)
    bbox_json: Mapped[list[float]] = mapped_column(JsonValue)
    page_width_pt: Mapped[float] = mapped_column(Float)
    page_height_pt: Mapped[float] = mapped_column(Float)
    page_rotation: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    raw_font_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    font_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    font_size_pt: Mapped[float | None] = mapped_column(Float, nullable=True)
    font_flags: Mapped[int | None] = mapped_column(Integer, nullable=True)
    color: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extraction_source: Mapped[str] = mapped_column(String(32), default="native_pdf")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("paper_version_id", "span_index", name="uq_pdf_text_span_version_index"),
        CheckConstraint("page_number >= 1", name="ck_pdf_text_spans_page"),
        CheckConstraint("span_index >= 0", name="ck_pdf_text_spans_index"),
    )


class MediaObjectRecord(Base):
    __tablename__ = "media_objects"

    media_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    paper_version_id: Mapped[str] = mapped_column(
        ForeignKey("paper_versions.paper_version_id", ondelete="CASCADE"), index=True
    )
    object_id: Mapped[str] = mapped_column(String(128))
    object_type: Mapped[str] = mapped_column(String(32))
    page_number: Mapped[int] = mapped_column(Integer)
    bbox_json: Mapped[list[float] | None] = mapped_column(JsonValue, nullable=True)
    source_ref: Mapped[str] = mapped_column(String(1024))
    image_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    required: Mapped[bool] = mapped_column(Boolean, default=True)
    ocr_status: Mapped[str] = mapped_column(String(32), default="pending")
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_engine: Mapped[str | None] = mapped_column(String(128), nullable=True)
    engines_json: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    raw_response_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    failure_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        UniqueConstraint("paper_version_id", "object_id", name="uq_media_object_version"),
        CheckConstraint("page_number >= 1", name="ck_media_objects_page"),
        CheckConstraint("retry_count >= 0", name="ck_media_objects_retry_count"),
        CheckConstraint(
            "ocr_status IN ('pending', 'processing', 'success', 'partial', 'failed', 'skipped')",
            name="ck_media_objects_ocr_status",
        ),
    )


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    paper_chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    # Python keeps chunk_id for compatibility; the database contract calls it source_chunk_id.
    chunk_id: Mapped[str] = mapped_column("source_chunk_id", String(128))
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    paper_version_id: Mapped[str] = mapped_column(
        ForeignKey("paper_versions.paper_version_id", ondelete="CASCADE"), index=True
    )
    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(32))
    content_role: Mapped[str] = mapped_column(String(64), default="paragraph")
    section_title: Mapped[str] = mapped_column(String(500), default="")
    section_path_json: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    page_number: Mapped[int] = mapped_column("page_start", Integer, default=1)
    page_end: Mapped[int] = mapped_column(Integer, default=1)
    source_ref: Mapped[str] = mapped_column(String(1024))
    object_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    parent_chunk_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prev_chunk_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    next_chunk_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    retrieval_weight: Mapped[float] = mapped_column(Float, default=1.0)
    quality_flags: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    indexable: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    parser_version: Mapped[str] = mapped_column(String(128), default="mineru-v1")
    cleaning_version: Mapped[str] = mapped_column(String(128), default="cleaning-v1")
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "paper_version_id", "source_chunk_id", name="uq_paper_chunk_version_source"
        ),
        CheckConstraint("page_start >= 1 AND page_end >= page_start", name="ck_chunks_pages"),
        CheckConstraint("retrieval_weight >= 0", name="ck_chunks_retrieval_weight"),
        Index("ix_chunks_paper_version_page", "paper_version_id", "page_start"),
        Index("ix_chunks_role_indexable", "content_role", "indexable"),
    )


class IngestionQualityReport(Base):
    __tablename__ = "ingestion_quality_reports"

    report_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), unique=True, index=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    paper_version_id: Mapped[str] = mapped_column(
        ForeignKey("paper_versions.paper_version_id"), index=True
    )
    status: Mapped[str] = mapped_column(String(32))
    indexable_chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    blocking_error_count: Mapped[int] = mapped_column(Integer, default=0)
    expected_mapping_count: Mapped[int] = mapped_column(Integer, default=0)
    mapped_chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    mapping_failure_count: Mapped[int] = mapped_column(Integer, default=0)
    report_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        CheckConstraint(
            "status IN ('ready', 'partial', 'failed')", name="ck_quality_reports_status"
        ),
        CheckConstraint(
            "indexable_chunk_count >= 0 AND blocking_error_count >= 0 "
            "AND expected_mapping_count >= 0 AND mapped_chunk_count >= 0 "
            "AND mapping_failure_count >= 0",
            name="ck_quality_reports_counts",
        ),
    )


class RagMapping(Base):
    __tablename__ = "rag_mappings"

    mapping_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.paper_id"), nullable=True, index=True
    )
    paper_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("paper_versions.paper_version_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    source_chunk_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dataset_id: Mapped[str] = mapped_column(String(128), index=True)
    document_id: Mapped[str] = mapped_column(String(128), index=True)
    ragflow_chunk_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="ready")
    failure_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "paper_version_id",
            "source_chunk_id",
            "content_sha256",
            name="uq_rag_chunk_map",
        ),
        CheckConstraint(
            "status IN ('pending', 'ready', 'failed', 'deleted')", name="ck_rag_mappings_status"
        ),
        Index("ix_rag_document_chunk", "dataset_id", "document_id", "ragflow_chunk_id"),
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    title: Mapped[str] = mapped_column(String(300), default="未命名会话")
    # Ordered immutable snapshot retained for API compatibility and reproducibility.
    paper_ids: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    knowledge_base_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    is_internal: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        Index("ix_sessions_user_last_message", "user_id", "last_message_at"),
        Index("ix_sessions_user_updated", "user_id", "updated_at"),
    )


class SessionPaper(Base):
    __tablename__ = "session_papers"

    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), primary_key=True
    )
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("session_id", "position", name="uq_session_paper_position"),
        CheckConstraint("position >= 0", name="ck_session_papers_position"),
        Index("ix_session_papers_paper", "paper_id", "session_id"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    message_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    role: Mapped[str] = mapped_column(String(16), default="user")
    content: Mapped[str] = mapped_column(Text)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(36), nullable=True, unique=True, index=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    route_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    retrieval_config_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    answer_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        CheckConstraint("role IN ('user')", name="ck_chat_messages_role"),
        CheckConstraint(
            "status IS NULL OR status IN "
            "('pending', 'running', 'succeeded', 'failed', 'cancelled', 'partial')",
            name="ck_chat_messages_status",
        ),
        CheckConstraint(
            "route_type IS NULL OR route_type IN "
            "('fact', 'explain', 'review', 'score', 'follow_up', 'out_of_scope')",
            name="ck_chat_messages_route",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_chat_messages_confidence",
        ),
        Index("ix_messages_session_created", "session_id", "created_at"),
    )


class Citation(Base):
    __tablename__ = "citations"

    citation_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("chat_messages.message_id", ondelete="CASCADE"), index=True
    )
    evidence_id: Mapped[str] = mapped_column(String(128))
    source_type: Mapped[str] = mapped_column(String(16))
    paper_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    document_id: Mapped[str] = mapped_column(String(128), index=True)
    chunk_id: Mapped[str] = mapped_column(String(128), index=True)
    content_type: Mapped[str] = mapped_column(String(32))
    source_text: Mapped[str] = mapped_column(Text)
    section_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    similarity: Mapped[float] = mapped_column(Float)
    source_uri: Mapped[str] = mapped_column(String(2048))
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JsonValue)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("message_id", "evidence_id", name="uq_citation_message_evidence"),
        _in_check("source_type", SOURCE_TYPES, "ck_citations_source_type"),
        CheckConstraint("similarity >= 0", name="ck_citations_similarity"),
    )


class ReviewResult(Base):
    __tablename__ = "review_results"

    review_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("chat_messages.message_id", ondelete="CASCADE"), index=True
    )
    reviewer: Mapped[str] = mapped_column(String(32))
    dimension: Mapped[str | None] = mapped_column(String(128), nullable=True)
    position: Mapped[str | None] = mapped_column(String(32), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_judgement: Mapped[str | None] = mapped_column(Text, nullable=True)
    opinion_json: Mapped[dict[str, Any]] = mapped_column(JsonValue)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("message_id", "reviewer", name="uq_review_message_reviewer"),
        CheckConstraint("reviewer IN ('review_a', 'review_b')", name="ck_review_reviewer"),
        CheckConstraint(
            "position IS NULL OR position IN ('critical', 'supportive', 'mixed')",
            name="ck_review_position",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_review_confidence",
        ),
    )


class Feedback(Base):
    __tablename__ = "message_feedback"

    feedback_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("chat_messages.message_id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    feedback_type: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        CheckConstraint(
            "feedback_type IN ('like', 'dislike', 'issue')", name="ck_feedback_type"
        ),
    )


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

    __table_args__ = (
        UniqueConstraint("user_id", "key", "method", "path", name="uq_idempotency"),
        CheckConstraint("status_code >= 100 AND status_code <= 599", name="ck_idempotency_status"),
    )


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

    __table_args__ = (
        _in_check("status", TASK_STATUSES, "ck_workflow_runs_status"),
    )


class ConfigurationVersion(Base):
    __tablename__ = "configuration_versions"

    configuration_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(128), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JsonValue)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        UniqueConstraint("kind", "version", name="uq_config_version"),
        CheckConstraint(
            "kind IN ('model', 'prompt', 'retrieval', 'standard')", name="ck_config_kind"
        ),
    )


class ConfigurationRevision(Base):
    __tablename__ = "configuration_revisions"

    revision_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    configuration_id: Mapped[str] = mapped_column(
        ForeignKey("configuration_versions.configuration_id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(128))
    payload_json: Mapped[dict[str, Any]] = mapped_column(JsonValue)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("configuration_id", "version", name="uq_config_revision_version"),
    )


class FormatProfile(Base):
    """A server-controlled manuscript-format standard and its RAGFlow source."""

    __tablename__ = "format_profiles"

    format_profile_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    # A profile key can have multiple immutable versions.  The composite
    # (profile_key, version) constraint below is the uniqueness boundary.
    profile_key: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(300))
    version: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Kept server-side only; normal users select a profile ID, never a dataset ID.
    ragflow_dataset_id: Mapped[str] = mapped_column(String(128))
    retrieval_query: Mapped[str] = mapped_column(Text)
    # The following fields are server controlled.  They isolate a venue/version
    # dataset and prevent a browser from selecting arbitrary RAGFlow documents.
    venue_id: Mapped[str] = mapped_column(String(128), default="")
    allowed_submission_modes: Mapped[list[str]] = mapped_column(
        JsonValue, default=lambda: ["initial_submission"]
    )
    shared_document_id: Mapped[str] = mapped_column(String(128), default="")
    mode_document_mapping_json: Mapped[dict[str, str]] = mapped_column(JsonValue, default=dict)
    rules_json: Mapped[list[dict[str, Any]]] = mapped_column(JsonValue, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        UniqueConstraint("profile_key", "version", name="uq_format_profile_key_version"),
    )


class FormatReview(Base):
    """A durable format-compliance review of one owned paper."""

    __tablename__ = "format_reviews"

    format_review_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    format_profile_id: Mapped[str] = mapped_column(
        ForeignKey("format_profiles.format_profile_id"), index=True
    )
    submission_mode: Mapped[str] = mapped_column(String(64), default="initial_submission")
    selected_rule_ids: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    profile_snapshot_json: Mapped[dict[str, Any]] = mapped_column(JsonValue)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    summary_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    coverage_report_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    annotation_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    # V1.1 keeps the immutable block plan with the review so a resumed worker
    # and the UI reason about the exact same ordering and scope allocation.
    unit_plan_json: Mapped[list[dict[str, Any]]] = mapped_column(JsonValue, default=list)
    synthesis_status: Mapped[str] = mapped_column(String(32), default="pending")
    event_sequence: Mapped[int] = mapped_column(Integer, default=0)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        _in_check("status", TASK_STATUSES, "ck_format_reviews_status"),
        CheckConstraint("event_sequence >= 0", name="ck_format_reviews_event_sequence"),
        Index("ix_format_reviews_user_created", "user_id", "created_at"),
    )


class FormatReviewItem(Base):
    """One category/aspect finding, including both manuscript and standard evidence."""

    __tablename__ = "format_review_items"

    format_review_item_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    format_review_id: Mapped[str] = mapped_column(
        ForeignKey("format_reviews.format_review_id", ondelete="CASCADE"), index=True
    )
    unit_id: Mapped[str | None] = mapped_column(String(96), nullable=True, index=True)
    unit_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_stage: Mapped[str] = mapped_column(String(32), default="final")
    rule_id: Mapped[str] = mapped_column(String(128))
    rule_title: Mapped[str] = mapped_column(String(500))
    category: Mapped[str] = mapped_column(String(64), default="body")
    aspect: Mapped[str] = mapped_column(String(500), default="")
    result: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(16), default="info")
    evidence_status: Mapped[str] = mapped_column(String(32), default="complete")
    finding: Mapped[str] = mapped_column(Text)
    suggestion: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_numbers: Mapped[list[int]] = mapped_column(JsonValue, default=list)
    paper_evidence_json: Mapped[list[dict[str, Any]]] = mapped_column(JsonValue, default=list)
    standard_evidence_json: Mapped[list[dict[str, Any]]] = mapped_column(JsonValue, default=list)
    annotation_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("format_review_id", "rule_id", name="uq_format_review_rule"),
        _in_check("result", FORMAT_CHECK_RESULTS, "ck_format_review_item_result"),
        _in_check("severity", FORMAT_SEVERITIES, "ck_format_review_item_severity"),
    )


class FormatReviewUnit(Base):
    """One durable V1.1 review unit and its deterministic rule allocation."""

    __tablename__ = "format_review_units"

    format_review_unit_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    format_review_id: Mapped[str] = mapped_column(
        ForeignKey("format_reviews.format_review_id", ondelete="CASCADE"), index=True
    )
    unit_id: Mapped[str] = mapped_column(String(96))
    unit_position: Mapped[int] = mapped_column(Integer)
    unit_kind: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(500))
    page_range_json: Mapped[list[int]] = mapped_column(JsonValue, default=list)
    block_ids_json: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    expected_rule_ids_json: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    allocated_rule_ids_json: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    global_rule_ids_json: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    not_applicable_rule_ids_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JsonValue, default=list
    )
    retrieved_rule_ids_json: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    coverage_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    validated_findings_json: Mapped[list[dict[str, Any]]] = mapped_column(JsonValue, default=list)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    unit_cycle_count: Mapped[int] = mapped_column(Integer, default=0)
    retry_budget_remaining: Mapped[int] = mapped_column(Integer, default=1)
    last_retry_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_sequence: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        UniqueConstraint("format_review_id", "unit_id", name="uq_format_review_unit"),
        UniqueConstraint(
            "format_review_id", "unit_position", name="uq_format_review_unit_position"
        ),
        CheckConstraint("unit_position >= 0", name="ck_format_review_unit_position"),
        CheckConstraint(
            "unit_cycle_count >= 0 AND unit_cycle_count <= 2", name="ck_format_review_unit_cycles"
        ),
        CheckConstraint(
            "retry_budget_remaining >= 0 AND retry_budget_remaining <= 1",
            name="ck_format_review_unit_retry_budget",
        ),
        CheckConstraint("event_sequence >= 0", name="ck_format_review_unit_event_sequence"),
        Index("ix_format_review_units_review_status", "format_review_id", "status"),
    )


class TraceRecord(Base):
    __tablename__ = "trace_records"

    trace_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), index=True)
    message_id: Mapped[str | None] = mapped_column(
        ForeignKey("chat_messages.message_id", ondelete="SET NULL"), nullable=True, index=True
    )
    request_id: Mapped[str] = mapped_column(String(128), index=True)
    correlation_id: Mapped[str] = mapped_column(String(128), index=True)
    node_name: Mapped[str] = mapped_column(String(128))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32))
    input_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    output_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        _in_check("status", TRACE_STATUSES, "ck_trace_records_status"),
        CheckConstraint("duration_ms >= 0", name="ck_trace_duration"),
        Index("ix_traces_request_created", "request_id", "created_at"),
    )


class QaEvaluation(Base):
    __tablename__ = "qa_evaluations"

    evaluation_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.task_id"), nullable=True, unique=True, index=True
    )
    message_id: Mapped[str | None] = mapped_column(
        ForeignKey("chat_messages.message_id", ondelete="SET NULL"), nullable=True, index=True
    )
    question_type: Mapped[str] = mapped_column(String(32))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_ids: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    verdict: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retrieval_result_json: Mapped[dict[str, Any] | None] = mapped_column(JsonValue, nullable=True)
    validator_result_json: Mapped[dict[str, Any]] = mapped_column(JsonValue, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReadingReport(Base):
    __tablename__ = "reading_reports"

    report_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("chat_sessions.session_id"), nullable=True
    )
    paper_ids: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    title: Mapped[str] = mapped_column(String(300))
    template_key: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    content_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    claims_json: Mapped[list[dict[str, Any]]] = mapped_column(JsonValue, default=list)
    evidence_ids: Mapped[list[str]] = mapped_column(JsonValue, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        _in_check("status", TASK_STATUSES, "ck_reading_reports_status"),
        Index("ix_reading_reports_user_created", "user_id", "created_at"),
    )


class ReadingReportPaper(Base):
    __tablename__ = "reading_report_papers"

    report_id: Mapped[str] = mapped_column(
        ForeignKey("reading_reports.report_id", ondelete="CASCADE"), primary_key=True
    )
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("report_id", "position", name="uq_report_paper_position"),
        CheckConstraint("position >= 0", name="ck_report_papers_position"),
    )


class ReportExport(Base):
    __tablename__ = "report_exports"

    export_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    report_id: Mapped[str] = mapped_column(ForeignKey("reading_reports.report_id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), unique=True, index=True)
    format: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint("format IN ('markdown', 'pdf', 'docx')", name="ck_report_exports_format"),
        _in_check("status", TASK_STATUSES, "ck_report_exports_status"),
    )
