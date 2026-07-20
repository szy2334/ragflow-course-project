"""Public views kept independent from ORM implementation details."""

from typing import Any

from app.db.models import ChatMessage, ChatSession, Paper, User
from app.runtime.adapters import task_view


def user_view(user: User) -> dict[str, Any]:
    return {
        "user_id": user.user_id,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at,
    }


def paper_view(paper: Paper) -> dict[str, Any]:
    return {
        "paper_id": paper.paper_id,
        "owner_id": paper.owner_id,
        "title": paper.title,
        "authors": [],
        "abstract": None,
        "language": None,
        "publication_year": None,
        "doi": None,
        "file_name": paper.file_name,
        "file_size_bytes": paper.file_size_bytes,
        "page_count": paper.page_count,
        "status": paper.status,
        "parse_progress": paper.parse_progress,
        "index_status": paper.index_status,
        "quality_status": paper.quality_status,
        "understanding": paper.understanding_json,
        "failure": paper.failure,
        "active_index_version": paper.active_index_version,
        "created_at": paper.created_at,
        "updated_at": paper.updated_at,
    }


def session_view(item: ChatSession) -> dict[str, Any]:
    return {
        "session_id": item.session_id,
        "user_id": item.user_id,
        "title": item.title,
        "paper_ids": item.paper_ids,
        "knowledge_base_id": item.knowledge_base_id,
        "last_message_at": item.last_message_at,
        "created_at": item.created_at,
    }


def message_view(item: ChatMessage) -> dict[str, Any]:
    return {
        "message_id": item.message_id,
        "session_id": item.session_id,
        "role": item.role,
        "content": item.content,
        "task_id": item.task_id,
        "status": item.status,
        "confidence": item.confidence,
        "created_at": item.created_at,
    }


__all__ = ["message_view", "paper_view", "session_view", "task_view", "user_view"]
