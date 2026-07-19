"""Persistent LangGraph checkpointer factory for PostgreSQL deployments."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from app.core.config import Settings


@asynccontextmanager
async def workflow_checkpointer(settings: Settings) -> AsyncIterator[Any | None]:
    """Yield a PostgreSQL-backed saver, or None for explicitly local development."""
    if not settings.database_url.startswith("postgresql+"):
        yield None
        return
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:  # pragma: no cover - dependency guards deployment mistakes
        raise RuntimeError(
            "langgraph-checkpoint-postgres is required for PostgreSQL runtime"
        ) from exc
    connection = settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    async with AsyncPostgresSaver.from_conn_string(connection) as saver:
        await saver.setup()
        yield saver
