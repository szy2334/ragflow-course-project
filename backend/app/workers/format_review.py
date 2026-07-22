"""Compatibility task entry point for the dedicated format-review graph."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.format_review import FormatReviewFailure, FormatReviewWorkflowService
from app.runtime.redis_store import RedisRuntime


async def execute_format_review(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    redis: RedisRuntime,
    task_id: str,
) -> None:
    """Preserve the task boundary while delegating business flow to LangGraph."""

    if not settings.llm_base_url or not settings.llm_api_key or not settings.llm_model:
        raise FormatReviewFailure("MODEL_NOT_CONFIGURED", "格式审查模型服务尚未配置。")
    await FormatReviewWorkflowService(settings, sessions, redis).run(task_id)


__all__ = ["FormatReviewFailure", "execute_format_review"]
