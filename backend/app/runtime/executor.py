"""Asynchronous workflow execution with durable task state and terminal events."""

# ruff: noqa: E501

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai import AiWorkflowService
from app.ai.errors import AiWorkflowError
from app.ai.llm import OpenAICompatibleClient
from app.ai.ports import WorkflowDependencies
from app.ai.schemas import (
    ConfigurationSnapshot,
    ModelConfigSnapshot,
    StartQaWorkflowCommand,
    StreamEvent,
)
from app.core.config import Settings
from app.db.models import ChatMessage, TaskRecord, WorkflowRun

from .adapters import (
    RagFlowRetrievalPort,
    RedisCancellationPort,
    RedisEventSink,
    SqlAlchemyAnswerPersistence,
    SqlAlchemyContextPort,
    SqlAlchemyTraceSink,
    task_view,
)
from .checkpoints import workflow_checkpointer
from .redis_store import RedisRuntime


class WorkflowTaskExecutor:
    """Runs jobs in the API process for development and keeps state in shared stores.

    The runner is deliberately stateless: a deployment can call ``run`` from a
    dedicated worker process with the same database and Redis configuration.
    """

    def __init__(
        self,
        settings: Settings,
        sessions: async_sessionmaker[AsyncSession],
        redis: RedisRuntime,
    ) -> None:
        self._settings = settings
        self._sessions = sessions
        self._redis = redis
        self._running: set[asyncio.Task[None]] = set()

    def submit(self, command: StartQaWorkflowCommand) -> None:
        task = asyncio.create_task(self.run(command), name=f"workflow:{command.task_id}")
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def run(self, command: StartQaWorkflowCommand) -> None:
        if not await self._mark_running(command.task_id):
            await self._fail(command, "TASK_CANCELLED", "任务已取消。", retryable=False)
            return
        if (
            not self._settings.llm_base_url
            or not self._settings.llm_api_key
            or not self._settings.llm_model
        ):
            await self._fail(command, "MODEL_NOT_CONFIGURED", "模型服务尚未配置。", retryable=False)
            return

        dependencies = WorkflowDependencies(
            retrieval=RagFlowRetrievalPort(self._settings, self._sessions),
            context=SqlAlchemyContextPort(self._sessions),
            events=RedisEventSink(self._redis, self._sessions),
            cancellation=RedisCancellationPort(self._redis),
            trace=SqlAlchemyTraceSink(self._sessions),
            persistence=SqlAlchemyAnswerPersistence(self._sessions, self._redis),
        )
        llm = OpenAICompatibleClient(self._settings.llm_api_key)
        try:
            async with workflow_checkpointer(self._settings) as checkpointer:
                result = await AiWorkflowService(llm).run(command, dependencies, checkpointer)
            async with self._sessions() as session:
                run = await session.scalar(
                    select(WorkflowRun).where(WorkflowRun.task_id == command.task_id)
                )
                if run is not None:
                    run.summary_json = {
                        **(run.summary_json or {}),
                        "workflow": result.workflow_summary,
                    }
                    await session.commit()
        except AiWorkflowError as exc:
            # AiWorkflowService emits the terminal error itself. Only durable state remains here.
            message = (
                "任务已取消。" if exc.code == "TASK_CANCELLED" else "工作流未能完成，请稍后重试。"
            )
            await self._mark_failed(command.task_id, exc.code, message)
        except Exception:
            await self._fail(
                command, "AI_WORKFLOW_ERROR", "工作流未能完成，请稍后重试。", retryable=True
            )

    async def _mark_running(self, task_id: str) -> bool:
        async with self._sessions() as session:
            task = await session.get(TaskRecord, task_id)
            if task is None:
                return False
            if await self._redis.is_cancelled(task_id):
                task.status = "cancelled"
                task.stage = "cancelled"
                task.completed_at = datetime.now(UTC)
            else:
                task.status = "running"
                task.stage = "starting"
                task.started_at = datetime.now(UTC)
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))
            return task.status != "cancelled"

    async def _mark_failed(self, task_id: str, code: str, message: str) -> None:
        async with self._sessions() as session:
            task = await session.get(TaskRecord, task_id)
            if task is None or task.status == "succeeded":
                return
            cancelled = code == "TASK_CANCELLED" or await self._redis.is_cancelled(task_id)
            task.status = "cancelled" if cancelled else "failed"
            task.stage = "cancelled" if cancelled else "failed"
            task.error_json = {"code": code, "message": message}
            task.completed_at = datetime.now(UTC)
            message_row = (
                await session.get(ChatMessage, task.message_id) if task.message_id else None
            )
            if message_row is not None:
                message_row.status = task.status
            run = await session.scalar(select(WorkflowRun).where(WorkflowRun.task_id == task_id))
            if run is not None:
                run.status = task.status
                run.completed_at = task.completed_at
            await session.commit()
            await self._redis.set_task_state(task.task_id, task_view(task))

    async def _fail(
        self, command: StartQaWorkflowCommand, code: str, message: str, *, retryable: bool
    ) -> None:
        latest = await self._redis.latest_event(command.message_id)
        if latest is None or latest.event_type not in {"final", "error"}:
            event = StreamEvent(
                event_id=str(uuid4()),
                event_type="error",
                task_id=command.task_id,
                message_id=command.message_id,
                session_id=command.session_id,
                sequence=(latest.sequence if latest else 0) + 1,
                timestamp=datetime.now(UTC),
                data={"code": code, "message": message, "retryable": retryable},
            )
            await RedisEventSink(self._redis, self._sessions).emit(event)
        await self._mark_failed(command.task_id, code, message)


def snapshot_from_settings(settings: Settings) -> ConfigurationSnapshot:
    if not settings.llm_base_url or not settings.llm_model:
        # This object is never sent to a provider until the executor checks configuration.
        base_url = "https://unconfigured.invalid"
        model = "unconfigured"
    else:
        base_url = settings.llm_base_url
        model = settings.llm_model
    return ConfigurationSnapshot(
        graph_version=settings.graph_version,
        prompt_version=settings.prompt_version,
        schema_version=settings.schema_version,
        model=ModelConfigSnapshot(
            config_version=settings.model_config_version,
            base_url=base_url,
            model=model,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=settings.llm_reasoning_effort,
            structured_mode=settings.llm_structured_mode,
        ),
    )
