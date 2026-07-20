from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.ai.schemas import (
    AnswerView,
    ConfigurationSnapshot,
    EvidenceItem,
    ModelConfigSnapshot,
    PersistAnswerCommand,
    StreamEvent,
)
from app.core.config import Settings
from app.db.base import Base, build_engine
from app.db.models import ChatMessage, ChatSession, Citation, TaskRecord, User, WorkflowRun
from app.runtime.adapters import SqlAlchemyAnswerPersistence
from app.runtime.redis_store import RedisRuntime


@pytest.mark.asyncio
async def test_persistence_adapter_commits_answer_before_terminal_event(tmp_path):
    engine = build_engine(f"sqlite+aiosqlite:///{(tmp_path / 'runtime.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        session.add(
            User(
                user_id="user-1",
                email="user@example.test",
                password_hash="scrypt$unused",
                display_name="User",
            )
        )
        await session.flush()
        session.add(ChatSession(session_id="session-1", user_id="user-1", paper_ids=[]))
        await session.flush()
        session.add(
            ChatMessage(
                message_id="message-1",
                session_id="session-1",
                user_id="user-1",
                role="user",
                content="问题",
                task_id="task-1",
                status="running",
            )
        )
        await session.flush()
        session.add_all(
            [
                TaskRecord(
                    task_id="task-1",
                    user_id="user-1",
                    task_type="qa_workflow",
                    status="running",
                    stage="finalizing",
                    message_id="message-1",
                    request_id="request-1",
                    correlation_id="correlation-1",
                ),
                WorkflowRun(
                    task_id="task-1",
                    session_id="session-1",
                    user_id="user-1",
                    configuration_json={},
                ),
            ]
        )
        await session.commit()

    redis = RedisRuntime(Settings(database_url="sqlite+aiosqlite:///unused.db"))
    evidence = EvidenceItem(
        evidence_id="P1",
        source_type="paper",
        paper_id="paper-1",
        document_id="document-1",
        chunk_id="chunk-1",
        content_type="text",
        quote="论文原文。",
        source_uri="paper://paper-1/chunk-1",
        retrieval_score=0.9,
    )
    answer = AnswerView(
        message_id="message-1",
        session_id="session-1",
        task_id="task-1",
        route_type="fact",
        answer="回答。",
        evidences=[evidence],
        confidence=0.9,
        is_refusal=False,
        completed_at=datetime.now(UTC),
    )
    config = ConfigurationSnapshot(
        graph_version="v1",
        prompt_version="v1",
        schema_version="v1",
        model=ModelConfigSnapshot(
            config_version="model-v1",
            base_url="https://model.example.test",
            model="model",
        ),
    )
    await SqlAlchemyAnswerPersistence(sessions, redis).persist(
        PersistAnswerCommand(
            request_id="request-1",
            correlation_id="correlation-1",
            user_id="user-1",
            session_id="session-1",
            task_id="task-1",
            message_id="message-1",
            answer=answer,
            configuration=config,
        )
    )
    async with sessions() as session:
        message = await session.get(ChatMessage, "message-1")
        task = await session.get(TaskRecord, "task-1")
        citations = list(
            await session.scalars(select(Citation).where(Citation.message_id == "message-1"))
        )
    assert message is not None and message.answer_json["answer"] == "回答。"
    assert message.answer_text == "回答。"
    assert message.route_type == "fact"
    assert message.model_version == "model-v1"
    assert message.prompt_version == "v1"
    assert task is not None and task.status == "succeeded"
    assert [citation.evidence_id for citation in citations] == ["P1"]
    assert citations[0].source_text == "论文原文。"
    assert citations[0].document_id == "document-1"
    assert citations[0].chunk_id == "chunk-1"
    assert citations[0].similarity == 0.9

    first = StreamEvent(
        event_id="event-1",
        event_type="status",
        task_id="task-1",
        message_id="message-1",
        session_id="session-1",
        sequence=1,
        timestamp=datetime.now(UTC),
        data={"stage": "finalizing", "label": "Finalizing"},
    )
    await redis.append_event(first)
    assert await redis.after_event_id("message-1", "event-1") == 1
    assert await redis.events_after("message-1", 1) == []
    await engine.dispose()
