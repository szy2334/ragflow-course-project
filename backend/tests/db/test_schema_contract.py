from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db.base import Base, build_engine
from app.db.models import (
    ChatMessage,
    ChatSession,
    Feedback,
    Paper,
    PaperChunk,
    PaperVersion,
    RagMapping,
    User,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def test_initial_migration_round_trip_matches_orm(tmp_path, monkeypatch):
    database_path = tmp_path / "migration.db"
    database_url = f"sqlite+aiosqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))

    command.upgrade(config, "head")
    sync_engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    schema = inspect(sync_engine)
    assert set(schema.get_table_names()) == set(Base.metadata.tables) | {"alembic_version"}
    assert {item["name"] for item in schema.get_foreign_keys("papers")} >= {
        "fk_papers_current_version"
    }
    assert {item["name"] for item in schema.get_foreign_keys("tasks")} >= {
        "fk_tasks_message"
    }
    sync_engine.dispose()

    command.check(config)
    command.downgrade(config, "base")
    sync_engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    assert inspect(sync_engine).get_table_names() == ["alembic_version"]
    sync_engine.dispose()


async def _new_schema(tmp_path):
    engine = build_engine(f"sqlite+aiosqlite:///{(tmp_path / 'schema.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _paper(owner_id: str, *, paper_id: str = "paper-1") -> Paper:
    return Paper(
        paper_id=paper_id,
        owner_id=owner_id,
        title="Paper",
        file_name="paper.pdf",
        file_path="objects/paper.pdf",
        content_sha256="a" * 64,
        file_size_bytes=128,
    )


def _version(paper_id: str, version_id: str, version_number: int) -> PaperVersion:
    return PaperVersion(
        paper_version_id=version_id,
        paper_id=paper_id,
        version_number=version_number,
        file_name="paper.pdf",
        object_key=f"objects/{version_id}.pdf",
        content_sha256=str(version_number) * 64,
        file_size_bytes=128,
    )


def _chunk(version_id: str, *, row_id: str, source_id: str = "chunk-1") -> PaperChunk:
    return PaperChunk(
        paper_chunk_id=row_id,
        chunk_id=source_id,
        paper_id="paper-1",
        paper_version_id=version_id,
        content=f"content for {version_id}",
        content_type="text",
        page_number=1,
        page_end=1,
        source_ref="paper.pdf#page=1",
        content_sha256=("b" if version_id == "version-1" else "c") * 64,
    )


@pytest.mark.asyncio
async def test_foreign_keys_checks_and_feedback_contract_are_enforced(tmp_path):
    engine, sessions = await _new_schema(tmp_path)
    async with engine.connect() as connection:
        assert await connection.scalar(text("PRAGMA foreign_keys")) == 1

    async with sessions() as session:
        session.add(_paper("missing-user", paper_id="orphan-paper"))
        with pytest.raises(IntegrityError):
            await session.commit()

    async with sessions() as session:
        session.add(
            User(
                user_id="invalid-user",
                email="invalid@example.test",
                password_hash="hash",
                display_name="Invalid",
                role="superuser",
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()

    async with sessions() as session:
        session.add(
            User(
                user_id="user-1",
                email="user@example.test",
                password_hash="hash",
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
                content="Question",
            )
        )
        await session.flush()
        session.add(
            Feedback(
                message_id="message-1",
                user_id="user-1",
                feedback_type="issue",
            )
        )
        await session.commit()

    await engine.dispose()


@pytest.mark.asyncio
async def test_versioned_chunks_and_rag_mappings_are_isolated_and_idempotent(tmp_path):
    engine, sessions = await _new_schema(tmp_path)
    async with sessions() as session:
        session.add(
            User(
                user_id="user-1",
                email="user@example.test",
                password_hash="hash",
                display_name="User",
            )
        )
        await session.flush()
        paper = _paper("user-1")
        session.add(paper)
        await session.flush()
        session.add_all(
            [
                _version("paper-1", "version-1", 1),
                _version("paper-1", "version-2", 2),
            ]
        )
        await session.flush()
        paper.paper_version_id = "version-2"
        session.add_all(
            [
                _chunk("version-1", row_id="row-1"),
                _chunk("version-2", row_id="row-2"),
                RagMapping(
                    mapping_id="map-1",
                    paper_id="paper-1",
                    paper_version_id="version-1",
                    source_chunk_id="chunk-1",
                    dataset_id="dataset-1",
                    document_id="document-1",
                    ragflow_chunk_id="rag-chunk-1",
                    content_sha256="b" * 64,
                ),
                RagMapping(
                    mapping_id="map-2",
                    paper_id="paper-1",
                    paper_version_id="version-2",
                    source_chunk_id="chunk-1",
                    dataset_id="dataset-1",
                    document_id="document-2",
                    ragflow_chunk_id="rag-chunk-2",
                    content_sha256="c" * 64,
                ),
            ]
        )
        await session.commit()

    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(PaperChunk)) == 2
        assert await session.scalar(select(func.count()).select_from(RagMapping)) == 2
        session.add(_chunk("version-2", row_id="duplicate-row"))
        with pytest.raises(IntegrityError):
            await session.commit()

    async with sessions() as session:
        session.add(
            RagMapping(
                mapping_id="duplicate-map",
                paper_id="paper-1",
                paper_version_id="version-2",
                source_chunk_id="chunk-1",
                dataset_id="dataset-1",
                document_id="document-2",
                ragflow_chunk_id="duplicate-rag-chunk",
                content_sha256="c" * 64,
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()

    await engine.dispose()
