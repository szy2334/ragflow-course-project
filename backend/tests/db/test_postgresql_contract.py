import asyncio
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from app.db.base import build_engine
from app.db.models import Base

BACKEND_ROOT = Path(__file__).resolve().parents[2]
POSTGRES_TEST_DATABASE_URL = os.getenv("POSTGRES_TEST_DATABASE_URL")


async def _table_names(database_url: str) -> set[str]:
    engine = build_engine(database_url)
    async with engine.connect() as connection:
        names = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).get_table_names()
        )
    await engine.dispose()
    return set(names)


@pytest.mark.skipif(
    not POSTGRES_TEST_DATABASE_URL,
    reason="POSTGRES_TEST_DATABASE_URL is only provided by PostgreSQL integration CI",
)
def test_postgresql_migration_round_trip_matches_orm(monkeypatch):
    assert POSTGRES_TEST_DATABASE_URL is not None
    monkeypatch.setenv("DATABASE_URL", POSTGRES_TEST_DATABASE_URL)
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))

    command.upgrade(config, "head")
    assert asyncio.run(_table_names(POSTGRES_TEST_DATABASE_URL)) == set(Base.metadata.tables) | {
        "alembic_version"
    }
    command.check(config)
    command.downgrade(config, "base")
    assert asyncio.run(_table_names(POSTGRES_TEST_DATABASE_URL)) == {"alembic_version"}
