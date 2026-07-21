"""Application startup database safety checks."""

from app.main import _may_bootstrap_schema


def test_metadata_bootstrap_is_limited_to_sqlite() -> None:
    assert _may_bootstrap_schema("sqlite+aiosqlite:///./local.db")
    assert _may_bootstrap_schema("sqlite:///./local.db")
    assert not _may_bootstrap_schema(
        "postgresql+asyncpg://paper_review:secret@localhost/paper_review"
    )
