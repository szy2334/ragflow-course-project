"""add paper understanding result

Revision ID: 20260720_0002
Revises: 20260720_0001
Create Date: 2026-07-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql

revision: str = "20260720_0002"
down_revision: str | None = "20260720_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD_STATUSES = (
    "'uploaded', 'mineru_parsing', 'ocr_processing', 'cleaning', 'quality_check', "
    "'indexing', 'ready', 'failed', 'deleting', 'deleted'"
)
_NEW_STATUSES = (
    "'uploaded', 'mineru_parsing', 'ocr_processing', 'cleaning', 'quality_check', "
    "'understanding', 'indexing', 'ready', 'failed', 'deleting', 'deleted'"
)


def upgrade() -> None:
    with op.batch_alter_table("papers") as batch_op:
        batch_op.add_column(
            sa.Column(
                "understanding_json",
                sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), "postgresql"),
                nullable=True,
            )
        )
        batch_op.drop_constraint("ck_papers_status", type_="check")
        batch_op.create_check_constraint("ck_papers_status", f"status IN ({_NEW_STATUSES})")


def downgrade() -> None:
    with op.batch_alter_table("papers") as batch_op:
        batch_op.drop_constraint("ck_papers_status", type_="check")
        batch_op.create_check_constraint("ck_papers_status", f"status IN ({_OLD_STATUSES})")
        batch_op.drop_column("understanding_json")
