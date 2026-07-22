"""add versioned upload-time paper summary

Revision ID: 20260722_0007
Revises: 20260722_0006
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260722_0007"
down_revision: str | None = "20260722_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("papers") as batch_op:
        batch_op.add_column(sa.Column("summary_markdown", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "summary_status",
                sa.String(length=32),
                nullable=False,
                server_default="pending",
            )
        )
        batch_op.add_column(
            sa.Column("summary_model_version", sa.String(length=128), nullable=True)
        )
        batch_op.add_column(
            sa.Column("summary_prompt_version", sa.String(length=128), nullable=True)
        )
        batch_op.add_column(
            sa.Column("summary_generated_at", sa.DateTime(timezone=True), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("papers") as batch_op:
        batch_op.drop_column("summary_generated_at")
        batch_op.drop_column("summary_prompt_version")
        batch_op.drop_column("summary_model_version")
        batch_op.drop_column("summary_status")
        batch_op.drop_column("summary_markdown")
