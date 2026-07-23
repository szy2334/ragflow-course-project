"""allow partial ingestion quality reports

Revision ID: 20260723_0008
Revises: 20260722_0007
Create Date: 2026-07-23
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260723_0008"
down_revision: str | None = "20260722_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("ingestion_quality_reports") as batch_op:
        batch_op.drop_constraint("ck_quality_reports_status", type_="check")
        batch_op.create_check_constraint(
            "ck_quality_reports_status",
            "status IN ('ready', 'partial', 'failed')",
        )


def downgrade() -> None:
    op.execute(
        "UPDATE ingestion_quality_reports SET status = 'ready' WHERE status = 'partial'"
    )
    with op.batch_alter_table("ingestion_quality_reports") as batch_op:
        batch_op.drop_constraint("ck_quality_reports_status", type_="check")
        batch_op.create_check_constraint(
            "ck_quality_reports_status",
            "status IN ('ready', 'failed')",
        )
