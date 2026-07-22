"""add V1.1 format-review units and stage-result persistence

Revision ID: 20260722_0006
Revises: 20260721_0005
Create Date: 2026-07-22
"""

from alembic import op
import sqlalchemy as sa


revision = "20260722_0006"
down_revision = "20260721_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    json_value = sa.JSON()
    with op.batch_alter_table("format_reviews") as batch:
        batch.add_column(sa.Column("unit_plan_json", json_value, nullable=False, server_default="[]"))
        batch.add_column(
            sa.Column("synthesis_status", sa.String(length=32), nullable=False, server_default="pending")
        )
        batch.add_column(sa.Column("event_sequence", sa.Integer(), nullable=False, server_default="0"))
        batch.alter_column("unit_plan_json", server_default=None)
        batch.alter_column("synthesis_status", server_default=None)
        batch.alter_column("event_sequence", server_default=None)
        batch.create_check_constraint("ck_format_reviews_event_sequence", "event_sequence >= 0")
    with op.batch_alter_table("format_review_items") as batch:
        batch.add_column(sa.Column("unit_id", sa.String(length=96), nullable=True))
        batch.add_column(sa.Column("unit_position", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("source_stage", sa.String(length=32), nullable=False, server_default="final")
        )
        batch.alter_column("source_stage", server_default=None)
    op.create_index("ix_format_review_items_unit_id", "format_review_items", ["unit_id"])

    op.create_table(
        "format_review_units",
        sa.Column("format_review_unit_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "format_review_id",
            sa.String(length=36),
            sa.ForeignKey("format_reviews.format_review_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("unit_id", sa.String(length=96), nullable=False),
        sa.Column("unit_position", sa.Integer(), nullable=False),
        sa.Column("unit_kind", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("page_range_json", json_value, nullable=False),
        sa.Column("block_ids_json", json_value, nullable=False),
        sa.Column("expected_rule_ids_json", json_value, nullable=False),
        sa.Column("allocated_rule_ids_json", json_value, nullable=False),
        sa.Column("global_rule_ids_json", json_value, nullable=False),
        sa.Column("not_applicable_rule_ids_json", json_value, nullable=False),
        sa.Column("retrieved_rule_ids_json", json_value, nullable=False),
        sa.Column("coverage_json", json_value, nullable=False),
        sa.Column("validated_findings_json", json_value, nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("unit_cycle_count", sa.Integer(), nullable=False),
        sa.Column("retry_budget_remaining", sa.Integer(), nullable=False),
        sa.Column("last_retry_reason", sa.Text(), nullable=True),
        sa.Column("event_sequence", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("format_review_id", "unit_id", name="uq_format_review_unit"),
        sa.UniqueConstraint("format_review_id", "unit_position", name="uq_format_review_unit_position"),
        sa.CheckConstraint("unit_position >= 0", name="ck_format_review_unit_position"),
        sa.CheckConstraint("unit_cycle_count >= 0 AND unit_cycle_count <= 2", name="ck_format_review_unit_cycles"),
        sa.CheckConstraint(
            "retry_budget_remaining >= 0 AND retry_budget_remaining <= 1",
            name="ck_format_review_unit_retry_budget",
        ),
        sa.CheckConstraint("event_sequence >= 0", name="ck_format_review_unit_event_sequence"),
    )
    op.create_index("ix_format_review_units_format_review_id", "format_review_units", ["format_review_id"])
    op.create_index("ix_format_review_units_status", "format_review_units", ["status"])
    op.create_index(
        "ix_format_review_units_review_status",
        "format_review_units",
        ["format_review_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_format_review_units_review_status", table_name="format_review_units")
    op.drop_index("ix_format_review_units_status", table_name="format_review_units")
    op.drop_index("ix_format_review_units_format_review_id", table_name="format_review_units")
    op.drop_table("format_review_units")
    op.drop_index("ix_format_review_items_unit_id", table_name="format_review_items")
    with op.batch_alter_table("format_review_items") as batch:
        batch.drop_column("source_stage")
        batch.drop_column("unit_position")
        batch.drop_column("unit_id")
    with op.batch_alter_table("format_reviews") as batch:
        batch.drop_constraint("ck_format_reviews_event_sequence", type_="check")
        batch.drop_column("event_sequence")
        batch.drop_column("synthesis_status")
        batch.drop_column("unit_plan_json")
