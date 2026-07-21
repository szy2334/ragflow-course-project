"""add server-controlled format profiles and rule-level review records

Revision ID: 20260720_0003
Revises: 20260720_0002
Create Date: 2026-07-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql

# ruff: noqa: E501

revision: str = "20260720_0003"
down_revision: str | None = "20260720_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JsonValue = sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "format_profiles",
        sa.Column("format_profile_id", sa.String(length=36), primary_key=True),
        sa.Column("profile_key", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=300), nullable=False),
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("ragflow_dataset_id", sa.String(length=128), nullable=False),
        sa.Column("retrieval_query", sa.Text(), nullable=False),
        sa.Column("rules_json", JsonValue, nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("profile_key", "version", name="uq_format_profile_key_version"),
    )
    op.create_index("ix_format_profiles_profile_key", "format_profiles", ["profile_key"])
    op.create_index("ix_format_profiles_is_active", "format_profiles", ["is_active"])

    op.create_table(
        "format_reviews",
        sa.Column("format_review_id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.user_id"), nullable=False),
        sa.Column("paper_id", sa.String(length=36), sa.ForeignKey("papers.paper_id"), nullable=False),
        sa.Column(
            "format_profile_id",
            sa.String(length=36),
            sa.ForeignKey("format_profiles.format_profile_id"),
            nullable=False,
        ),
        sa.Column("selected_rule_ids", JsonValue, nullable=False),
        sa.Column("profile_snapshot_json", JsonValue, nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("summary_markdown", sa.Text(), nullable=True),
        sa.Column("metrics_json", JsonValue, nullable=False),
        sa.Column("error_json", JsonValue, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_format_reviews_status",
        ),
    )
    op.create_index("ix_format_reviews_user_id", "format_reviews", ["user_id"])
    op.create_index("ix_format_reviews_paper_id", "format_reviews", ["paper_id"])
    op.create_index("ix_format_reviews_format_profile_id", "format_reviews", ["format_profile_id"])
    op.create_index("ix_format_reviews_status", "format_reviews", ["status"])
    op.create_index("ix_format_reviews_user_created", "format_reviews", ["user_id", "created_at"])

    op.create_table(
        "format_review_items",
        sa.Column("format_review_item_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "format_review_id",
            sa.String(length=36),
            sa.ForeignKey("format_reviews.format_review_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rule_id", sa.String(length=128), nullable=False),
        sa.Column("rule_title", sa.String(length=500), nullable=False),
        sa.Column("result", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("finding", sa.Text(), nullable=False),
        sa.Column("suggestion", sa.Text(), nullable=True),
        sa.Column("page_numbers", JsonValue, nullable=False),
        sa.Column("paper_evidence_json", JsonValue, nullable=False),
        sa.Column("standard_evidence_json", JsonValue, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("format_review_id", "rule_id", name="uq_format_review_rule"),
        sa.CheckConstraint(
            "result IN ('compliant', 'non_compliant', 'needs_manual_check', 'not_applicable')",
            name="ck_format_review_item_result",
        ),
        sa.CheckConstraint(
            "severity IN ('info', 'low', 'medium', 'high')",
            name="ck_format_review_item_severity",
        ),
    )
    op.create_index("ix_format_review_items_format_review_id", "format_review_items", ["format_review_id"])


def downgrade() -> None:
    op.drop_index("ix_format_review_items_format_review_id", table_name="format_review_items")
    op.drop_table("format_review_items")
    op.drop_index("ix_format_reviews_user_created", table_name="format_reviews")
    op.drop_index("ix_format_reviews_status", table_name="format_reviews")
    op.drop_index("ix_format_reviews_format_profile_id", table_name="format_reviews")
    op.drop_index("ix_format_reviews_paper_id", table_name="format_reviews")
    op.drop_index("ix_format_reviews_user_id", table_name="format_reviews")
    op.drop_table("format_reviews")
    op.drop_index("ix_format_profiles_is_active", table_name="format_profiles")
    op.drop_index("ix_format_profiles_profile_key", table_name="format_profiles")
    op.drop_table("format_profiles")
