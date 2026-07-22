"""upgrade format review from rule selection to composite findings

Revision ID: 20260721_0004
Revises: 20260720_0003
Create Date: 2026-07-21
"""

from alembic import op
import sqlalchemy as sa


revision = "20260721_0004"
down_revision = "20260720_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    json_value = sa.JSON()
    with op.batch_alter_table("format_profiles") as batch:
        batch.add_column(sa.Column("venue_id", sa.String(length=128), nullable=False, server_default=""))
        batch.add_column(
            sa.Column("allowed_submission_modes", json_value, nullable=False, server_default='["initial_submission"]')
        )
        batch.add_column(sa.Column("shared_document_id", sa.String(length=128), nullable=False, server_default=""))
        batch.add_column(sa.Column("mode_document_mapping_json", json_value, nullable=False, server_default="{}"))
        batch.alter_column("venue_id", server_default=None)
        batch.alter_column("allowed_submission_modes", server_default=None)
        batch.alter_column("shared_document_id", server_default=None)
        batch.alter_column("mode_document_mapping_json", server_default=None)
    with op.batch_alter_table("format_reviews") as batch:
        batch.add_column(sa.Column("submission_mode", sa.String(length=64), nullable=False, server_default="initial_submission"))
        batch.add_column(sa.Column("coverage_report_json", json_value, nullable=False, server_default="{}"))
        batch.add_column(sa.Column("annotation_json", json_value, nullable=False, server_default="{}"))
        batch.alter_column("submission_mode", server_default=None)
        batch.alter_column("coverage_report_json", server_default=None)
        batch.alter_column("annotation_json", server_default=None)
    with op.batch_alter_table("format_review_items") as batch:
        batch.drop_constraint("ck_format_review_item_result", type_="check")
        batch.add_column(sa.Column("category", sa.String(length=64), nullable=False, server_default="body"))
        batch.add_column(sa.Column("aspect", sa.String(length=500), nullable=False, server_default=""))
        batch.add_column(sa.Column("evidence_status", sa.String(length=32), nullable=False, server_default="complete"))
        batch.add_column(sa.Column("annotation_json", json_value, nullable=False, server_default="{}"))
        batch.alter_column("category", server_default=None)
        batch.alter_column("aspect", server_default=None)
        batch.alter_column("evidence_status", server_default=None)
        batch.alter_column("annotation_json", server_default=None)
        batch.create_check_constraint(
            "ck_format_review_item_result",
            "result IN ('compliant', 'non_compliant', 'unverifiable', 'not_applicable')",
        )


def downgrade() -> None:
    with op.batch_alter_table("format_review_items") as batch:
        batch.drop_constraint("ck_format_review_item_result", type_="check")
        batch.create_check_constraint(
            "ck_format_review_item_result",
            "result IN ('compliant', 'non_compliant', 'needs_manual_check', 'not_applicable')",
        )
        batch.drop_column("annotation_json")
        batch.drop_column("evidence_status")
        batch.drop_column("aspect")
        batch.drop_column("category")
    with op.batch_alter_table("format_reviews") as batch:
        batch.drop_column("annotation_json")
        batch.drop_column("coverage_report_json")
        batch.drop_column("submission_mode")
    with op.batch_alter_table("format_profiles") as batch:
        batch.drop_column("mode_document_mapping_json")
        batch.drop_column("shared_document_id")
        batch.drop_column("allowed_submission_modes")
        batch.drop_column("venue_id")
