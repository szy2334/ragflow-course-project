"""persist native PDF text spans for format-review evidence

Revision ID: 20260721_0005
Revises: 20260721_0004
Create Date: 2026-07-21
"""

from alembic import op
import sqlalchemy as sa

from app.db.models import JsonValue


revision = "20260721_0005"
down_revision = "20260721_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("pdf_text_spans"):
        op.create_table(
            "pdf_text_spans",
            sa.Column("pdf_text_span_id", sa.String(length=36), primary_key=True),
            sa.Column("paper_id", sa.String(length=36), sa.ForeignKey("papers.paper_id"), nullable=False),
            sa.Column(
                "paper_version_id",
                sa.String(length=36),
                sa.ForeignKey("paper_versions.paper_version_id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("span_index", sa.Integer(), nullable=False),
            sa.Column("page_number", sa.Integer(), nullable=False),
            sa.Column("bbox_json", JsonValue, nullable=False),
            sa.Column("page_width_pt", sa.Float(), nullable=False),
            sa.Column("page_height_pt", sa.Float(), nullable=False),
            sa.Column("page_rotation", sa.Integer(), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("raw_font_name", sa.String(length=512), nullable=True),
            sa.Column("font_name", sa.String(length=512), nullable=True),
            sa.Column("font_size_pt", sa.Float(), nullable=True),
            sa.Column("font_flags", sa.Integer(), nullable=True),
            sa.Column("color", sa.Integer(), nullable=True),
            sa.Column("extraction_source", sa.String(length=32), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("paper_version_id", "span_index", name="uq_pdf_text_span_version_index"),
            sa.CheckConstraint("page_number >= 1", name="ck_pdf_text_spans_page"),
            sa.CheckConstraint("span_index >= 0", name="ck_pdf_text_spans_index"),
        )
        inspector = sa.inspect(op.get_bind())

    index_names = {item["name"] for item in inspector.get_indexes("pdf_text_spans")}
    if "ix_pdf_text_spans_paper_id" not in index_names:
        op.create_index("ix_pdf_text_spans_paper_id", "pdf_text_spans", ["paper_id"])
    if "ix_pdf_text_spans_paper_version_id" not in index_names:
        op.create_index("ix_pdf_text_spans_paper_version_id", "pdf_text_spans", ["paper_version_id"])


def downgrade() -> None:
    op.drop_index("ix_pdf_text_spans_paper_version_id", table_name="pdf_text_spans")
    op.drop_index("ix_pdf_text_spans_paper_id", table_name="pdf_text_spans")
    op.drop_table("pdf_text_spans")
