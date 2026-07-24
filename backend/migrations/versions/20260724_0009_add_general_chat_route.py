"""add general chat route

Revision ID: 20260724_0009
Revises: 20260723_0008
Create Date: 2026-07-24
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260724_0009"
down_revision: str | None = "20260723_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.drop_constraint("ck_chat_messages_route", type_="check")
        batch_op.create_check_constraint(
            "ck_chat_messages_route",
            "route_type IS NULL OR route_type IN "
            "('fact', 'explain', 'review', 'score', 'follow_up', "
            "'general_chat', 'out_of_scope')",
        )


def downgrade() -> None:
    op.execute(
        "UPDATE chat_messages SET route_type = 'out_of_scope' "
        "WHERE route_type = 'general_chat'"
    )
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.drop_constraint("ck_chat_messages_route", type_="check")
        batch_op.create_check_constraint(
            "ck_chat_messages_route",
            "route_type IS NULL OR route_type IN "
            "('fact', 'explain', 'review', 'score', 'follow_up', 'out_of_scope')",
        )
