"""add label_value to prodcution table

Revision ID: 264786a9fe85
Revises: 1b990cbc04e4
Create Date: 2019-02-26 13:17:05.365654

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '264786a9fe85'
down_revision = '1b990cbc04e4'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("list_predictions", schema="production")
    op.create_table(
        "list_predictions",
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("as_of_date", sa.DateTime(), nullable=False),
        sa.Column("score", sa.Numeric(), nullable=True),
        sa.Column('label_value', sa.Integer, nullable=True),
        sa.Column("rank_abs", sa.Integer(), nullable=True),
        sa.Column("rank_pct", sa.Float(), nullable=True),
        sa.Column("matrix_uuid", sa.Text(), nullable=True),
        sa.Column("test_label_timespan", sa.Interval(), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["triage_metadata.models.model_id"]),
        sa.PrimaryKeyConstraint("model_id", "entity_id", "as_of_date"),
        schema="production",
    )


def downgrade():
    op.drop_table("list_predictions", schema="production")
    op.create_table(
        "list_predictions",
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("as_of_date", sa.DateTime(), nullable=False),
        sa.Column("score", sa.Numeric(), nullable=True),
        sa.Column("rank_abs", sa.Integer(), nullable=True),
        sa.Column("rank_pct", sa.Float(), nullable=True),
        sa.Column("matrix_uuid", sa.Text(), nullable=True),
        sa.Column("test_label_timespan", sa.Interval(), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["triage_metadata.models.model_id"]),
        sa.PrimaryKeyConstraint("model_id", "entity_id", "as_of_date"),
        schema="results",
    )

