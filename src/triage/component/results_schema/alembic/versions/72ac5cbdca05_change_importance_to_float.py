"""Change importance to float

Revision ID: 72ac5cbdca05
Revises: 264245ddfce2
Create Date: 2017-09-01 14:31:09.302828

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "72ac5cbdca05"
down_revision = "264245ddfce2"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column(
        table_name="individual_importances",
        column_name="importance_score",
        type_=sa.Float(),
        schema="results",
        postgresql_using="importance_score::double precision",
    )


def downgrade():
    op.alter_column(
        table_name="individual_importances",
        column_name="importance_score",
        type_=sa.Text(),
        schema="results",
    )
