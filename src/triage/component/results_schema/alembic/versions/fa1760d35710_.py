"""empty message

Revision ID: fa1760d35710
Revises: a20104116533
Create Date: 2020-07-16 18:07:58.229213

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'fa1760d35710'
down_revision = 'a20104116533'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('experiments', sa.Column('random_seed', sa.Integer(), nullable=True), schema='triage_metadata')
    # ### end Alembic commands ###


def downgrade():
    op.drop_column('experiments', 'random_seed', schema='triage_metadata')
    # ### end Alembic commands ###
