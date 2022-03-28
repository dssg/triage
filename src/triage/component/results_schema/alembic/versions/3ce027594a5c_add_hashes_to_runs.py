"""add hashes to runs

Revision ID: 3ce027594a5c
Revises: 5dd2ba8222b1
Create Date: 2022-03-25 12:58:38.370271

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3ce027594a5c'
down_revision = '5dd2ba8222b1'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('triage_runs', sa.Column('cohort_table_name', sa.String(), nullable=True), schema='triage_metadata')
    op.add_column('triage_runs', sa.Column('labels_table_name', sa.String(), nullable=True), schema='triage_metadata')
    op.add_column('triage_runs', sa.Column('bias_hash', sa.String(), nullable=True), schema='triage_metadata')


def downgrade():
    op.drop_column('triage_runs', 'bias_hash', schema='triage_metadata')
    op.drop_column('triage_runs', 'labels_table_name', schema='triage_metadata')
    op.drop_column('triage_runs', 'cohort_table_name', schema='triage_metadata')
