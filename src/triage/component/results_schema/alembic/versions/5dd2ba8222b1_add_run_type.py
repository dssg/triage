"""add run_type

Revision ID: 5dd2ba8222b1
Revises: 079a74c15e8b
Create Date: 2021-07-22 23:53:04.043651

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '5dd2ba8222b1'
down_revision = '079a74c15e8b'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('experiment_runs', sa.Column('run_type', sa.Text(), nullable=True), schema='triage_metadata')

    op.add_column('experiment_runs', sa.Column('run_hash', sa.Text(), nullable=True), schema='triage_metadata')
    op.drop_column('experiment_runs', 'experiment_hash', schema='triage_metadata')
    op.execute("ALTER TABLE triage_metadata.experiment_runs RENAME TO triage_runs")

    op.create_table('retrain',
        sa.Column('retrain_hash', sa.Text(), nullable=False),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('prediction_date', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('retrain_hash'),
        schema='triage_metadata',
    )

    op.alter_column('models', 'built_in_experiment_run', nullable=False, new_column_name='built_in_triage_run', schema='triage_metadata')
    op.drop_column('models', 'built_by_experiment', schema='triage_metadata')

    op.create_table('retrain_models',
        sa.Column('retrain_hash', sa.String(), nullable=False),
        sa.Column('model_hash', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['retrain_hash'], ['triage_metadata.retrain.retrain_hash'], ),
        sa.PrimaryKeyConstraint('retrain_hash', 'model_hash'),
        schema='triage_metadata'
    )


def downgrade():
    op.execute("ALTER TABLE triage_metadata.triage_runs RENAME TO experiment_runs")
    op.drop_column('experiment_runs', 'run_type', schema='triage_metadata')
    op.drop_column('experiment_runs', 'run_hash', schema='triage_metadata')
    op.drop_table('retrain_models', schema='triage_metadata')
    op.drop_table('retrain', schema='triage_metadata')
    op.add_column('models', sa.Column('built_by_experiment', sa.Text(), nullable=True), schema='triage_metadata')
    op.alter_column('models', 'built_in_triage_run', nullable=False, new_column_name='built_in_experiment_run', schema='triage_metadata')
