"""empty message

Revision ID: 50e1f1bc2cac
Revises: 0bca1ba9706e
Create Date: 2019-02-19 17:14:31.702012

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '50e1f1bc2cac'
down_revision = '0bca1ba9706e'
branch_labels = None
depends_on = None

def upgrade():
    """
    This upgrade:
        1. adds the model_metadata.subsets table to track evaluation subsets
        2. adds the subset_hash column to the evaluations table, defaulting to
           '' for existing evaluations (on the assumption that they were over
           the whole cohort)
        3. alters (really, drops and re-adds) the primary key for the
           evaluations tables to include the subset_hash
    """
    # 1. Add subsets table
    op.create_table(
        "subsets",
        sa.Column("subset_hash", sa.String(), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True
        ),
        sa.PrimaryKeyConstraint("subset_hash"),
        schema="model_metadata",
    )

    # 2. Add subset_hash column
    op.add_column(
        "evaluations",
        sa.Column("subset_hash", sa.String(), nullable=False, server_default=""),
        schema="test_results"
    )
    op.add_column(
        "evaluations",
        sa.Column("subset_hash", sa.String(), nullable=False, server_default=""),
        schema="train_results"
    )

    # 3. Alter primary keys
    # Actual triage databases have been observed with different variats of the
    # primary key name in the train_results schema. To ensure that all
    # databases can be appropriately updated, procedural is used to look up
    # the name of the primary key before droppint it.
    op.drop_constraint("evaluations_pkey", "evaluations", schema="test_results")
    op.execute(
        """
        DO
        $body$
        DECLARE _pkey_name varchar(100) := (
            SELECT conname
              FROM pg_catalog.pg_constraint con
                   INNER JOIN pg_catalog.pg_class rel ON rel.oid = con.conrelid
                   INNER JOIN pg_catalog.pg_namespace nsp ON nsp.oid = con.connamespace
             WHERE rel.relname = 'evaluations'
               AND nspname = 'train_results'
               AND contype = 'p'
        );
        BEGIN
            EXECUTE('ALTER TABLE train_results.evaluations DROP CONSTRAINT ' || _pkey_name);
        END
        $body$
        """
    )

    op.create_primary_key(
        constraint_name="evaluations_pkey",
        table_name="evaluations",
        columns=[
            "model_id",
            "subset_hash",
            "evaluation_start_time",
            "evaluation_end_time",
            "as_of_date_frequency",
            "metric",
            "parameter"
        ],
        schema="test_results",
    )
    op.create_primary_key(
        constraint_name="train_evaluations_pkey",
        table_name="evaluations",
        columns=[
            "model_id",
            "subset_hash",
            "evaluation_start_time",
            "evaluation_end_time",
            "as_of_date_frequency",
            "metric",
            "parameter"
        ],
        schema="train_results",
    )


def downgrade():
    """
    This downgrade revereses the steps of the upgrade:
        1. Alters the primary key on the evaluations tables to exclude
           subset_hash
        2. Drops the subset hash columns from the evaluations tables
        3. Drops the model_metadata.subsets table
    """
    # 1. Alter primary keys
    op.drop_constraint("evaluations_pkey", "evaluations", schema="test_results")
    op.drop_constraint("train_evaluations_pkey", "evaluations", schema="train_results")

    op.create_primary_key(
        name="evaluations_pkey",
        table_name="evaluations",
        columns=[
            "model_id",
            "evaluation_start_time",
            "evaluation_end_time",
            "as_of_date_frequency",
            "metric",
            "parameter"
        ],
        schema="test_results",
    )
    op.create_primary_key(
        name="train_evaluations_pkey",
        table_name="evaluations",
        columns=[
            "model_id",
            "evaluation_start_time",
            "evaluation_end_time",
            "as_of_date_frequency",
            "metric",
            "parameter"
        ],
        schema="train_results",
    )

    # 2. Drop subset_hash columns
    op.drop_column("evaluations", "subset_hash", schema="train_results")
    op.drop_column("evaluations", "subset_hash", schema="test_results")

    # 3. Drop subsets table
    op.drop_table("subsets", schema="model_metadata")
