"""hash-partitioning predictions tables

Revision ID: 45219f25072b
Revises: a98acf92fd48
Create Date: 2020-08-21 09:29:04.751933

"""
from alembic import op
import sqlalchemy as sa
import os

# revision identifiers, used by Alembic.
revision = '45219f25072b'
down_revision = 'a98acf92fd48'
branch_labels = None
depends_on = None

def upgrade():
    hash_partitioning_filename = os.path.join(
        os.path.dirname(__file__), "../../sql/predictions_hash_partitioning.sql"
    )
    with open(hash_partitioning_filename) as fd:
        stmt = fd.read()
        op.execute(stmt)

def downgrade():
    undo_hash_partitioning_filename = os.path.join(
        os.path.dirname(__file__), "../../sql/undo_predictions_hash_partitioning.sql"
    )
    with open(undo_hash_partitioning_filename) as fd:
        stmt = fd.read()
        op.execute(stmt)
