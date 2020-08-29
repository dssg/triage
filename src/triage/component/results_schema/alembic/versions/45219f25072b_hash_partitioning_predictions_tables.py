"""hash-partitioning predictions tables

Revision ID: 45219f25072b
Revises: a98acf92fd48
Create Date: 2020-08-21 09:29:04.751933

"""
from alembic import op
import os

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)


# revision identifiers, used by Alembic.
revision = '45219f25072b'
down_revision = 'a98acf92fd48'
branch_labels = None
depends_on = None


def get_pg_major_version(op):
    conn = op.get_bind()
    pg_major_version = conn.execute('show server_version').fetchone()[0].split('.')[0]
    logger.debug(f'PostgreSQL major version {pg_major_version}')
    return int(pg_major_version)


def upgrade():

    pg_major_version = get_pg_major_version(op)

    if pg_major_version >= 11:
        logger.info(f'PostgreSQL 11 or greater found (PostgreSQL {pg_major_version}): Using hash partitioning')
        hash_partitioning_filename = os.path.join(
            os.path.dirname(__file__), "../../sql/predictions_hash_partitioning.sql"
        )
        with open(hash_partitioning_filename) as fd:
            stmt = fd.read()
            op.execute(stmt)
    else:
        logger.info(f'No hash partitioning implemented because PostgreSQL 11 or greater not found (using: PostgreSQL {pg_major_version})')


def downgrade():

    pg_major_version = get_pg_major_version(op)

    if pg_major_version >= 11:
        logger.info(f'PostgreSQL 11 or greater found  (PostgreSQL {pg_major_version}): Removing hash partitioning')
        undo_hash_partitioning_filename = os.path.join(
            os.path.dirname(__file__), "../../sql/undo_predictions_hash_partitioning.sql"
        )
        with open(undo_hash_partitioning_filename) as fd:
            stmt = fd.read()
            op.execute(stmt)
    else:
        logger.info(f'No hash partitioning implemented because PostgreSQL 11 or greater not found (using: PostgreSQL {pg_major_version})')
