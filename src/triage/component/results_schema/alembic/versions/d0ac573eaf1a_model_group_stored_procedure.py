"""model_group_stored_procedure

Revision ID: d0ac573eaf1a
Revises: 2446a931de7a
Create Date: 2018-06-20 17:44:27.162699

"""
from alembic import op
import os
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd0ac573eaf1a'
down_revision = '2446a931de7a'
branch_labels = None
depends_on = None


def upgrade():
    group_proc_filename = os.path.join(
        os.path.dirname(__file__),
        '../../model_group_stored_procedure.sql'
    )
    with open(group_proc_filename) as fd:
        stmt = fd.read()
        op.execute(stmt)


def downgrade():
    pass
