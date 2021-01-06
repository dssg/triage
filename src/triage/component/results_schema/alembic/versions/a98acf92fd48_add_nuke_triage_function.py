"""add nuke triage function

Revision ID: a98acf92fd48
Revises: 4ae804cc0977
Create Date: 2020-07-19 01:46:02.751987

"""
from alembic import op
import os

# revision identifiers, used by Alembic.
revision = 'a98acf92fd48'
down_revision = '4ae804cc0977'
branch_labels = None
depends_on = None


def upgrade():
    nuke_triage_filename = os.path.join(
        os.path.dirname(__file__), "../../sql/nuke_triage.sql"
    )
    with open(nuke_triage_filename) as fd:
        stmt = fd.read()
        op.execute(stmt)



def downgrade():
    pass
