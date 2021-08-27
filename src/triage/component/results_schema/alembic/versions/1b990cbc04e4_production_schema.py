"""empty message

Revision ID: 1b990cbc04e4
Revises: 0bca1ba9706e
Create Date: 2019-02-20 16:41:22.810452

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1b990cbc04e4'
down_revision = '45219f25072b'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE SCHEMA IF NOT EXISTS production")
    op.execute("ALTER TABLE triage_metadata.list_predictions SET SCHEMA production;")


def downgrade():
    op.execute("ALTER TABLE production.list_predictions SET SCHEMA triage_metadata;")
    op.execute("DROP SCHEMA IF EXISTS production")
