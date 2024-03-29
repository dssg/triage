"""Break ties in list predictions

Revision ID: ce5b50ffa8e2
Revises: 264786a9fe85
Create Date: 2021-01-08 21:59:13.403934

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'ce5b50ffa8e2'
down_revision = '264786a9fe85'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('list_predictions', sa.Column('rank_abs_with_ties', sa.Integer(), nullable=True), schema='production')
    op.add_column('list_predictions', sa.Column('rank_pct_with_ties', sa.Float(), nullable=True), schema='production')
    op.alter_column('list_predictions', 'rank_abs', new_column_name='rank_abs_no_ties', schema='production')
    op.alter_column('list_predictions', 'rank_pct', new_column_name='rank_pct_no_ties', schema='production')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('list_predictions', 'rank_abs_no_ties', new_column_name='rank_abs', schema='production')
    op.alter_column('list_predictions', 'rank_pct_no_ties', new_column_name='rank_pct', schema='production')
    op.drop_column('list_predictions', 'rank_pct_with_ties', schema='production')
    op.drop_column('list_predictions', 'rank_abs_with_ties', schema='production')
    # ### end Alembic commands ###
