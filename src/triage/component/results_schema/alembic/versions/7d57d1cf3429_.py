"""empty message

Revision ID: 7d57d1cf3429
Revises: 72ac5cbdca05
Create Date: 2017-11-06 11:34:23.046005

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "7d57d1cf3429"
down_revision = "72ac5cbdca05"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column(
        "evaluations", "example_frequency", new_column_name="as_of_date_frequency"
    )
    op.alter_column(
        "models", "train_label_window", new_column_name="training_label_timespan"
    )
    op.alter_column(
        "predictions", "test_label_window", new_column_name="test_label_timespan"
    )
    op.alter_column(
        "list_predictions", "test_label_window", new_column_name="test_label_timespan"
    )


def downgrade():
    op.alter_column(
        "evaluations", "as_of_date_frequency", new_column_name="example_frequency"
    )
    op.alter_column(
        "models", "training_label_timespan", new_column_name="train_label_window"
    )
    op.alter_column(
        "predictions", "test_label_timespan", new_column_name="test_label_window"
    )
    op.alter_column(
        "list_predictions", "test_label_timespan", new_column_name="test_label_window"
    )
