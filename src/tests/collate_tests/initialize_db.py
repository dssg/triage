import pathlib

import pandas as pd
from sqlalchemy import text

DATA_NAME = "food_inspections_subset.csv"
DATA_PATH = pathlib.Path(__file__).with_name(DATA_NAME)


def load_data(engine):
    """Load food inspections test data into the database.

    This function is designed to work with pytest-postgresql fixtures.
    It creates the food_inspections table and related state tables.
    """
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS food_inspections"))
        conn.commit()

    # Load CSV with pandas and insert into database
    df = pd.read_csv(DATA_PATH)

    # Parse inspection_date as datetime for proper type inference
    df["inspection_date"] = pd.to_datetime(df["inspection_date"])

    # Use pandas to_sql - replaces csvsql subprocess call
    df.to_sql("food_inspections", engine, if_exists="replace", index=False)

    with engine.connect() as conn:
        conn.execute(
            text("CREATE INDEX ON food_inspections(license_no, inspection_date)")
        )

        # create a state table for license/date
        conn.execute(text("DROP TABLE IF EXISTS inspection_states"))
        conn.execute(
            text(
                """\
            CREATE TABLE inspection_states AS (
                SELECT license_no, date
                FROM (SELECT DISTINCT license_no FROM food_inspections) a
                CROSS JOIN (SELECT DISTINCT inspection_date as date FROM food_inspections) b
            )"""
            )
        )
        conn.execute(text("CREATE INDEX ON inspection_states(license_no, date)"))

        # create an alternate state table with a different date column
        conn.execute(text("DROP TABLE IF EXISTS inspection_states_diff_colname"))
        conn.execute(
            text(
                """\
            CREATE TABLE inspection_states_diff_colname
            AS select license_no, date as aggregation_date
            FROM inspection_states
            """
            )
        )
        conn.execute(
            text(
                """\
            CREATE INDEX ON
            inspection_states_diff_colname(license_no, aggregation_date)
            """
            )
        )

        # create a state table for license only
        conn.execute(text("DROP TABLE IF EXISTS all_licenses"))
        conn.execute(
            text(
                """\
            CREATE TABLE all_licenses AS (
                SELECT DISTINCT license_no FROM food_inspections
            )"""
            )
        )
        conn.execute(text("CREATE INDEX ON all_licenses(license_no)"))
        conn.commit()


# Legacy handler for testing.postgresql compatibility (deprecated)
def handler(database):
    """Legacy handler for testing.postgresql.PostgresqlFactory.

    This is kept for backwards compatibility but the preferred method
    is to use load_data() with a pytest-postgresql fixture.
    """
    from sqlalchemy import create_engine as legacy_create_engine

    engine = legacy_create_engine(database.url())
    load_data(engine)
    engine.dispose()
