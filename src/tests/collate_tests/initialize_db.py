import pathlib

import pandas as pd
from sqlalchemy import text

DATA_NAME = "food_inspections_subset.csv"
DATA_PATH = pathlib.Path(__file__).with_name(DATA_NAME)


def load_data(db_engine):
    """Load food inspections data and create necessary tables for collate tests."""
    # Load CSV data using pandas, parsing the date column
    df = pd.read_csv(DATA_PATH, parse_dates=["inspection_date"])

    with db_engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS food_inspections"))

    # Write dataframe to database
    df.to_sql(
        "food_inspections",
        db_engine,
        if_exists="replace",
        index=False,
    )

    with db_engine.begin() as conn:
        conn.execute(
            text("CREATE INDEX ON food_inspections(license_no, inspection_date)")
        )

        # create a state table for license/date
        conn.execute(text("DROP TABLE IF EXISTS inspection_states"))
        conn.execute(text("""
                CREATE TABLE inspection_states AS (
                    SELECT license_no, date
                    FROM (SELECT DISTINCT license_no FROM food_inspections) a
                    CROSS JOIN (SELECT DISTINCT inspection_date as date FROM food_inspections) b
                )"""))
        conn.execute(text("CREATE INDEX ON inspection_states(license_no, date)"))

        # create an alternate state table with a different date column
        conn.execute(text("DROP TABLE IF EXISTS inspection_states_diff_colname"))
        conn.execute(text("""
                    CREATE TABLE inspection_states_diff_colname
                    AS select license_no, date as aggregation_date
                    FROM inspection_states
                """))
        conn.execute(text("""
                CREATE INDEX ON
                inspection_states_diff_colname(license_no, aggregation_date)
                """))

        # create a state table for license only
        conn.execute(text("DROP TABLE IF EXISTS all_licenses"))
        conn.execute(text("""
                CREATE TABLE all_licenses AS (
                    SELECT DISTINCT license_no FROM food_inspections
                )"""))
        conn.execute(text("CREATE INDEX ON all_licenses(license_no)"))
