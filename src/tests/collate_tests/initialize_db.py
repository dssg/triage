import pathlib
import subprocess
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

    # Use csvsql to load the CSV data - it needs the engine URL
    # render_as_string with hide_password=False ensures the password is included
    db_url = str(engine.url.render_as_string(hide_password=False))
    subprocess.run(
        [
            "csvsql",
            "-v",
            "--no-constraints",
            "--tables",
            "food_inspections",
            "--insert",
            "--db",
            db_url,
            str(DATA_PATH),
        ],
        check=True,
    )

    with engine.connect() as conn:
        conn.execute(text("CREATE INDEX ON food_inspections(license_no, inspection_date)"))

        # create a state table for license/date
        conn.execute(text("DROP TABLE IF EXISTS inspection_states"))
        conn.execute(
            text("""\
            CREATE TABLE inspection_states AS (
                SELECT license_no, date
                FROM (SELECT DISTINCT license_no FROM food_inspections) a
                CROSS JOIN (SELECT DISTINCT inspection_date as date FROM food_inspections) b
            )""")
        )
        conn.execute(text("CREATE INDEX ON inspection_states(license_no, date)"))

        # create an alternate state table with a different date column
        conn.execute(text("DROP TABLE IF EXISTS inspection_states_diff_colname"))
        conn.execute(
            text("""\
            CREATE TABLE inspection_states_diff_colname
            AS select license_no, date as aggregation_date
            FROM inspection_states
            """)
        )
        conn.execute(
            text("""\
            CREATE INDEX ON
            inspection_states_diff_colname(license_no, aggregation_date)
            """)
        )

        # create a state table for license only
        conn.execute(text("DROP TABLE IF EXISTS all_licenses"))
        conn.execute(
            text("""\
            CREATE TABLE all_licenses AS (
                SELECT DISTINCT license_no FROM food_inspections
            )""")
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
