import pathlib
import subprocess
from sqlalchemy import create_engine, text


DATA_NAME = "food_inspections_subset.csv"
DATA_PATH = pathlib.Path(__file__).with_name(DATA_NAME)


def handler(database):
    engine = create_engine(database.url())
    connection = engine
    #try:
    load_data(connection)
    #finally:
    #    connection.close()


def load_data(connection_):
    with connection_.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS food_inspections"))
        subprocess.run(
            [
                "csvsql",
                "-v",
                "--no-constraints",
                "--tables",
                "food_inspections",
                "--insert",
                "--db",
                str(connection.engine.url),
                str(DATA_PATH),
            ],
            check=True,
        )
        connection.execute(text("CREATE INDEX ON food_inspections(license_no, inspection_date)"))

        # create a state table for license/date
        connection.execute(text("DROP TABLE IF EXISTS inspection_states"))
        connection.execute(
            text(
                """
                CREATE TABLE inspection_states AS (
                    SELECT license_no, date
                    FROM (SELECT DISTINCT license_no FROM food_inspections) a
                    CROSS JOIN (SELECT DISTINCT inspection_date as date FROM food_inspections) b
                )"""
            )
        )
        connection.execute(text("CREATE INDEX ON inspection_states(license_no, date)"))

        # create an alternate state table with a different date column
        connection.execute(text("DROP TABLE IF EXISTS inspection_states_diff_colname"))
        connection.execute(
            text(
                """
                    CREATE TABLE inspection_states_diff_colname
                    AS select license_no, date as aggregation_date
                    FROM inspection_states
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX ON
                inspection_states_diff_colname(license_no, aggregation_date)
                """
            )
        )

        # create a state table for licenseo only
        connection.execute(text("DROP TABLE IF EXISTS all_licenses"))
        connection.execute(
            text(
                """
                CREATE TABLE all_licenses AS (
                    SELECT DISTINCT license_no FROM food_inspections
                )"""
            )
        )
        connection.execute(text("CREATE INDEX ON all_licenses(license_no)"))
