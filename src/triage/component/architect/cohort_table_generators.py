import logging

from triage.component.architect.database_reflection import table_has_data
from triage.database_reflection import table_row_count, table_exists


DEFAULT_ACTIVE_STATE = "active"


class CohortTableGenerator(object):
    """Create a table containing cohort membership on different dates

    The structure of the output table is:
        entity_id
        date
        active (boolean): Whether or not the entity is considered 'active'
            (in the cohort) on that date

    Args:
        db_engine (sqlalchemy.engine)
        experiment_hash (string) unique identifier for the experiment
        query (string) SQL query string to select entities for a given as_of_date
            The as_of_date should be parameterized with brackets: {as_of_date}
        replace (boolean) Whether or not to overwrite old rows.
            If false, each as-of-date will query to see if there are existing rows
                and not run the query if so.
            If true, the existing cohort table will be dropped and recreated.
    """
    def __init__(self, query, db_engine, cohort_table_name, replace=True):
        self.db_engine = db_engine
        self.query = query
        self.cohort_table_name = cohort_table_name
        self.replace = replace

    def generate_cohort_table(self, as_of_dates):
        """Convert the object's input table
        into a cohort states table for the given as_of_dates

        Args:
            as_of_dates (list of datetime.dates) Dates to include in the cohort
                state table
        """
        logging.debug("Generating cohort table using as_of_dates: %s", as_of_dates)
        self._create_and_populate_cohort_table(as_of_dates)
        self.db_engine.execute(
            "create index on {} (entity_id, as_of_date)".format(self.cohort_table_name)
        )
        logging.info(
            "Indices created on entity_id and as_of_date for cohort table"
        )
        if not table_has_data(self.cohort_table_name, self.db_engine):
            raise ValueError(self._empty_table_message(as_of_dates))

        logging.info("Cohort table generated at %s", self.cohort_table_name)
        logging.info("Generating stats on %s", self.cohort_table_name)
        logging.info(
            "Row count of %s: %s",
            self.cohort_table_name,
            table_row_count(self.cohort_table_name, self.db_engine),
        )

    def _maybe_create_cohort_table(self):
        if self.replace or not table_exists(self.cohort_table_name, self.db_engine):
            self.db_engine.execute(f"drop table if exists {self.cohort_table_name}")
            self.db_engine.execute(
                f"""create table {self.cohort_table_name} (
                    entity_id integer,
                    as_of_date timestamp,
                    {DEFAULT_ACTIVE_STATE} boolean
                )
                """
            )
            logging.info("Created cohort table")
        else:
            logging.info("Not dropping and recreating cohort table because "
                         "replace flag was set to False and table was found to exist")

    def _create_and_populate_cohort_table(self, as_of_dates):
        """Create a cohort table by sequentially running a
            given date-parameterized query for all known dates.

        Args:
        as_of_dates (list of datetime.date): Dates to calculate entity states as of
        """
        self._maybe_create_cohort_table()
        logging.info("Inserting rows into cohort table")
        for as_of_date in as_of_dates:
            formatted_date = f"{as_of_date.isoformat()}"
            logging.info("Looking for existing cohort rows for as of date %s", as_of_date)
            any_existing = list(self.db_engine.execute(
                f"""select 1 from {self.cohort_table_name}
                where as_of_date = '{formatted_date}'
                limit 1
                """
            ))
            if len(any_existing) == 1:
                logging.info("Since >0 cohort rows found for date %s, skipping", as_of_date)
                continue
            dated_query = self.query.format(as_of_date=formatted_date)
            full_query = f"""insert into {self.cohort_table_name}
                select q.entity_id, '{formatted_date}'::timestamp, true
                from ({dated_query}) q
                group by 1, 2, 3
            """
            logging.info("Running cohort query for date: %s, %s", as_of_date, full_query)
            self.db_engine.execute(full_query)

    def _empty_table_message(self, as_of_dates):
        return """Query does not return any rows for the given as_of_dates:
            {as_of_dates}
            '{query}'""".format(
            query=self.query,
            as_of_dates=", ".join(
                str(as_of_date)
                for as_of_date in (
                    as_of_dates if len(as_of_dates) <= 5 else as_of_dates[:5] + ["…"]
                )
            ),
        )

    def clean_up(self):
        self.db_engine.execute("drop table if exists {}".format(self.cohort_table_name))


class CohortTableGeneratorNoOp(CohortTableGenerator):
    def __init__(self):
        pass

    def generate_cohort_table(self, as_of_dates):
        logging.warning(
            "No cohort configuration is available, so no cohort will be created"
        )
        return

    def clean_up(self):
        logging.warning("No cohort table exists, so nothing to tear down")
        return

    @property
    def cohort_table_name(self):
        return None
