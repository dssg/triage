import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from triage.component.architect.database_reflection import table_has_data
from triage.database_reflection import table_row_count, table_exists


DEFAULT_ACTIVE_STATE = "active"


class EntityDateTableGenerator:
    """Create a table containing state membership on different dates

    The structure of the output table is:
        entity_id
        date
        active (boolean): Whether or not the entity is considered 'active'
            (i.e., in the cohort or subset) on that date

    Args:
        db_engine (sqlalchemy.engine)
        experiment_hash (string) unique identifier for the experiment
        query (string) SQL query string to select entities for a given as_of_date
            The as_of_date should be parameterized with brackets: {as_of_date}
        replace (boolean) Whether or not to overwrite old rows.
            If false, each as-of-date will query to see if there are existing rows
                and not run the query if so.
            If true, the existing table will be dropped and recreated.
    """
    def __init__(self, query, db_engine, entity_date_table_name, replace=True):
        self.db_engine = db_engine
        self.query = query
        self.entity_date_table_name = entity_date_table_name
        self.replace = replace

    def generate_entity_date_table(self, as_of_dates):
        """Convert the object's input table
        into a states table for the given as_of_dates

        Args:
            as_of_dates (list of datetime.dates) Dates to include in the
                state table
        """
        logger.spam(f"Generating entity_date table {self.entity_date_table_name} using as_of_dates: {as_of_dates}")
        self._create_and_populate_entity_date_table(as_of_dates)
        logger.spam(f"Table {self.entity_date_table_name} created and populated")

        if not table_has_data(self.entity_date_table_name, self.db_engine):
            raise ValueError(self._empty_table_message(as_of_dates))

        logger.debug(f"Entity-date table generated at {self.entity_date_table_name}")
        logger.spam(f"Generating stats on {self.entity_date_table_name}")
        logger.spam(f"Row count of {self.entity_date_table_name}: {table_row_count(self.entity_date_table_name, self.db_engine)}")


    def _maybe_create_entity_date_table(self):
        if self.replace or not table_exists(self.entity_date_table_name, self.db_engine):
            logger.spam(f"Creating entity_date table {self.entity_date_table_name}")
            self.db_engine.execute(f"drop table if exists {self.entity_date_table_name}")
            self.db_engine.execute(
                f"""create table {self.entity_date_table_name} (
                    entity_id integer,
                    as_of_date timestamp,
                    {DEFAULT_ACTIVE_STATE} boolean
                )
                """
            )

            logger.spam(f"Creating indices on entity_id and as_of_date for entity_date table {self.entity_date_table_name}")
            self.db_engine.execute(
                f"create index on {self.entity_date_table_name} (entity_id, as_of_date)"
            )
        else:
            logger.notice(
                f"Not dropping and recreating entity_date {self.entity_date_table_name} table because "
                f"replace flag was set to False and table was found to exist"
            )

    def _create_and_populate_entity_date_table(self, as_of_dates):
        """Create an entity_date table by sequentially running a
            given date-parameterized query for all known dates.

        Args:
        as_of_dates (list of datetime.date): Dates to calculate entity states as of
        """
        self._maybe_create_entity_date_table()
        logger.spam(f"Inserting rows into entity_date table {self.entity_date_table_name}")
        for as_of_date in as_of_dates:
            formatted_date = f"{as_of_date.isoformat()}"
            logger.spam(f"Looking for existing entity_date rows for as of date {as_of_date}")
            any_existing = list(self.db_engine.execute(
                f"""select 1 from {self.entity_date_table_name}
                where as_of_date = '{formatted_date}'
                limit 1
                """
            ))
            if len(any_existing) == 1:
                logger.spam(f"Since >0 entity_date rows found for date {as_of_date}, skipping")
                continue
            dated_query = self.query.format(as_of_date=formatted_date)
            full_query = f"""insert into {self.entity_date_table_name}
                select q.entity_id, '{formatted_date}'::timestamp, true
                from ({dated_query}) q
                group by 1, 2, 3
            """
            logger.spam(f"Running entity_date query for date: {as_of_date}, {full_query}")
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
        self.db_engine.execute(f"drop table if exists {self.entity_date_table_name}")


class CohortTableGeneratorNoOp(EntityDateTableGenerator):
    def __init__(self):
        pass

    def generate_entity_date_table(self, as_of_dates):
        logger.warning(
            "No cohort configuration is available, so no cohort will be created"
        )
        return

    def clean_up(self):
        logger.warning("No cohort configuration is available, so no cohort will be tear down")
        return

    @property
    def entity_date_table_name(self):
        return None
