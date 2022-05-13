import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import textwrap
from triage.database_reflection import table_row_count, table_exists, table_has_duplicates

DEFAULT_LABEL_NAME = "outcome"


class LabelGeneratorNoOp:
    def generate_all_labels(self, labels_table, as_of_dates, label_timespans):
        logger.warning(
            "No label configuration is available, so no labels will be created"
        )

    def generate(self, start_date, label_timespan, labels_table):
        logger.warning(
            "No label configuration is available, so no labels will be created"
        )

    def clean_up(self, labels_table_name):
        pass


class LabelGenerator:
    def __init__(self, db_engine, query, label_name=None, replace=True):
        self.db_engine = db_engine
        self.replace = replace
        # query is expected to select a number of entity ids
        # and an outcome for each given an as-of-date
        self.query = query
        self.label_name = label_name or DEFAULT_LABEL_NAME

    def _create_labels_table(self, labels_table_name):
        if self.replace or not table_exists(labels_table_name, self.db_engine):
            self.db_engine.execute(f"drop table if exists {labels_table_name}")
            self.db_engine.execute(
                f"""
                create table {labels_table_name} (
                entity_id int,
                as_of_date date,
                label_timespan interval,
                label_name varchar,
                label_type varchar,
                label smallint
                )"""
            )
        else:
            logger.notice(f"Not dropping and recreating {labels_table_name} table because "
                          f"replace flag was set to False and table was found to exist")

    def generate_all_labels(self, labels_table, as_of_dates, label_timespans):
        self._create_labels_table(labels_table)
        logger.spam(f"Creating labels for {len(as_of_dates)} as of dates and {len(label_timespans)} label timespans")
        for as_of_date in as_of_dates:
            for label_timespan in label_timespans:
                if not self.replace:
                    logger.spam(f"Looking for existing labels for as of date {as_of_date} and label timespan {label_timespan}")
                    any_existing_labels = list(self.db_engine.execute(
                        f"""select 1 from {labels_table}
                        where as_of_date = '{as_of_date}'
                        and label_timespan = '{label_timespan}'::interval
                        and label_name = '{self.label_name}'
                        limit 1
                        """)
                    )
                    if len(any_existing_labels) == 1:
                        logger.spam("Since nonzero existing labels found, skipping")
                        continue

                logger.debug(
                    f"Generating labels for as of date {as_of_date} and label timespan {label_timespan}",
                )
                self.generate(
                    start_date=as_of_date,
                    label_timespan=label_timespan,
                    labels_table=labels_table,
                )

        self.db_engine.execute(
            f"create index on {labels_table} (entity_id, as_of_date)"
        )
        logger.spam("Added index to labels table")

        nrows = table_row_count(labels_table, self.db_engine)

        if nrows == 0:
            logger.warning(f"Done creating labels, but no rows in {labels_table} table!")
            raise ValueError(f"{labels_table} is empty!")

        if table_has_duplicates(
            labels_table,
            ['entity_id', 'as_of_date', 'label_timespan', 'label_name', 'label_type'],
            self.db_engine
            ):
            raise ValueError(f"Duplicates found in {labels_table}!")

        logger.debug(f"Labels table generated at {labels_table}")
        logger.spam(f"Row count of {labels_table}: {nrows}")

    def generate(self, start_date, label_timespan, labels_table):
        """Generate labels table using a query

        Parameters
        ----------
        start_date: str
            as of date
        label_timespan: str
            postgresql readable time interval
        labels_table: str
            name of labels table
        """
        # we want to apply the as-of-date and label in the database driver,
        # so replace the user {as_of_date} with the SQL %(as_of_date)

        query_with_db_variables = self.query.format(
            as_of_date=start_date, label_timespan=label_timespan
        )

        full_insert_query = textwrap.dedent(
            f"""
            insert into {labels_table}
            select
                entities_and_outcomes.entity_id,
                '{start_date}' as as_of_date,
                '{label_timespan}'::interval as label_timespan,
                '{self.label_name}' as label_name,
                'binary' as label_type,
                entities_and_outcomes.outcome as label
            from ({query_with_db_variables}) entities_and_outcomes
            """
        )

        logger.spam("Running label insertion query")
        logger.spam(full_insert_query)
        self.db_engine.execute(full_insert_query)

    def clean_up(self, labels_table_name):
        self.db_engine.execute(f"drop table if exists {labels_table_name}")
