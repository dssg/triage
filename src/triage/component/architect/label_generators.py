import logging
import textwrap
from triage.database_reflection import table_exists


DEFAULT_LABEL_NAME = "outcome"


class LabelGeneratorNoOp(object):
    def generate_all_labels(self, labels_table, as_of_dates, label_timespans):
        logging.warning(
            "No label configuration is available, so no labels will be created"
        )

    def generate(self, start_date, label_timespan, labels_table):
        logging.warning(
            "No label configuration is available, so no labels will be created"
        )

    def clean_up(self, labels_table_name):
        pass


class LabelGenerator(object):
    def __init__(self, db_engine, query, label_name=None, replace=True):
        self.db_engine = db_engine
        self.replace = replace
        # query is expected to select a number of entity ids
        # and an outcome for each given an as-of-date
        self.query = query
        self.label_name = label_name or DEFAULT_LABEL_NAME

    def _create_labels_table(self, labels_table_name):
        if self.replace or not table_exists(labels_table_name, self.db_engine):
            self.db_engine.execute("drop table if exists {}".format(labels_table_name))
            self.db_engine.execute(
                """
                create table {} (
                entity_id int,
                as_of_date date,
                label_timespan interval,
                label_name varchar(30),
                label_type varchar(30),
                label int
            )""".format(
                    labels_table_name
                )
            )
        else:
            logging.info("Not dropping and recreating table because "
                         "replace flag was set to False and table was found to exist")

    def generate_all_labels(self, labels_table, as_of_dates, label_timespans):
        self._create_labels_table(labels_table)
        logging.info(
            "Creating labels for %s as of dates and %s label timespans",
            len(as_of_dates),
            len(label_timespans),
        )
        for as_of_date in as_of_dates:
            for label_timespan in label_timespans:
                if not self.replace:
                    logging.info(
                        "Looking for existing labels for as of date %s and label timespan %s",
                        as_of_date,
                        label_timespan,
                    )
                    any_existing_labels = list(self.db_engine.execute(
                        """select 1 from {labels_table}
                        where as_of_date = '{as_of_date}'
                        and label_timespan = '{label_timespan}'::interval
                        and label_name = '{label_name}'
                        limit 1
                        """.format(
                            labels_table=labels_table,
                            as_of_date=as_of_date,
                            label_timespan=label_timespan,
                            label_name=self.label_name
                        )
                    ))
                    if len(any_existing_labels) == 1:
                        logging.info("Since nonzero existing labels found, skipping")
                        continue

                logging.info(
                    "Generating labels for as of date %s and label timespan %s",
                    as_of_date,
                    label_timespan,
                )
                self.generate(
                    start_date=as_of_date,
                    label_timespan=label_timespan,
                    labels_table=labels_table,
                )
        nrows = [
            row[0]
            for row in self.db_engine.execute(
                "select count(*) from {}".format(labels_table)
            )
        ][0]
        if nrows == 0:
            logging.warning("Done creating labels, but no rows in labels table!")
        else:
            logging.info("Done creating labels table %s: rows: %s", labels_table, nrows)

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
            """
            insert into {labels_table}
            select
                entities_and_outcomes.entity_id,
                '{as_of_date}' as as_of_date,
                '{label_timespan}'::interval as label_timespan,
                '{label_name}' as label_name,
                'binary' as label_type,
                entities_and_outcomes.outcome as label
            from ({user_query}) entities_and_outcomes
        """
        ).format(
            user_query=query_with_db_variables,
            labels_table=labels_table,
            as_of_date=start_date,
            label_timespan=label_timespan,
            label_name=self.label_name,
        )

        logging.debug("Running label creation query")
        logging.debug(full_insert_query)
        self.db_engine.execute(full_insert_query)

    def clean_up(self, labels_table_name):
        self.db_engine.execute("drop table if exists {}".format(labels_table_name))
