import logging
import textwrap
from abc import ABCMeta, abstractmethod


class BinaryLabelBase(object, metaclass=ABCMeta):
    def __init__(self, db_engine):
        self.db_engine = db_engine

    def _create_labels_table(self, labels_table_name):
        self.db_engine.execute(
            'drop table if exists {}'.format(labels_table_name)
        )
        self.db_engine.execute('''
            create table {} (
            entity_id int,
            as_of_date date,
            label_timespan interval,
            label_name varchar(30),
            label_type varchar(30),
            label int
        )'''.format(labels_table_name))

    def generate_all_labels(
        self,
        labels_table,
        as_of_dates,
        label_timespans,
    ):
        self._create_labels_table(labels_table)
        logging.info('Creating labels for %s as of dates and %s label timespans',
                     len(as_of_dates),
                     len(label_timespans))
        for as_of_date in as_of_dates:
            for label_timespan in label_timespans:
                logging.info('Generating labels for as of date %s and '
                             'label timespan %s', as_of_date, label_timespan)
                self.generate(
                    start_date=as_of_date,
                    label_timespan=label_timespan,
                    labels_table=labels_table,
                )
        nrows = [
            row[0] for row in
            self.db_engine.execute(
                'select count(*) from {}'.format(labels_table))
        ][0]
        if nrows == 0:
            logging.warning(
                'Done creating labels, but no rows in labels table!')
        else:
            logging.info('Rows in labels table: %s', nrows)

    @abstractmethod
    def generate(self):
        pass


class InspectionsLabelGenerator(BinaryLabelBase):
    def __init__(self, events_table, *args, **kwargs):
        super(InspectionsLabelGenerator, self).__init__(*args, **kwargs)
        self.events_table = events_table

    def generate(
        self,
        start_date,
        label_timespan,
        labels_table,
    ):
        query = """insert into {labels_table} (
            select
                {events_table}.entity_id,
                '{start_date}'::date as as_of_date,
                '{label_timespan}'::interval as label_timespan,
                'outcome' as label_name,
                'binary' as label_type,
                bool_or(outcome::bool)::int as label
            from {events_table}
            where '{start_date}' <= outcome_date
            and outcome_date < '{start_date}'::timestamp + interval '{label_timespan}'
            group by 1, 2, 3, 4, 5
        )""".format(
            events_table=self.events_table,
            labels_table=labels_table,
            start_date=start_date,
            label_timespan=label_timespan,
        )

        logging.debug('Running label generation query: %s', query)
        self.db_engine.execute(query)
        return labels_table


class QueryBinaryLabelGenerator(BinaryLabelBase):
    def __init__(self, query, *args, **kwargs):
        super(QueryBinaryLabelGenerator, self).__init__(*args, **kwargs)
        # query is expected to select a number of entity ids
        # and an outcome for each given an as-of-date
        self.query = query

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
        self._create_labels_table(labels_table)
        # we want to apply the as-of-date and label in the database driver,
        # so replace the user {as_of_date} with the SQL %(as_of_date)

        query_with_db_variables = self.query.format(
            as_of_date=start_date,
            label_timespan=label_timespan
        )

        full_insert_query = textwrap.dedent('''
            insert into {labels_table}
            select
                entities_and_outcomes.entity_id,
                '{as_of_date}' as as_of_date,
                '{label_timespan}'::interval as label_timespan,
                'outcome' as label_name,
                'binary' as label_type,
                entities_and_outcomes.outcome as label
            from ({user_query}) entities_and_outcomes
        ''').format(
            user_query=query_with_db_variables,
            labels_table=labels_table,
            as_of_date=start_date,
            label_timespan=label_timespan,
        )

        logging.debug("Running label creation query")
        logging.debug(full_insert_query)
        self.db_engine.execute(
            full_insert_query,
        )
