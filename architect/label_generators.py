import logging


class BinaryLabelGenerator(object):
    def __init__(self, events_table, db_engine):
        self.events_table = events_table
        self.db_engine = db_engine

    def generate(
        self,
        start_date,
        label_window,
        labels_table,
        state_filter=None,
    ):
        if state_filter:
            state_join_sql = state_filter.join_sql(self.events_table, start_date)
        else:
            state_join_sql = ''

        query = """insert into {labels_table} (
            select
                {events_table}.entity_id,
                '{start_date}'::date as as_of_date,
                '{label_window}'::interval as label_window,
                'outcome' as label_name,
                'binary' as label_type,
                bool_or(outcome::bool)::int as label
            from {events_table}
            {state_join_sql}
            where '{start_date}' <= outcome_date
            and outcome_date < '{start_date}'::timestamp + interval '{label_window}'
            group by 1, 2, 3, 4, 5
        )""".format(
            events_table=self.events_table,
            labels_table=labels_table,
            start_date=start_date,
            label_window=label_window,
            state_join_sql=state_join_sql
        )
        logging.debug(query)
        self.db_engine.execute(query)
        return labels_table

    def _create_labels_table(self, labels_table_name):
        self.db_engine.execute(
            'drop table if exists {}'.format(labels_table_name)
        )
        self.db_engine.execute('''
            create table {} (
            entity_id int,
            as_of_date date,
            label_window interval,
            label_name varchar(30),
            label_type varchar(30),
            label int
        )'''.format(labels_table_name))

    def generate_all_labels(
        self,
        labels_table,
        as_of_times,
        label_windows,
        state_filter=None
    ):
        self._create_labels_table(labels_table)
        logging.info('Creating labels for %s as of times and %s label windows',
                     len(as_of_times),
                     len(label_windows))
        for as_of_time in as_of_times:
            for label_window in label_windows:
                self.generate(
                    start_date=as_of_time,
                    label_window=label_window,
                    labels_table=labels_table,
                    state_filter=state_filter
                )
        nrows = [
            row[0] for row in
            self.db_engine.execute('select count(*) from {}'.format(labels_table))
        ][0]
        if nrows == 0:
            logging.warning('Done creating labels, but no rows in labels table!')
        else:
            logging.info('Rows in labels table: %s', nrows)
