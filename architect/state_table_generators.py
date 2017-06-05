import logging


class StateFilter(object):
    def __init__(self, sparse_state_table, filter_logic):
        self.sparse_state_table = sparse_state_table
        self.filter_logic = filter_logic

    def join_sql(self, join_table, join_date):
        return '''
            join {state_table} on (
                {state_table}.entity_id = {join_table}.entity_id and
                {state_table}.as_of_date = '{join_date}'::timestamp and
                ({state_filter_logic})
            )
        '''.format(
            state_table=self.sparse_state_table,
            state_filter_logic=self.filter_logic,
            join_date=join_date,
            join_table=join_table
        )


class StateTableGenerator(object):
    """Take a dense state table and create a sparse state table"""
    def __init__(self, db_engine, experiment_hash):
        self.db_engine = db_engine
        self.experiment_hash = experiment_hash

    @property
    def sparse_table_name(self):
        return 'tmp_sparse_states_{}'.format(self.experiment_hash)

    def _all_known_states(self, dense_state_table):
        all_states = [
            row[0] for row in
            self.db_engine.execute('''
                select distinct(state) from {} order by state
            '''.format(dense_state_table))
        ]
        logging.info('Distinct states found: %s', all_states)
        return all_states

    def _sparse_table_query(self, dense_state_table, as_of_dates):
        state_columns = [
            'bool_or(state = \'{desired_state}\') as {desired_state}'
            .format(desired_state=state)
            for state in self._all_known_states(dense_state_table)
        ]
        query = '''
            create table {sparse_state_table} as (
            select d.entity_id, a.as_of_date::timestamp, {state_column_string}
                from {dense_state_table} d
                join (select unnest(ARRAY{as_of_dates}) as as_of_date) a
                on (
                    d.start_time <= a.as_of_date::timestamp and
                    d.end_time > a.as_of_date::timestamp
                )
                group by d.entity_id, a.as_of_date
            )
        '''.format(
            sparse_state_table=self.sparse_table_name,
            dense_state_table=dense_state_table,
            as_of_dates=[date.isoformat() for date in as_of_dates],
            state_column_string=', '.join(state_columns)
        )
        return query

    def generate_sparse_table(self, dense_state_table, as_of_dates):
        """
        input table:
        entity_id, state, start_date, end_date

        output table:
        entity_id, as_of_date, state_one, state_two
        """
        self.db_engine.execute(
            self._sparse_table_query(dense_state_table, as_of_dates)
        )
        logging.info('Sparse state table generated')
        self.db_engine.execute(
            'create index on {} (entity_id, as_of_date)'
            .format(self.sparse_table_name)
        )
        logging.info('Indices created for sparse state table')

    def clean_up(self):
        self.db_engine.execute(
            'drop table if exists {}'.format(self.sparse_table_name)
        )
