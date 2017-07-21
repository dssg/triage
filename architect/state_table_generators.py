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


DEFAULT_ACTIVE_STATE = 'active'

class StateTableGenerator(object):
    """Create a table containing the state of entities on different dates

    Throughout the class we will refer to two types of state tables:
        'dense' and 'sparse'.

    dense: A table containing entities and *time ranges* for different states
        
        An example of this would be a building permits table, with 
        time ranges that buildings are permitted.

        The expected format would be entity id/state/start/end

    sparse: A table containing entities, dates, and boolean values for
        different states.

        These types of tables rarely exist in source data, but are useful
        internally in pipelines to express state logic in SQL

        For instance: '(in_jail OR booked) AND in_mental_health_facility'

        The output format is entity id/date/state1/state3/state3...

    The main interface of StateTableGenerator objects is the
    `generate_sparse_table` method, which produces the latter
    'sparse'-style table.

    Args:
        db_engine (sqlalchemy.engine)
        experiment_hash (string) unique identifier for the experiment
        events_table (string, optional) name of SQL table containing
            outcome events for entities
        dense_state_table (string, optional) name of SQL table containing
            entities and state time ranges
    """
    def __init__(
        self,
        db_engine,
        experiment_hash,
        events_table=None,
        dense_state_table=None
    ):
        self.db_engine = db_engine
        self.experiment_hash = experiment_hash
        self.dense_state_table = dense_state_table
        self.events_table = events_table

    @property
    def sparse_table_name(self):
        return 'tmp_sparse_states_{}'.format(self.experiment_hash)

    @property
    def sparse_table_query_func(self):
        if self.dense_state_table:
            return self._sparse_table_query_from_dense
        else:
            return self._sparse_table_query_from_events

    def _all_known_states(self, dense_state_table):
        all_states = [
            row[0] for row in
            self.db_engine.execute('''
                select distinct(state) from {} order by state
            '''.format(dense_state_table))
        ]
        logging.info('Distinct states found: %s', all_states)
        return all_states

    def _sparse_table_query_from_dense(self, as_of_dates):
        """A query to convert a dense-style state table to a 'sparse'-style
        table containing a specific set of dates.

        Args:
        as_of_dates (list of datetime.date): Dates to calculate entity states as of

        Returns: (string) A query to produce a sparse states table
        """
        state_columns = [
            'bool_or(state = \'{desired_state}\') as {desired_state}'
            .format(desired_state=state)
            for state in self._all_known_states(self.dense_state_table)
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
            dense_state_table=self.dense_state_table,
            as_of_dates=[date.isoformat() for date in as_of_dates],
            state_column_string=', '.join(state_columns)
        )
        return query

    def _sparse_table_query_from_events(self, as_of_dates):
        """A query to convert an events table to a 'sparse'-style
        table containing a specific set of dates.

        This will include all entities for all given dates

        Args:
        as_of_dates (list of datetime.date): Dates to calculate entity states as of

        Returns: (string) A query to produce a sparse states table
        """

        query = '''
            create table {sparse_state_table} as (
            select e.entity_id, a.as_of_date::timestamp, true {active_state}
                from {events_table} e
                cross join (select unnest(ARRAY{as_of_dates}) as as_of_date) a
                group by e.entity_id, a.as_of_date
            )
        '''.format(
            sparse_state_table=self.sparse_table_name,
            events_table=self.events_table,
            as_of_dates=[date.isoformat() for date in as_of_dates],
            active_state=DEFAULT_ACTIVE_STATE 
        )
        return query

    def generate_sparse_table(self, as_of_dates):
        """Convert the object's input table (either dense states or events)
        into a sparse states table for the given as_of_dates

        Args:
            as_of_dates (list of datetime.dates) Dates to include in the sparse
                state table
        """
        self._generate_sparse_table(self.sparse_table_query_func(as_of_dates))

    def _generate_sparse_table(self, generate_query):
        """Generate and index a sparse table from a given query

        Args:
            generate_query (string) A full query to generate a sparse table
        """
        self.db_engine.execute(generate_query)
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
