import logging
from abc import ABC, abstractmethod

from triage.component.architect.database_reflection import table_has_data
from triage.database_reflection import table_row_count


DEFAULT_ACTIVE_STATE = 'active'


class StateTableGeneratorBase(ABC):
    """Create a table containing the state of entities on different dates

    Referred to as a 'sparse states table', this table contains
        entities, dates, and boolean values for different states.

        These types of tables rarely exist in source data, but are useful
        internally in pipelines to express state logic in SQL

        For instance: '(in_jail OR booked) AND in_mental_health_facility'

        The output format is entity id/date/state1/state3/state3...

    Subclasses must implement the following methods:
        '_create_and_populate_sparse_table' to take dates
            and return a query to create the states table for those dates
        '_empty_table_message' to provide a helpful message to the user
            if no rows are found in the resultant table

    The main interface of StateTableGenerator objects is the
    `generate_sparse_table` method, which produces the latter
    'sparse'-style table.

    Args:
        db_engine (sqlalchemy.engine)
        experiment_hash (string) unique identifier for the experiment

    """
    def __init__(self, db_engine, experiment_hash):
        self.db_engine = db_engine
        self.experiment_hash = experiment_hash

    @abstractmethod
    def _create_and_populate_sparse_table(self, as_of_dates):
        pass

    @abstractmethod
    def _empty_table_message(self, as_of_dates):
        pass

    @property
    def sparse_table_name(self):
        return 'tmp_sparse_states_{}'.format(self.experiment_hash)

    def generate_sparse_table(self, as_of_dates):
        """Convert the object's input table
        into a sparse states table for the given as_of_dates

        Args:
            as_of_dates (list of datetime.dates) Dates to include in the sparse
                state table
        """
        logging.debug('Generating sparse table using as_of_dates: %s', as_of_dates)
        self._create_and_populate_sparse_table(as_of_dates)
        self.db_engine.execute(
            'create index on {} (entity_id, as_of_date)'
            .format(self.sparse_table_name)
        )
        logging.info('Indices created on entity_id and as_of_date for sparse state table')
        if not table_has_data(self.sparse_table_name, self.db_engine):
            raise ValueError(self._empty_table_message(as_of_dates))

        logging.info('Sparse states table generated at %s', self.sparse_table_name)
        logging.info('Generating stats on %s', self.sparse_table_name)
        logging.info('Row count of %s: %s',
                     self.sparse_table_name,
                     table_row_count(self.sparse_table_name, self.db_engine))

    def clean_up(self):
        self.db_engine.execute(
            'drop table if exists {}'.format(self.sparse_table_name)
        )


class StateTableGeneratorFromEntities(StateTableGeneratorBase):
    """Generates a 'sparse'-style states table from a table containing entity_ids

    This will include all entities found in the table for all given dates

    Args:
        entities_table (string, optional) name of SQL table containing
           entities
    """

    def __init__(self, entities_table, *args, **kwargs):

        super(StateTableGeneratorFromEntities, self).__init__(*args, **kwargs)
        self.entities_table = entities_table

    def _create_and_populate_sparse_table(self, as_of_dates):
        """Create a 'sparse'-style table from an entities table
            for a specific set of dates

        This will include all entities for all given dates

        Args:
        as_of_dates (list of datetime.date): Dates to calculate entity states as of
        """

        query = '''
            create table {sparse_state_table} as (
            select e.entity_id, a.as_of_date::timestamp, true {active_state}
                from {entities_table} e
                cross join (select unnest(ARRAY{as_of_dates}) as as_of_date) a
                group by e.entity_id, a.as_of_date
            )
        '''.format(
            sparse_state_table=self.sparse_table_name,
            entities_table=self.entities_table,
            as_of_dates=[date.isoformat() for date in as_of_dates],
            active_state=DEFAULT_ACTIVE_STATE
        )
        logging.debug('Assembled sparse state table query: %s', query)
        self.db_engine.execute(query)

    def _empty_table_message(self, as_of_dates):
        return "No entities in entities table '{input_table}'".format(
            input_table=self.entities_table,
        )


class StateTableGeneratorFromQuery(StateTableGeneratorBase):
    """Generates a 'sparse'-style states table from a given query

    Args:
    query (string) SQL query string to select entities for a given as_of_date
        The as_of_date should be parameterized with brackets: {as_of_date}
    """

    def __init__(self, query, *args, **kwargs):

        super(StateTableGeneratorFromQuery, self).__init__(*args, **kwargs)
        self.query = query

    def _create_and_populate_sparse_table(self, as_of_dates):
        """Create a 'sparse'-style states table by sequentially running a
            given date-parameterized query for all known dates.

        Args:
        as_of_dates (list of datetime.date): Dates to calculate entity states as of
        """

        self.db_engine.execute(
            f'''create table {self.sparse_table_name} (
                entity_id integer,
                as_of_date timestamp,
                {DEFAULT_ACTIVE_STATE} boolean
            )
            '''
        )
        logging.info('Created sparse state table, now inserting rows')

        for as_of_date in as_of_dates:
            formatted_date = f"{as_of_date.isoformat()}"
            dated_query = self.query.replace('{as_of_date}', formatted_date)
            full_query = f'''insert into {self.sparse_table_name}
                select q.entity_id, '{formatted_date}'::timestamp, true
                from ({dated_query}) q
                group by 1, 2, 3
            '''
            logging.info(
                f'Running state query for date: {as_of_date}, {full_query}',
            )
            self.db_engine.execute(full_query)

    def _empty_table_message(self, as_of_dates):
        return """Query does not return any rows for the given as_of_dates:
            {as_of_dates}
            '{query}'""".format(
            query=self.query,
            as_of_dates=', '.join(str(as_of_date) for as_of_date in (
                as_of_dates if len(as_of_dates) <= 5 else as_of_dates[:5] + ['…']
            )),
        )


class StateTableGeneratorFromDense(StateTableGeneratorBase):
    """Creates a 'sparse'-style states table from a 'dense'-style states table.

    A dense states table contains entities and *time ranges* for different states

    An example of this would be a building permits table, with
    time ranges that buildings are permitted.

    The expected format is entity id/state/start/end

    Args:
        dense_state_table (string, optional) name of SQL table containing
            entities and state time ranges
    """
    def __init__(self, dense_state_table, *args, **kwargs):
        super(StateTableGeneratorFromDense, self).__init__(*args, **kwargs)
        self.dense_state_table = dense_state_table

    def all_known_states(self):
        all_states = [
            row[0] for row in
            self.db_engine.execute('''
                select distinct(state) from {} order by state
            '''.format(self.dense_state_table))
        ]
        logging.info('Distinct states found: %s', all_states)
        return all_states

    def state_columns(self):
        state_columns = [
            "bool_or(state = '{desired_state}') as {desired_state}"
            .format(desired_state=state)
            for state in self.all_known_states()
        ]
        if not state_columns:
            raise ValueError("Unable to identify states from table",
                             self.dense_state_table)
        return state_columns

    def _create_and_populate_sparse_table(self, as_of_dates):
        """Creates a sparse states table from a dense states table for a given list of dates

        Args:
        as_of_dates (list of datetime.date): Dates to calculate entity states as of

        Returns: (string) A query to produce a sparse states table
        """
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
            state_column_string=', '.join(self.state_columns())
        )
        logging.debug('Assembled sparse state table query: %s', query)
        self.db_engine.execute(query)

    def _empty_table_message(self, as_of_dates):
        return \
            "No entities in dense state table '{input_table}' define time ranges " + \
            "that encompass any of experiment's \"as-of-dates\":\n\n" + \
            "\t{as_of_dates}\n\n" + \
            "Please check temporal config and dense state table".format(
                input_table=self.dense_state_table,
                as_of_dates=', '.join(str(as_of_date) for as_of_date in (
                    as_of_dates if len(as_of_dates) <= 5 else as_of_dates[:5] + ['…']
                )),
            )


class StateTableGeneratorNoOp(object):
    def generate_sparse_table(self, as_of_dates):
        logging.warning('No cohort configuration is available, so no cohort will be created')
        return

    def clean_up(self):
        logging.warning('No cohort table exists, so nothing to tear down')
        return

    @property
    def sparse_table_name(self):
        return None
