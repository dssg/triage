from collections import OrderedDict
from collate.collate import Aggregate, Categorical, Compare
from collate.spacetime import SpacetimeAggregation
from architect.utils import convert_str_to_relativedelta
import sqlalchemy
import logging


class FeatureGenerator(object):
    def __init__(
        self,
        db_engine,
        features_schema_name,
        replace=True,
        beginning_of_time=None
    ):
        """Generates aggregate features using collate

        Args:
            db_engine (sqlalchemy.db.engine)
            features_schema_name (string) Name of schema where feature
                tables should be written to
            replace (boolean, optional) Whether or not existing features
                should be replaced
            beginning_of_time (string/datetime, optional) point in time before which
                should not be included in features
        """
        self.db_engine = db_engine
        self.features_schema_name = features_schema_name
        self.categorical_cache = {}
        self.replace = replace
        self.beginning_of_time = beginning_of_time
        self.entity_id_column = 'entity_id'

    def _validate_keys(self, aggregation_config):
        for key in ['from_obj', 'intervals', 'groups', 'knowledge_date_column', 'prefix']:
            if key not in aggregation_config:
                raise ValueError(
                    '{} required as key: aggregation config: {}'
                    .format(key, aggregation_config)
                )

    def _validate_aggregates(self, aggregation_config):
        if 'aggregates' not in aggregation_config \
                and 'categoricals' not in aggregation_config \
                and 'array_categoricals' not in aggregation_config:
            raise ValueError(
                'Need either aggregates, categoricals, or array_categoricals' +
                ' in {}'.format(aggregation_config)
            )

    def _validate_categoricals(self, categoricals):
        conn = self.db_engine.connect()
        for categorical in categoricals:
            if 'choice_query' in categorical:
                logging.info('Validating choice query')
                try:
                    conn.execute('explain {}'.format(categorical['choice_query']))
                except Exception as e:
                    raise ValueError(
                        'choice query does not run. \nchoice query: "{}"\nFull error: {}'
                        .format(categorical['choice_query'], e)
                    )

    def _validate_from_obj(self, from_obj):
        conn = self.db_engine.connect()
        logging.info('Validating from_obj')
        try:
            conn.execute('explain select * from {}'.format(from_obj))
        except Exception as e:
            raise ValueError(
                'from_obj query does not run. \nfrom_obj: "{}"\nFull error: {}'
                .format(from_obj, e)
            )

    def _validate_time_intervals(self, intervals):
        logging.info('Validating time intervals')
        for interval in intervals:
            if interval != 'all':
                _ = convert_str_to_relativedelta(interval)

    def _validate_groups(self, groups):
        if 'entity_id' not in groups:
            raise ValueError('One of the aggregation groups is required to be entity_id')

    def _validate_aggregation(self, aggregation_config):
        logging.info('Validating aggregation config %s', aggregation_config)
        self._validate_keys(aggregation_config)
        self._validate_aggregates(aggregation_config)
        self._validate_categoricals(aggregation_config.get('categoricals', []))
        self._validate_from_obj(aggregation_config['from_obj'])
        self._validate_time_intervals(aggregation_config['intervals'])
        self._validate_groups(aggregation_config['groups'])

    def validate(self, feature_aggregation_config):
        """Validate a feature aggregation config applied to this object

        The validations range from basic type checks, key presence checks,
        as well as validating the sql in from objects.

        Args:
            feature_aggregation_config (list) all values, except for feature
                date, necessary to instantiate a collate.SpacetimeAggregation

        Raises: ValueError if any part of the config is found to be invalid
        """
        for aggregation in feature_aggregation_config:
            self._validate_aggregation(aggregation)

    def _compute_choices(self, choice_query):
        if choice_query not in self.categorical_cache:
            self.categorical_cache[choice_query] = [
                row[0]
                for row
                in self.db_engine.execute(choice_query)
            ]
        return self.categorical_cache[choice_query]

    def _build_choices(self, categorical):
        if 'choices' in categorical:
            return categorical['choices']
        else:
            return self._compute_choices(categorical['choice_query'])

    def _build_categoricals(self, categorical_config):
        return [
            Categorical(
                col=categorical['column'],
                choices=self._build_choices(categorical),
                function=categorical['metrics']
            )
            for categorical in categorical_config
        ]

    def _build_array_categoricals(self, categorical_config):
        return [
            Compare(
                col=categorical['column'],
                op='@>',
                choices={
                    choice: "array['{}'::varchar]".format(choice)
                    for choice in
                    self._build_choices(categorical)
                },
                function=categorical['metrics'],
                op_in_name=False,
                quote_choices=False,
            )
            for categorical in categorical_config
        ]

    def _aggregation(self, aggregation_config, feature_dates):
        aggregates = [
            Aggregate(aggregate['quantity'], aggregate['metrics'])
            for aggregate in aggregation_config.get('aggregates', [])
        ]
        logging.info('Found %s quantity aggregates', len(aggregates))
        categoricals = self._build_categoricals(
            aggregation_config.get('categoricals', [])
        )
        logging.info('Found %s categorical aggregates', len(categoricals))
        array_categoricals = self._build_array_categoricals(
            aggregation_config.get('array_categoricals', [])
        )
        logging.info('Found %s array categorical aggregates', len(array_categoricals))
        return SpacetimeAggregation(
            aggregates + categoricals + array_categoricals,
            from_obj=aggregation_config['from_obj'],
            intervals=aggregation_config['intervals'],
            groups=aggregation_config['groups'],
            dates=feature_dates,
            date_column=aggregation_config['knowledge_date_column'],
            output_date_column='as_of_date',
            input_min_date=self.beginning_of_time,
            schema=self.features_schema_name,
            prefix=aggregation_config['prefix']
        )

    def aggregations(self, feature_aggregation_config, feature_dates):
        """Creates collate.SpacetimeAggregations from the given arguments

        Args:
            feature_aggregation_config (list) all values, except for feature
                date, necessary to instantiate a collate.SpacetimeAggregation
            feature_dates (list) dates to generate features as of

        Returns: (list) collate.SpacetimeAggregations
        """
        return [
            self._aggregation(aggregation_config, feature_dates)
            for aggregation_config in feature_aggregation_config
        ]

    def generate_all_table_tasks(self, aggregations):
        """Generates SQL commands for creating, populating, and indexing
        feature group tables

        Args:
            aggregations (list) collate.SpacetimeAggregation objects

        Returns: (dict) keys are group table names, values are themselves dicts,
            each with keys for different stages of table creation (prepare, inserts, finalize)
            and with values being lists of SQL commands
        """
        logging.debug('---------------------')
        logging.debug('---------FEATURE GENERATION------------')
        logging.debug('---------------------')
        table_tasks = OrderedDict()
        for aggregation in aggregations:
            table_tasks.update(self._generate_table_tasks_for(aggregation))
        logging.info('Created %s tables', len(table_tasks.keys()))
        return table_tasks

    def create_all_tables(self, feature_aggregation_config, feature_dates):
        """Creates all feature tables.

        Args:
            feature_aggregation_config (list) all values, except for feature
                date, necessary to instantiate a collate.SpacetimeAggregation
            feature_dates (list) dates to generate features as of

        Returns: (list) table names
        """
        table_tasks = self.generate_all_table_tasks(
            self.aggregations(
                feature_aggregation_config,
                feature_dates
            )
        )

        return self.process_table_tasks(table_tasks)

    def process_table_tasks(self, table_tasks):
        for table_name, task in table_tasks.items():
            logging.info('Running feature table queries for %s', table_name)
            self.run_commands(task.get('prepare', []))
            self.run_commands(task.get('inserts', []))
            self.run_commands(task.get('finalize', []))
        return table_tasks.keys()

    def _explain_selects(self, aggregations):
        conn = self.db_engine.connect()
        for aggregation in aggregations:
            for selectlist in aggregation.get_selects().values():
                for select in selectlist:
                    result = [row for row in conn.execute('explain ' + str(select))]
                    logging.debug(str(select))
                    logging.debug(result)

    def _clean_table_name(self, table_name):
        # remove the schema and quotes from the name
        return table_name.split('.')[1].replace('"', "")

    def _table_exists(self, table_name):
        try:
            self.db_engine.execute(
                'select * from {}.{} limit 1'.format(
                    self.features_schema_name,
                    table_name
                )
            )
            return True
        except sqlalchemy.exc.ProgrammingError:
            return False

    def run_commands(self, command_list):
        conn = self.db_engine.connect()
        trans = conn.begin()
        for command in command_list:
            conn.execute(command)
        trans.commit()


    def _aggregation_index_query(self, aggregation):
        return 'CREATE INDEX ON {} ({}, {})'.format(
            aggregation.get_table_name(),
            self.entity_id_column,
            aggregation.output_date_column
        )

    def _aggregation_index_columns(self, aggregation):
        return sorted([group for group in aggregation.groups.keys()] + [aggregation.output_date_column])

    def index_column_lookup(self, aggregations):
        return dict((
            self._clean_table_name(aggregation.get_table_name()),
            self._aggregation_index_columns(aggregation)
        ) for aggregation in aggregations)

    def _generate_table_tasks_for(self, aggregation):
        """Generates SQL commands for preparing, populating, and finalizing
        each feature group table in the given aggregation

        Args:
            aggregation (collate.SpacetimeAggregation)

        Returns: (dict) of structure {
            'prepare': list of commands to prepare table for population
            'inserts': list of commands to populate table
            'finalize': list of commands to finalize table after population
        }
        """
        create_schema = aggregation.get_create_schema()
        creates = aggregation.get_creates()
        drops = aggregation.get_drops()
        indexes = aggregation.get_indexes()
        inserts = aggregation.get_inserts()

        if create_schema is not None:
            self.db_engine.execute(create_schema)

        table_tasks = OrderedDict()
        for group in aggregation.groups:
            group_table = self._clean_table_name(
                aggregation.get_table_name(group=group)
            )
            if self.replace or not self._table_exists(group_table):
                table_tasks[group_table] = {
                    'prepare': [drops[group], creates[group]],
                    'inserts': inserts[group],
                    'finalize': [indexes[group]],
                }
                logging.info('Created table tasks for %s', group_table)
            else:
                logging.info(
                    'Skipping feature table creation for %s',
                    group_table
                )
                table_tasks[group_table] = {}
        logging.info('Created table tasks for aggregation')
        table_tasks[self._clean_table_name(aggregation.get_table_name())] = {
            'prepare': [aggregation.get_drop(), aggregation.get_create()],
            'inserts': [],
            'finalize': [self._aggregation_index_query(aggregation)],
        }

        return table_tasks
