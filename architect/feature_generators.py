from collate.collate import Aggregate, Categorical, Compare
from collate.spacetime import SpacetimeAggregation
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

    def validate(self, feature_aggregation_config, feature_dates):
        aggregations = self.aggregations(feature_aggregation_config, feature_dates)
        self._explain_selects(aggregations)

    def generate_all_table_tasks(self, feature_aggregation_config, feature_dates):
        """Generates SQL commands for creating, populating, and indexing
        feature group tables

        Args:
            feature_aggregation_config (list) all values, except for feature
                date, necessary to instantiate a collate.SpacetimeAggregation
            feature_dates (list) dates to generate features as of

        Returns: (dict) keys are group table names, values are themselves dicts,
            each with keys for different stages of table creation (prepare, inserts, finalize)
            and with values being lists of SQL commands
        """
        logging.debug('---------------------')
        logging.debug('---------FEATURE GENERATION------------')
        logging.debug('---------------------')
        table_tasks = {}
        aggregations = self.aggregations(feature_aggregation_config, feature_dates)
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
            feature_aggregation_config,
            feature_dates
        )

        return self.process_table_tasks(table_tasks)

    def process_table_tasks(self, table_tasks):
        for table_name, task in table_tasks.items():
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

        table_tasks = {}
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
            else:
                logging.info(
                    'Skipping feature table creation for %s',
                    group_table
                )
                table_tasks[group_table] = {}

        return table_tasks
