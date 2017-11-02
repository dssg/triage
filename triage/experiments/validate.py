import logging
import architect
from architect.utils import convert_str_to_relativedelta
from triage.validation_primitives import table_should_have_data,\
    column_should_be_intlike,\
    column_should_be_booleanlike,\
    column_should_be_stringlike,\
    column_should_be_timelike


class Validator(object):
    def __init__(self, db_engine):
        self.db_engine = db_engine


class FeatureAggregationsValidator(Validator):
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
        if not any(group == 'entity_id' for group in groups):
            raise ValueError('entity_id needs to be present in each feature_aggregation\'s list of groups')

    def _validate_aggregation(self, aggregation_config):
        logging.info('Validating aggregation config %s', aggregation_config)
        self._validate_keys(aggregation_config)
        self._validate_aggregates(aggregation_config)
        self._validate_categoricals(aggregation_config.get('categoricals', []))
        self._validate_from_obj(aggregation_config['from_obj'])
        self._validate_time_intervals(aggregation_config['intervals'])
        self._validate_groups(aggregation_config['groups'])

    def run(self, feature_aggregation_config):
        """Validate a feature aggregation config applied to this object

        The validations range from basic type checks, key presence checks,
        as well as validating the sql in from objects.

        Args:
            feature_aggregation_config (list) all values, except for feature
                date, necessary to instantiate a collate.SpacetimeAggregation

        Raises: ValueError if any part of the config is found to be invalid
        """
        if not feature_aggregation_config:
            raise ValueError("configuration key 'feature_aggregations' not found")
        for aggregation in feature_aggregation_config:
            self._validate_aggregation(aggregation)

class EventsTableValidator(Validator):
    def run(self, events_table):
        if not events_table:
            raise ValueError("configuration key 'events_table' not found")
        table_should_have_data(events_table, self.db_engine)
        column_should_be_intlike(events_table, 'entity_id', self.db_engine)
        column_should_be_timelike(events_table, 'outcome_date', self.db_engine)
        column_should_be_booleanlike(events_table, 'outcome', self.db_engine)


class StateConfigValidator(Validator):
    def run(self, state_config):
        if 'table_name' in state_config:
            dense_state_table = state_config['table_name']
            table_should_have_data(dense_state_table, self.db_engine)
            column_should_be_intlike(dense_state_table, 'entity_id', self.db_engine)
            column_should_be_stringlike(dense_state_table, 'state', self.db_engine)
            column_should_be_timelike(dense_state_table, 'start_time', self.db_engine)
            column_should_be_timelike(dense_state_table, 'end_time', self.db_engine)
            if 'state_filters' not in state_config or len(state_config['state_filters']) < 1:
                raise ValueError('If a table_name is given in state_config, at least one state filter must be present')


class FeatureGroupDefinitionValidator(Validator):
    def run(self, feature_group_definition, feature_aggregation_config):
        if not isinstance(feature_group_definition, dict):
            raise ValueError('Feature Group Definition must be a dictionary')

        available_subsetters = architect.feature_group_creator.FeatureGroupCreator.subsetters
        for subsetter_name, value in feature_group_definition.items():
            if subsetter_name not in available_subsetters:
                raise ValueError('''Unknown feature_group_definition key {} received.
                Available keys are {}'''.format(subsetter_name, available_subsetters))
            if not hasattr(value, '__iter__') or isinstance(value, (str, bytes)):
                raise ValueError('Each value in FeatureGroupCreator must be iterable and not a string')

        if 'prefix' in feature_group_definition:
            available_prefixes = [
                aggregation['prefix']
                for aggregation in feature_aggregation_config
            ]
            for prefix in feature_group_definition['prefix']:
                if prefix not in available_prefixes:
                    raise ValueError('''Aggregation prefix of '{}' was given as a
                    feature group, but no such prefix exists in the available
                    feature aggregations. The available feature aggregations are {}
                    '''.format(prefix, available_prefixes))

class FeatureGroupStrategyValidator(Validator):
    def run(self, feature_group_strategies):
        if not isinstance(feature_group_strategies, list):
            raise ValueError('feature_group_strategies key must be a list')
        available_strategies = architect.feature_group_mixer.FeatureGroupMixer.strategy_lookup
        for strategy in feature_group_strategies:
            if strategy not in available_strategies:
                raise ValueError('''Unknown feature_group_strategies key {} received.
                Available keys are {}'''.format(strategy, available_strategies))


class UserMetadataValidator(Validator):
    def run(self, user_metadata):
        if not isinstance(user_metadata, dict):
            raise ValueError('user_metadata key must be a dict')


class ModelGroupKeysValidator(Validator):
    def run(self, model_group_keys, user_metadata):
        if not isinstance(model_group_keys, list):
            raise ValueError('model_group_keys must be a list')
        # planner_keys are defined in architect.Planner._make_metadata
        planner_keys = [
            'beginning_of_time',
            'end_time',
            'indices',
            'feature_names',
            'label_name',
            'label_type',
            'state',
            'matrix_id',
            'matrix_type'
        ]
        # temporal_keys are defined in timechop.Timechop.generate_matrix_definition
        temporal_keys = [
            'matrix_start_time',
            'matrix_end_time',
            'as_of_times',
            'label_window',
            'example_frequency',
            'train_duration'
        ]
        available_keys = [key for key in user_metadata.keys()] + planner_keys + temporal_keys
        for model_group_key in model_group_keys:
            if model_group_key not in available_keys:
                raise ValueError('''Unknown model_group_keys entry '{}' received.
                Available keys are {}'''.format(model_group_key, available_keys))


class GridConfigValidator(Validator):
    def run(self, grid_config):
        pass


class ScoringConfigValidator(Validator):
    def run(self, scoring_config):
        pass


class ExperimentValidator(Validator):
    def run(self, experiment_config):
        FeatureAggregationsValidator(self.db_engine).run(experiment_config.get('feature_aggregations', {}))
        EventsTableValidator(self.db_engine).run(experiment_config.get('events_table', None))
        StateConfigValidator(self.db_engine).run(experiment_config.get('state_config', {}))
        FeatureGroupDefinitionValidator(self.db_engine).run(
            experiment_config.get('feature_group_definition', {}),
            experiment_config['feature_aggregations']
        )
        FeatureGroupStrategyValidator(self.db_engine).run(
            experiment_config.get('feature_group_strategies', []),
        )
        UserMetadataValidator(self.db_engine).run(
            experiment_config.get('user_metadata', {})
        )
        ModelGroupKeysValidator(self.db_engine).run(
            experiment_config.get('model_group_keys', []),
            experiment_config.get('user_metadata', {})
        )
        GridConfigValidator(self.db_engine).run(experiment_config.get('grid_config', {}))
        ScoringConfigValidator(self.db_engine).run(experiment_config.get('scoring', {}))

        # show the success message in the console as well as the logger
        # as we don't really know how they have configured logging
        success_message = 'Experiment validation ran to completion with no errors'
        logging.info(success_message)
        print(success_message)
