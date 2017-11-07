import logging
from datetime import datetime
from timechop.timechop import Timechop
import architect
from architect.utils import convert_str_to_relativedelta
import catwalk
from triage.validation_primitives import table_should_have_data,\
    column_should_be_intlike,\
    column_should_be_booleanlike,\
    column_should_be_stringlike,\
    column_should_be_timelike
import importlib
from sklearn.model_selection import ParameterGrid


class Validator(object):
    def __init__(self, db_engine=None):
        self.db_engine = db_engine


class TemporalValidator(Validator):
    def run(self, temporal_config):
        def dt_from_str(dt_str):
            return datetime.strptime(dt_str, '%Y-%m-%d')
        splits = []
        try:
            chopper = Timechop(
                feature_start_time=dt_from_str(temporal_config['feature_start_time']),
                feature_end_time=dt_from_str(temporal_config['feature_end_time']),
                label_start_time=dt_from_str(temporal_config['label_start_time']),
                label_end_time=dt_from_str(temporal_config['label_end_time']),
                model_update_frequency=temporal_config['model_update_frequency'],
                training_label_timespans=temporal_config['training_label_timespans'],
                test_label_timespans=temporal_config['test_label_timespans'],
                training_as_of_date_frequencies=temporal_config['training_as_of_date_frequencies'],
                test_as_of_date_frequencies=temporal_config['test_as_of_date_frequencies'],
                max_training_histories=temporal_config['max_training_histories'],
                test_durations=temporal_config['test_durations'],
            )
            splits = chopper.chop_time()
        except Exception as e:
            raise ValueError('''Section: temporal_config -
            Timechop could not produce temporal splits from config {}.
            Error: {}
            '''.format(temporal_config, e))
        for split in splits:
            if len(split['train_matrix']['as_of_times']) == 0:
                raise ValueError('''Section: temporal_config -
                Computed split {} has a train matrix with no as_of_times.
                '''.format(split))
            for test_matrix in split['test_matrices']:
                if len(test_matrix['as_of_times']) == 0:
                    raise ValueError('''Section: temporal_config -
                    Computed split {} has a test matrix with no as_of_times.
                    '''.format(split))


class FeatureAggregationsValidator(Validator):
    def _validate_keys(self, aggregation_config):
        for key in [
            'from_obj',
            'intervals',
            'groups',
            'knowledge_date_column',
            'prefix'
        ]:
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
            if 'choice_query' in categorical and 'choices' in categorical:
                raise ValueError('''Section: feature_aggregations -
                Both 'choice_query' and 'choices' specified for {}.
                Please only specify one.'''.format(categorical))
            if not ('choice_query' in categorical or 'choices' in categorical):
                raise ValueError('''Section: feature_aggregations -
                Neither 'choice_query' and 'choices' specified for {}.
                Please specify one.'''.format(categorical))
            if 'choice_query' in categorical:
                logging.info('Validating choice query')
                choice_query = categorical['choice_query']
                try:
                    conn.execute('explain {}'.format(choice_query))
                except Exception as e:
                    raise ValueError('''Section: feature_aggregations -
                    choice query does not run.
                    choice query: "{}"
                    Full error: {}'''.format(choice_query, e))

    def _validate_from_obj(self, from_obj):
        conn = self.db_engine.connect()
        logging.info('Validating from_obj')
        try:
            conn.execute('explain select * from {}'.format(from_obj))
        except Exception as e:
            raise ValueError('''Section: feature_aggregations -
                from_obj query does not run.
                from_obj: "{}"
                Full error: {}'''.format(from_obj, e))

    def _validate_time_intervals(self, intervals):
        logging.info('Validating time intervals')
        for interval in intervals:
            if interval != 'all':
                # this function, used elsewhere to break up time intervals,
                # will throw an error if the interval can't be converted to a
                # relativedelta
                try:
                    convert_str_to_relativedelta(interval)
                except Exception as e:
                    raise ValueError('''Section: feature_aggregations -
                    Time interval is invalid.
                    interval: "{}"
                    Full error: {}'''.format(interval, e))

    def _validate_groups(self, groups):
        if 'entity_id' not in groups:
            raise ValueError(''''Section: feature_aggregations -
            List of groups needs to include 'entity_id'.
            Passed list: {}'''.format(groups))

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
            raise ValueError('''Section: feature_aggregations -
            Section not found. You must define feature aggregations.''')
        for aggregation in feature_aggregation_config:
            self._validate_aggregation(aggregation)


class EventsTableValidator(Validator):
    def run(self, events_table):
        if not events_table:
            raise ValueError('''Section: events_table -
            Section not found. You must define an events table.''')
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
                raise ValueError('''Section: state_config -
                If a table_name is given in state_config,
                at least one state filter must be present''')
        else:
            logging.warning('No table_name found in state_config.' +
                            'The provided events table will be used, which ' +
                            'may result in unnecessarily large matrices')


class FeatureGroupDefinitionValidator(Validator):
    def run(self, feature_group_definition, feature_aggregation_config):
        if not isinstance(feature_group_definition, dict):
            raise ValueError('''Section: feature_group_definition -
            feature_group_definition must be a dictionary''')

        available_subsetters = architect.feature_group_creator.FeatureGroupCreator.subsetters
        for subsetter_name, value in feature_group_definition.items():
            if subsetter_name not in available_subsetters:
                raise ValueError('''Section: feature_group_definition -
                Unknown feature_group_definition key {} received.
                Available keys are {}'''.format(subsetter_name, available_subsetters))
            if not hasattr(value, '__iter__') or isinstance(value, (str, bytes)):
                raise ValueError('''Section: feature_group_definition -
                feature_group_definition value for {}, {}
                should be a list'''.format(subsetter_name, value))

        if 'prefix' in feature_group_definition:
            available_prefixes = {
                aggregation['prefix']
                for aggregation in feature_aggregation_config
            }
            bad_prefixes = set(feature_group_definition['prefix']) - available_prefixes
            if bad_prefixes:
                raise ValueError('''Section: feature_group_definition -
                The following given feature group prefixes: '{}'
                are invalid. Available prefixes from this experiment's feature
                aggregations are: '{}'
                '''.format(bad_prefixes, available_prefixes))

        if 'tables' in feature_group_definition:
            available_tables = {
                aggregation['prefix'] + '_aggregation'
                for aggregation in feature_aggregation_config
            }
            bad_tables = set(feature_group_definition['tables']) - available_tables
            if bad_tables:
                raise ValueError('''Section: feature_group_definition -
                The following given feature group tables: '{}'
                are invalid. Available tables from this experiment's feature
                aggregations are: '{}'
                '''.format(bad_tables, available_tables))


class FeatureGroupStrategyValidator(Validator):
    def run(self, feature_group_strategies):
        if not isinstance(feature_group_strategies, list):
            raise ValueError('''Section: feature_group_strategies -
            feature_group_strategies section must be a list''')
        available_strategies = {
            key for key in
            architect.feature_group_mixer.FeatureGroupMixer.strategy_lookup.keys()
        }
        bad_strategies = set(feature_group_strategies) - available_strategies
        if bad_strategies:
            raise ValueError('''Section: feature_group_strategies -
            The following given feature group strategies:
            '{}' are invalid. Available strategies are: '{}'
            '''.format(bad_strategies, available_strategies))


class UserMetadataValidator(Validator):
    def run(self, user_metadata):
        if not isinstance(user_metadata, dict):
            raise ValueError('''Section: user_metadata -
            user_metadata section must be a dict''')


class ModelGroupKeysValidator(Validator):
    def run(self, model_group_keys, user_metadata):
        if not isinstance(model_group_keys, list):
            raise ValueError('''Section: model_group_keys -
            model_group_keys section must be a list''')
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
        # temporal_keys are defined in
        # timechop.Timechop.generate_matrix_definition
        temporal_keys = [
            'matrix_start_time',
            'matrix_end_time',
            'as_of_times',
            'label_window',
            'example_frequency',
            'train_duration'
        ]
        available_keys = [key for key in user_metadata.keys()] + \
            planner_keys +\
            temporal_keys
        for model_group_key in model_group_keys:
            if model_group_key not in available_keys:
                raise ValueError('''Section: model_group_keys -
                unknown entry '{}' received. Available keys are {}
                '''.format(model_group_key, available_keys))


class GridConfigValidator(Validator):
    def run(self, grid_config):
        for classpath, parameter_config in grid_config.items():
            try:
                module_name, class_name = classpath.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                for parameters in ParameterGrid(parameter_config):
                    try:
                        cls(**parameters)
                    except Exception as e:
                        raise ValueError('''Section: grid_config -
                        Unable to instantiate classifier {} with parameters {}, error thrown: {}
                        '''.format(classpath, parameters, e))
            except Exception as e:
                raise ValueError('''Section: grid_config -
                Unable to import classifier {}, error thrown: {}
                '''.format(classpath, e))


class ScoringConfigValidator(Validator):
    def run(self, scoring_config):
        if 'metric_groups' not in scoring_config:
            logging.warning('Section: scoring - No metric_groups configured. ' +
                            'Your experiment may run, but you will not have any ' +
                            'evaluation metrics computed'
                            )
        metric_lookup = catwalk.evaluation.ModelEvaluator.available_metrics
        available_metrics = set(metric_lookup.keys())
        for metric_group in scoring_config['metric_groups']:
            given_metrics = set(metric_group['metrics'])
            bad_metrics = given_metrics - available_metrics
            if bad_metrics:
                raise ValueError('''Section: scoring -
                The following given metrics '{}' are unavailable. Available metrics are: '{}'
                '''.format(bad_metrics, available_metrics))
            for given_metric in given_metrics:
                metric_function = metric_lookup[given_metric]
                if not hasattr(metric_function, 'greater_is_better'):
                    raise ValueError('''Section: scoring -
                    The metric {} does not define the attribute
                    'greater_is_better'. This can only be fixed in the catwalk.metrics
                    module. If you still would like to use this metric, consider
                    submitting a pull request'''.format(given_metric))


class ExperimentValidator(Validator):
    def run(self, experiment_config):
        TemporalValidator().run(experiment_config.get('temporal_config', {}))
        FeatureAggregationsValidator(self.db_engine)\
            .run(experiment_config.get('feature_aggregations', {}))
        EventsTableValidator(self.db_engine).run(experiment_config.get('events_table', None))
        StateConfigValidator(self.db_engine).run(experiment_config.get('state_config', {}))
        FeatureGroupDefinitionValidator().run(
            experiment_config.get('feature_group_definition', {}),
            experiment_config['feature_aggregations']
        )
        FeatureGroupStrategyValidator().run(
            experiment_config.get('feature_group_strategies', []),
        )
        UserMetadataValidator().run(
            experiment_config.get('user_metadata', {})
        )
        ModelGroupKeysValidator().run(
            experiment_config.get('model_group_keys', []),
            experiment_config.get('user_metadata', {})
        )
        GridConfigValidator().run(experiment_config.get('grid_config', {}))
        ScoringConfigValidator().run(experiment_config.get('scoring', {}))

        # show the success message in the console as well as the logger
        # as we don't really know how they have configured logging
        success_message = 'Experiment validation ran to completion with no errors'
        logging.info(success_message)
        print(success_message)
