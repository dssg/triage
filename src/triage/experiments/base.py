import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial

from descriptors import cachedproperty
from timeout import timeout

from triage.component.architect.label_generators import LabelGenerator, DEFAULT_LABEL_NAME

from triage.component.architect.features import (
    FeatureGenerator,
    FeatureDictionaryCreator,
    FeatureGroupCreator,
    FeatureGroupMixer,
)
from triage.component.architect.planner import Planner
from triage.component.architect.builders import HighMemoryCSVBuilder
from triage.component.architect.state_table_generators import (
    StateTableGeneratorFromDense,
    StateTableGeneratorFromEntities,
    StateTableGeneratorFromQuery
)
from triage.component.timechop import Timechop
from triage.component.catwalk.db import ensure_db
from triage.component.catwalk.model_grouping import ModelGrouper
from triage.component.catwalk.model_trainers import ModelTrainer
from triage.component.catwalk.predictors import Predictor
from triage.component.catwalk.individual_importance import IndividualImportanceCalculator
from triage.component.catwalk.evaluation import ModelEvaluator
from triage.component.catwalk.utils import save_experiment_and_get_hash
from triage.component.catwalk.storage import CSVMatrixStore

from triage.experiments import CONFIG_VERSION
from triage.experiments.validate import ExperimentValidator


def dt_from_str(dt_str):
    return datetime.strptime(dt_str, '%Y-%m-%d')


class ExperimentBase(ABC):
    """The base class for all Experiments."""

    cleanup_timeout = 60  # seconds

    def __init__(
        self,
        config,
        db_engine,
        model_storage_class=None,
        project_path=None,
        replace=True,
        cleanup_timeout=None,
    ):
        self._check_config_version(config)
        self.config = config

        self.db_engine = db_engine
        if model_storage_class:
            self.model_storage_engine = model_storage_class(
                project_path=project_path)
        self.matrix_store_class = CSVMatrixStore  # can't be configurable until Architect obeys
        self.project_path = project_path
        self.replace = replace
        ensure_db(self.db_engine)

        self.labels_table_name = 'labels'
        self.features_schema_name = 'features'
        if project_path:
            self.matrices_directory = os.path.join(self.project_path, 'matrices')
            if not os.path.exists(self.matrices_directory):
                os.makedirs(self.matrices_directory)

        self.experiment_hash = save_experiment_and_get_hash(self.config,
                                                            self.db_engine)
        self.initialize_factories()
        self.initialize_components()

        self.cleanup_timeout = (self.cleanup_timeout if cleanup_timeout is None
                                else cleanup_timeout)

    def _check_config_version(self, config):
        if 'config_version' in config:
            config_version = config['config_version']
        else:
            logging.warning('config_version key not found in experiment config. '
                            'Assuming v1, which may not be correct')
            config_version = 'v1'
        if config_version != CONFIG_VERSION:
            raise ValueError(
                "Experiment config '{}' "
                "does not match current version '{}'. "
                "Will not run experiment."
                .format(config_version, CONFIG_VERSION)
            )

    def initialize_factories(self):
        split_config = self.config['temporal_config']

        self.chopper_factory = partial(
            Timechop,
            feature_start_time=dt_from_str(split_config['feature_start_time']),
            feature_end_time=dt_from_str(split_config['feature_end_time']),
            label_start_time=dt_from_str(split_config['label_start_time']),
            label_end_time=dt_from_str(split_config['label_end_time']),
            model_update_frequency=split_config['model_update_frequency'],
            training_label_timespans=split_config['training_label_timespans'],
            test_label_timespans=split_config['test_label_timespans'],
            training_as_of_date_frequencies=split_config['training_as_of_date_frequencies'],
            test_as_of_date_frequencies=split_config['test_as_of_date_frequencies'],
            max_training_histories=split_config['max_training_histories'],
            test_durations=split_config['test_durations'],
        )

        cohort_config = self.config.get('cohort_config', {})
        if 'query' in cohort_config:
            self.state_table_generator_factory = partial(
                StateTableGeneratorFromQuery,
                experiment_hash=self.experiment_hash,
                query=cohort_config['query']
            )
        elif 'entities_table' in cohort_config:
            self.state_table_generator_factory = partial(
                StateTableGeneratorFromEntities,
                experiment_hash=self.experiment_hash,
                entities_table=cohort_config['entities_table']
            )
        elif 'dense_states' in cohort_config:
            self.state_table_generator_factory = partial(
                StateTableGeneratorFromDense,
                experiment_hash=self.experiment_hash,
                dense_state_table=cohort_config['dense_states']['table_name']
            )
        else:
            raise ValueError('Cohort config missing or unrecognized')

        self.label_generator_factory = partial(
            LabelGenerator,
            label_name=self.config['label_config'].get('name', None),
            query=self.config['label_config']['query']
        )

        self.feature_dictionary_creator_factory = partial(
            FeatureDictionaryCreator,
            features_schema_name=self.features_schema_name,
        )

        self.feature_generator_factory = partial(
            FeatureGenerator,
            features_schema_name=self.features_schema_name,
            replace=self.replace,
            feature_start_time=split_config['feature_start_time']
        )

        self.feature_group_creator_factory = partial(
            FeatureGroupCreator,
            self.config.get('feature_group_definition', {'all': [True]})
        )

        self.feature_group_mixer_factory = partial(
            FeatureGroupMixer,
            self.config.get('feature_group_strategies', ['all'])
        )

        self.planner_factory = partial(
            Planner,
            feature_start_time=dt_from_str(split_config['feature_start_time']),
            label_names=[self.config.get('label_config', {}).get('name', DEFAULT_LABEL_NAME)],
            label_types=['binary'],
            matrix_directory=self.matrices_directory,
            cohort_name=self.config.get('cohort_config', {}).get('name', None),
            states=self.config.get('cohort_config', {}).get('dense_states', {})
            .get('state_filters', []),
            user_metadata=self.config.get('user_metadata', {}),
        )

        self.matrix_builder_factory = partial(
            HighMemoryCSVBuilder,
            db_config={
                'features_schema_name': self.features_schema_name,
                'labels_schema_name': 'public',
                'labels_table_name': self.labels_table_name,
                # TODO: have planner/builder take state table later on, so we
                # can grab it from the StateTableGenerator instead of
                # duplicating it here
                'sparse_state_table_name': 'tmp_sparse_states_{}'
                                           .format(self.experiment_hash),
            },
            matrix_directory=self.matrices_directory,
            include_missing_labels_in_train_as=self.config['label_config']
            .get('include_missing_labels_in_train_as', None),
            replace=self.replace
        )

        self.trainer_factory = partial(
            ModelTrainer,
            project_path=self.project_path,
            experiment_hash=self.experiment_hash,
            model_storage_engine=self.model_storage_engine,
            model_grouper=ModelGrouper(self.config.get('model_group_keys', [])),
            replace=self.replace
        )

        self.predictor_factory = partial(
            Predictor,
            model_storage_engine=self.model_storage_engine,
            project_path=self.project_path,
            replace=self.replace
        )

        self.indiv_importance_factory = partial(
            IndividualImportanceCalculator,
            n_ranks=self.config.get('individual_importance', {}).get('n_ranks', 5),
            methods=self.config.get('individual_importance', {}).get('methods', ['uniform']),
            replace=self.replace
        )

        self.evaluator_factory = partial(
            ModelEvaluator,
            sort_seed=self.config['scoring'].get('sort_seed', None),
            metric_groups=self.config['scoring']['metric_groups'],
        )

    def initialize_components(self):
        self.chopper = self.chopper_factory()
        self.label_generator = self.label_generator_factory(db_engine=self.db_engine)
        self.state_table_generator = self.state_table_generator_factory(db_engine=self.db_engine)
        self.feature_generator = self.feature_generator_factory(db_engine=self.db_engine)
        self.feature_dictionary_creator = self.feature_dictionary_creator_factory(
            db_engine=self.db_engine)
        self.feature_group_creator = self.feature_group_creator_factory()
        self.feature_group_mixer = self.feature_group_mixer_factory()
        self.planner = self.planner_factory()
        self.matrix_builder = self.matrix_builder_factory(engine=self.db_engine)
        self.trainer = self.trainer_factory(db_engine=self.db_engine)
        self.predictor = self.predictor_factory(db_engine=self.db_engine)
        self.individual_importance_calculator = self.indiv_importance_factory(
            db_engine=self.db_engine)
        self.evaluator = self.evaluator_factory(db_engine=self.db_engine)

    @cachedproperty
    def split_definitions(self):
        """Temporal splits based on the experiment's configuration

        Returns: (dict) temporal splits

        Example:
        ```
        {
            'feature_start_time': {datetime},
            'feature_end_time': {datetime},
            'label_start_time': {datetime},
            'label_end_time': {datetime},
            'train_matrix': {
                'first_as_of_time': {datetime},
                'last_as_of_time': {datetime},
                'matrix_info_end_time': {datetime},
                'training_label_timespan': {str},
                'training_as_of_date_frequency': {str},
                'max_training_history': {str},
                'as_of_times': [list of {datetime}s]
            },
            'test_matrices': [list of matrix defs similar to train_matrix]
        }
        ```

        (When updating/setting split definitions, matrices should have
        UUIDs.)

        """
        split_definitions = self.chopper.chop_time()
        logging.info('Computed and stored split definitions: %s',
                     split_definitions)
        return split_definitions

    def print_time_split_summary(self):
        print('\n----TIME SPLIT SUMMARY----\n')
        print('Number of time splits: {}'.format(len(self.split_definitions)))
        for split_index, split in enumerate(self.split_definitions):
            train_times = split['train_matrix']['as_of_times']
            test_times = [as_of_time for test_matrix in split['test_matrices']
                          for as_of_time in test_matrix['as_of_times']]
            print('''Split index {}:
            Training as_of_time_range: {} to {} ({} total)
            Testing as_of_time range: {} to {} ({} total)\n\n'''.format(
                split_index,
                min(train_times),
                max(train_times),
                len(train_times),
                min(test_times),
                max(test_times),
                len(test_times)
            ))
        print('For more detailed information on your time splits, '
              'inspect the experiment `split_definitions` property')

    @cachedproperty
    def all_as_of_times(self):
        """All 'as of times' in experiment config

        Used for label and feature generation.

        Returns: (list) of datetimes

        """
        all_as_of_times = []
        for split in self.split_definitions:
            all_as_of_times.extend(split['train_matrix']['as_of_times'])
            logging.info('Adding as_of_times from train matrix: %s',
                         split['train_matrix']['as_of_times'])
            for test_matrix in split['test_matrices']:
                logging.info('Adding as_of_times from test matrix: %s',
                             test_matrix['as_of_times'])
                all_as_of_times.extend(test_matrix['as_of_times'])

        logging.info(
            'Computed %s total as_of_times for label and feature generation',
            len(all_as_of_times)
        )
        distinct_as_of_times = list(set(all_as_of_times))
        logging.info(
            'Computed %s distinct as_of_times for label and feature generation',
            len(distinct_as_of_times)
        )
        return distinct_as_of_times

    @cachedproperty
    def collate_aggregations(self):
        """Collation of ``Aggregation`` objects used by this experiment.

        Returns: (list) of ``collate.Aggregation`` objects

        """
        logging.info('Creating collate aggregations')
        return self.feature_generator.aggregations(
            feature_aggregation_config=self.config['feature_aggregations'],
            feature_dates=self.all_as_of_times,
            state_table=self.state_table_generator.sparse_table_name,
        )

    @cachedproperty
    def feature_aggregation_table_tasks(self):
        """All feature table query tasks specified by this
        ``Experiment``.

        Returns: (dict) keys are group table names, values are
            themselves dicts, each with keys for different stages of
            table creation (prepare, inserts, finalize) and with values
            being lists of SQL commands

        """
        logging.info('Calculating feature tasks for %s as_of_times',
                     len(self.all_as_of_times))
        return self.feature_generator.generate_all_table_tasks(
            self.collate_aggregations,
            task_type='aggregation'
        )

    @cachedproperty
    def feature_imputation_table_tasks(self):
        """All feature imputation query tasks specified by this
        ``Experiment``.

        Returns: (dict) keys are group table names, values are
            themselves dicts, each with keys for different stages of
            table creation (prepare, inserts, finalize) and with values
            being lists of SQL commands

        """
        logging.info('Calculating feature tasks for %s as_of_times',
                     len(self.all_as_of_times))
        return self.feature_generator.generate_all_table_tasks(
            self.collate_aggregations,
            task_type='imputation'
        )

    @cachedproperty
    def master_feature_dictionary(self):
        """All possible features found in the database. Not all features
        will necessarily end up in matrices

        Returns: (list) of dicts, keys being feature table names and
        values being lists of feature names

        """
        result = self.feature_dictionary_creator.feature_dictionary(
            feature_table_names=self.feature_imputation_table_tasks.keys(),
            index_column_lookup=self.feature_generator.index_column_lookup(
                self.collate_aggregations
            )
        )
        logging.info('Computed master feature dictionary: %s', result)
        return result

    @property
    def feature_dicts(self):
        """Feature dictionaries, representing the feature tables and
        columns configured in this experiment after computing feature
        groups.

        Returns: (list) of dicts, keys being feature table names and
        values being lists of feature names

        """
        return self.feature_group_mixer.generate(
            self.feature_group_creator.subsets(self.master_feature_dictionary)
        )

    @cachedproperty
    def matrix_build_tasks(self):
        """Tasks for all matrices that need to be built as a part of
        this Experiment.

        Each task contains arguments understood by
        ``Architect.build_matrix``.

        Returns: (list) of dicts

        """
        (
            updated_split_definitions,
            matrix_build_tasks
        ) = self.planner.generate_plans(
            self.split_definitions,
            self.feature_dicts
        )
        self.full_matrix_definitions = updated_split_definitions
        return matrix_build_tasks

    @cachedproperty
    def full_matrix_definitions(self):
        """Full matrix definitions

        Returns: (list) temporal and feature information for each matrix

        """
        (
            updated_split_definitions,
            matrix_build_tasks
        ) = self.planner.generate_plans(
            self.split_definitions,
            self.feature_dicts
        )
        self.matrix_build_tasks = matrix_build_tasks
        return updated_split_definitions

    @property
    def all_label_timespans(self):
        """All train and test label timespans

        Returns: (list) label timespans, in string form as they appeared in the experiment config

        """
        return list(set(
            self.config['temporal_config']['training_label_timespans'] +
            self.config['temporal_config']['test_label_timespans']
        ))

    def generate_labels(self):
        """Generate labels based on experiment configuration

        Results are stored in the database, not returned
        """
        self.label_generator.generate_all_labels(
            self.labels_table_name,
            self.all_as_of_times,
            self.all_label_timespans
        )

    def generate_sparse_states(self):
        self.state_table_generator.generate_sparse_table(
            as_of_dates=self.all_as_of_times
        )

    def log_split(self, split_num, split):
        logging.info(
            'Starting train/test for %s out of %s: train range: %s to %s',
            split_num+1,
            len(self.full_matrix_definitions),
            split['train_matrix']['first_as_of_time'],
            split['train_matrix']['matrix_info_end_time'],
        )

    def matrix_store(self, matrix_uuid):
        """Construct a matrix store for a given matrix uuid, using the Experiment's #matrix_store_class

        Args:
            matrix_uuid (string) A uuid for a matrix
        """
        matrix_store = self.matrix_store_class(
            matrix_path=os.path.join(
                self.matrices_directory,
                '{}.csv'.format(matrix_uuid)
            ),
            metadata_path=os.path.join(
                self.matrices_directory,
                '{}.yaml'.format(matrix_uuid)
            )
        )
        return matrix_store

    @abstractmethod
    def build_matrices(self):
        """Generate labels, features, and matrices"""
        pass

    @abstractmethod
    def catwalk(self):
        """Train, test, and evaluate models"""
        pass

    def validate(self):
        ExperimentValidator(self.db_engine).run(self.config)
        self.print_time_split_summary()

    def _run(self):
        try:
            logging.info('Building matrices')
            self.build_matrices()
        finally:
            logging.info('Cleaning up state table')
            with timeout(self.cleanup_timeout):
                self.state_table_generator.clean_up()

        self.catwalk()

    def run(self):
        try:
            self._run()
        except Exception:
            logging.exception('Run interrupted by uncaught exception')
            raise

    __call__ = run
