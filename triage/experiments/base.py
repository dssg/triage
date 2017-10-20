from catwalk.db import ensure_db
from timechop.timechop import Timechop
from architect.label_generators import BinaryLabelGenerator
from architect.features import \
    FeatureGenerator,\
    FeatureDictionaryCreator,\
    FeatureGroupCreator,\
    FeatureGroupMixer
from architect.state_table_generators import StateTableGenerator
from architect.planner import Planner
from catwalk.model_trainers import ModelTrainer
from catwalk.predictors import Predictor
from catwalk.individual_importance import IndividualImportanceCalculator
from catwalk.evaluation import ModelEvaluator
from catwalk.utils import save_experiment_and_get_hash
from catwalk.storage import CSVMatrixStore
import os
from datetime import datetime
from abc import ABCMeta, abstractmethod
from functools import partial
import logging
from triage.experiments import CONFIG_VERSION


def dt_from_str(dt_str):
    return datetime.strptime(dt_str, '%Y-%m-%d')


class ExperimentBase(object):
    """The Base class for all Experiments."""
    __metaclass__ = ABCMeta

    def __init__(
        self,
        config,
        db_engine,
        model_storage_class=None,
        project_path=None,
        replace=True,
    ):
        self._check_config_version(config)
        self.config = config
        self.db_engine = db_engine
        if model_storage_class:
            self.model_storage_engine =\
                model_storage_class(project_path=project_path)
        self.matrix_store_class = CSVMatrixStore # can't make this configurable yet until the Architect obeys
        self.project_path = project_path
        self.replace = replace
        ensure_db(self.db_engine)

        self.labels_table_name = 'labels'
        self.features_schema_name = 'features'
        if project_path:
            self.matrices_directory = os.path.join(self.project_path, 'matrices')
            if not os.path.exists(self.matrices_directory):
                os.makedirs(self.matrices_directory)
        self.experiment_hash = save_experiment_and_get_hash(
            self.config,
            self.db_engine
        )
        self._split_definitions = None
        self._matrix_build_tasks = None
        self._collate_aggregations = None
        self._feature_aggregation_table_tasks = None
        self._feature_imputation_table_tasks = None
        self._all_as_of_times = None
        self._master_feature_dictionary = None
        self.initialize_factories()
        self.initialize_components()

    def _check_config_version(self, config):
        if 'config_version' in config:
            config_version = config['config_version']
        else:
            logging.warning('config_version key not found in experiment config. Assuming v1, which may not be correct')
            config_version = 'v1'
        if config_version != CONFIG_VERSION:
            raise ValueError(
                '''Experiment config '{}'
                does not match current version '{}'.
                Will not run experiment.'''\
                .format(config_version, CONFIG_VERSION)
            )

    def initialize_factories(self):
        split_config = self.config['temporal_config']

        self.chopper_factory = partial(
            Timechop,
            beginning_of_time=dt_from_str(split_config['beginning_of_time']),
            modeling_start_time=dt_from_str(split_config['modeling_start_time']),
            modeling_end_time=dt_from_str(split_config['modeling_end_time']),
            update_window=split_config['update_window'],
            train_label_windows=split_config['train_label_windows'],
            test_label_windows=split_config['test_label_windows'],
            train_example_frequency=split_config['train_example_frequency'],
            test_example_frequency=split_config['test_example_frequency'],
            train_durations=split_config['train_durations'],
            test_durations=split_config['test_durations'],
        )

        self.state_table_generator_factory = partial(
            StateTableGenerator,
            experiment_hash=self.experiment_hash,
            dense_state_table=self.config.get('state_config', {})
            .get('table_name', None),
            events_table=self.config['events_table']
        )

        self.label_generator_factory = partial(
            BinaryLabelGenerator,
            events_table=self.config['events_table'],
        )

        self.feature_dictionary_creator_factory = partial(
            FeatureDictionaryCreator,
            features_schema_name=self.features_schema_name,
        )

        self.feature_generator_factory = partial(
            FeatureGenerator,
            features_schema_name=self.features_schema_name,
            replace=self.replace,
            beginning_of_time=split_config['beginning_of_time']
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
            beginning_of_time=dt_from_str(split_config['beginning_of_time']),
            label_names=['outcome'],
            label_types=['binary'],
            db_config={
                'features_schema_name': self.features_schema_name,
                'labels_schema_name': 'public',
                'labels_table_name': self.labels_table_name,
                # TODO: have planner/builder take state table later on, so we
                # can grab it from the StateTableGenerator instead of
                # duplicating it here
                'sparse_state_table_name': 'tmp_sparse_states_{}'.format(self.experiment_hash),
            },
            matrix_directory=self.matrices_directory,
            states=self.config.get('state_config', {}).get('state_filters', []),
            user_metadata=self.config.get('user_metadata', {}),
            replace=self.replace
        )

        self.trainer_factory = partial(
            ModelTrainer,
            project_path=self.project_path,
            experiment_hash=self.experiment_hash,
            model_storage_engine=self.model_storage_engine,
            model_group_keys=self.config['model_group_keys'],
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
        self.feature_dictionary_creator = self.feature_dictionary_creator_factory(db_engine=self.db_engine)
        self.feature_group_creator = self.feature_group_creator_factory()
        self.feature_group_mixer = self.feature_group_mixer_factory()
        self.planner = self.planner_factory(engine=self.db_engine)
        self.trainer = self.trainer_factory(db_engine=self.db_engine)
        self.predictor = self.predictor_factory(db_engine=self.db_engine)
        self.individual_importance_calculator = self.indiv_importance_factory(db_engine=self.db_engine)
        self.evaluator = self.evaluator_factory(db_engine=self.db_engine)

    @property
    def split_definitions(self):
        """Temporal splits based on the experiment's configuration

        Returns: (dict) temporal splits

        Example:
        ```
        {
            'beginning_of_time': {datetime},
            'modeling_start_time': {datetime},
            'modeling_end_time': {datetime},
            'train_matrix': {
                'matrix_start_time': {datetime},
                'matrix_end_time': {datetime},
                'as_of_times': [list of {datetime}s]
            },
            'test_matrices': [list of matrix defs similar to train_matrix]
        }
        ```
        """
        if not self._split_definitions:
            self._split_definitions = self.chopper.chop_time()
            logging.info(
                'Computed and saved split definitions: %s',
                self._split_definitions
            )
        return self._split_definitions

    @property
    def all_as_of_times(self):
        """All 'as of times' in experiment config

        Used for label and feature generation.

        Returns: (list) of datetimes
        """
        if not self._all_as_of_times:
            all_as_of_times = []
            for split in self.split_definitions:
                all_as_of_times.extend(split['train_matrix']['as_of_times'])
                logging.info(
                    'Adding as_of_times from train matrix: %s',
                    split['train_matrix']['as_of_times']
                )
                for test_matrix in split['test_matrices']:
                    logging.info(
                        'Adding as_of_times from test matrix: %s',
                        test_matrix['as_of_times']
                    )
                    all_as_of_times.extend(test_matrix['as_of_times'])

            logging.info(
                'Computed %s total as_of_times for label and feature generation',
                len(all_as_of_times)
            )
            self._all_as_of_times = list(set(all_as_of_times))
            logging.info(
                'Computed %s distinct as_of_times for label and feature generation',
                len(self._all_as_of_times)
            )
        return self._all_as_of_times

    @property
    def collate_aggregations(self):
        """collate Aggregation objects used by this experiment.
        
        Returns: (list) of collate.Aggregation objects
        """
        if not self._collate_aggregations:
            logging.info('Creating collate aggregations')
            self._collate_aggregations = self.feature_generator.aggregations(
                feature_aggregation_config=self.config['feature_aggregations'],
                feature_dates=self.all_as_of_times,
                state_table=self.state_table_generator.sparse_table_name,
            )
        return self._collate_aggregations


    @property
    def feature_aggregation_table_tasks(self):
        """All feature table query tasks specified by this Experiment

        Returns: (dict) keys are group table names, values are themselves dicts, each with keys for different stages of table creation (prepare, inserts, finalize) and with values being lists of SQL commands
        """
        if not self._feature_aggregation_table_tasks:
            logging.info(
                'Calculating feature tasks for %s as_of_times',
                len(self.all_as_of_times)
            )
            self._feature_aggregation_table_tasks = self.feature_generator\
                .generate_all_table_tasks(self.collate_aggregations, task_type='aggregation')
        return self._feature_aggregation_table_tasks

    @property
    def feature_imputation_table_tasks(self):
        """All feature imputation query tasks specified by this Experiment

        Returns: (dict) keys are group table names, values are themselves dicts,
            each with keys for different stages of table creation (prepare, inserts, finalize)
            and with values being lists of SQL commands
        """
        if not self._feature_imputation_table_tasks:
            logging.info(
                'Calculating feature tasks for %s as_of_times',
                len(self.all_as_of_times)
            )
            self._feature_imputation_table_tasks = self.feature_generator\
                .generate_all_table_tasks(self.collate_aggregations, task_type='imputation')
        return self._feature_imputation_table_tasks

    @property
    def master_feature_dictionary(self):
        """All possible features found in the database. Not all features will necessarily end up in matrices

        Returns: (list) of dicts, keys being feature table names and values being lists of feature names
        """
        if not self._master_feature_dictionary:
            self._master_feature_dictionary = self.feature_dictionary_creator\
                .feature_dictionary(
                    feature_table_names=self.feature_imputation_table_tasks.keys(),
                    index_column_lookup=self.feature_generator.index_column_lookup(self.collate_aggregations)
                )
            logging.info(
                'Computed master feature dictionary: %s',
                self._master_feature_dictionary
            )

        return self._master_feature_dictionary

    @property
    def feature_dicts(self):
        """Feature dictionaries, representing the feature tables and columns configured in this experiment after computing feature groups.

        Returns: (list) of dicts, keys being feature table names and values being lists of feature names
        """

        return self.feature_group_mixer.generate(
            self.feature_group_creator.subsets(self.master_feature_dictionary)
        )

    @property
    def matrix_build_tasks(self):
        """Tasks for all matrices that need to be built as a part of this Experiment.

        Each task contains arguments understood by Architect.build_matrix

        Returns: (list) of dicts
        """
        if not self._matrix_build_tasks:
            updated_split_definitions, matrix_build_tasks =\
                self.planner.generate_plans(
                    self.split_definitions,
                    self.feature_dicts
                )
            self._full_matrix_definitions = updated_split_definitions
            self._matrix_build_tasks = matrix_build_tasks
        return self._matrix_build_tasks

    @property
    def full_matrix_definitions(self):
        """Full matrix definitions

        Returns: (list) temporal and feature information for each matrix
        """
        if not self._full_matrix_definitions:
            updated_split_definitions, matrix_build_tasks =\
                self.planner.generate_plans(
                    self.split_definitions,
                    self.feature_dicts
                )
            self._full_matrix_definitions = updated_split_definitions
            self._matrix_build_tasks = matrix_build_tasks
        return self._full_matrix_definitions

    @property
    def all_label_windows(self):
        """All train and test label windows
        
        Returns: (list) label windows, in string form as they appeared in the experiment config
        """
        return list(set(
            self.config['temporal_config']['train_label_windows'] + \
            self.config['temporal_config']['test_label_windows']
        ))

    def generate_labels(self):
        """Generate labels based on experiment configuration

        Results are stored in the database, not returned
        """
        self.label_generator.generate_all_labels(
            self.labels_table_name,
            self.all_as_of_times,
            self.all_label_windows
        )

    def generate_sparse_states(self):
        self.state_table_generator.generate_sparse_table(
            as_of_dates=self.all_as_of_times
        )

    def update_split_definitions(self, new_split_definitions):
        """Update split definitions

        Args: (dict) split definitions (should have matrix uuids)
        """
        self._split_definitions = new_split_definitions

    def log_split(self, split_num, split):
        logging.info(
            'Starting train/test for %s out of %s: train range: %s to %s',
            split_num+1,
            len(self.full_matrix_definitions),
            split['train_matrix']['matrix_start_time'],
            split['train_matrix']['matrix_end_time'],
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

    def run(self):
        try:
            self.build_matrices()
        finally:
            logging.info('Matrix build done or errored, cleaning up state table')
            self.state_table_generator.clean_up()
        self.catwalk()
