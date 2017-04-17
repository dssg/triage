from triage.db import ensure_db
from triage.label_generators import BinaryLabelGenerator
from triage.features import \
    FeatureGenerator,\
    FeatureDictionaryCreator,\
    FeatureGroupCreator,\
    FeatureGroupMixer
from triage.model_trainers import ModelTrainer
from triage.predictors import Predictor
from triage.scoring import ModelScorer
from triage.utils import save_experiment_and_get_hash
from timechop.timechop import Inspections
from timechop.architect import Architect
import os
from datetime import datetime
from abc import ABCMeta, abstractmethod
from functools import partial
import logging


def dt_from_str(dt_str):
    return datetime.strptime(dt_str, '%Y-%m-%d')


class PipelineBase(object):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        config,
        db_engine,
        model_storage_class=None,
        project_path=None,
        replace=True
    ):
        self.config = config
        self.db_engine = db_engine
        if model_storage_class:
            self.model_storage_engine =\
                model_storage_class(project_path=project_path)
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
        self._feature_table_tasks = None
        self._all_as_of_times = None
        self.initialize_factories()
        self.initialize_components()

    def initialize_factories(self):
        split_config = self.config['temporal_config']

        self.chopper_factory = partial(
            Inspections,
            beginning_of_time=dt_from_str(split_config['beginning_of_time']),
            modeling_start_time=dt_from_str(split_config['modeling_start_time']),
            modeling_end_time=dt_from_str(split_config['modeling_end_time']),
            update_window=split_config['update_window'],
            look_back_durations=split_config['look_back_durations'],
            test_durations=split_config['test_durations'],
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
            replace=self.replace
        )

        self.feature_group_creator_factory = partial(
            FeatureGroupCreator,
            self.config.get('feature_group_definition', {'all': [True]})
        )

        self.feature_group_mixer_factory = partial(
            FeatureGroupMixer,
            self.config.get('feature_group_strategies', ['all'])
        )

        self.architect_factory = partial(
            Architect,
            beginning_of_time=dt_from_str(split_config['beginning_of_time']),
            label_names=['outcome'],
            label_types=['binary'],
            db_config={
                'features_schema_name': self.features_schema_name,
                'labels_schema_name': 'public',
                'labels_table_name': self.labels_table_name,
            },
            matrix_directory=self.matrices_directory,
            user_metadata={},
            replace=self.replace
        )

        self.trainer_factory = partial(
            ModelTrainer,
            project_path=self.project_path,
            experiment_hash=self.experiment_hash,
            model_storage_engine=self.model_storage_engine,
            replace=self.replace
        )

        self.predictor_factory = partial(
            Predictor,
            model_storage_engine=self.model_storage_engine,
            project_path=self.project_path,
            replace=self.replace
        )

        self.model_scorer_factory = partial(
            ModelScorer,
            metric_groups=self.config['scoring'],
        )

    def initialize_components(self):
        self.chopper = self.chopper_factory()
        self.label_generator = self.label_generator_factory(db_engine=self.db_engine)
        self.feature_generator = self.feature_generator_factory(db_engine=self.db_engine)
        self.feature_dictionary_creator = self.feature_dictionary_creator_factory(db_engine=self.db_engine)
        self.feature_group_creator = self.feature_group_creator_factory()
        self.feature_group_mixer = self.feature_group_mixer_factory()
        self.architect = self.architect_factory(engine=self.db_engine)
        self.trainer = self.trainer_factory(db_engine=self.db_engine)
        self.predictor = self.predictor_factory(db_engine=self.db_engine)
        self.model_scorer = self.model_scorer_factory(db_engine=self.db_engine)

    def chop_time(self):
        # TODO: timechop should take care of this. remove when
        # https://github.com/dssg/timechop/issues/26 is resolved
        prediction_window = self.config['temporal_config']['prediction_window']
        split_definitions = self.chopper.chop_time()
        for split_definition in split_definitions:
            split_definition['train_matrix']['prediction_window'] = prediction_window
            for test_matrix in split_definition['test_matrices']:
                test_matrix['prediction_window'] = prediction_window
        return split_definitions

    @property
    def split_definitions(self):
        if not self._split_definitions:
            self._split_definitions = self.chop_time()
        return self._split_definitions

    @property
    def all_as_of_times(self):
        if not self._all_as_of_times:
            all_as_of_times = []
            for split in self.split_definitions:
                all_as_of_times.extend(split['train_matrix']['as_of_times'])
                for test_matrix in split['test_matrices']:
                    all_as_of_times.extend(test_matrix['as_of_times'])

            logging.info(
                'Found %s distinct as_of_times for label and feature generation',
                len(all_as_of_times)
            )
            self._all_as_of_times = list(set(all_as_of_times))
        return self._all_as_of_times

    @property
    def feature_table_tasks(self):
        if not self._feature_table_tasks:
            logging.info(
                'Calculating feature tasks for %s as_of_times',
                len(self.all_as_of_times)
            )
            self._feature_table_tasks = self.feature_generator.generate_all_table_tasks(
                feature_aggregation_config=self.config['feature_aggregations'],
                feature_dates=self.all_as_of_times,
            )
        return self._feature_table_tasks

    @property
    def feature_dicts(self):
        master_feature_dict = self.feature_dictionary_creator\
            .feature_dictionary(self.feature_table_tasks.keys())

        return self.feature_group_mixer.generate(
            self.feature_group_creator.subsets(master_feature_dict)
        )

    @property
    def matrix_build_tasks(self):
        if not self._matrix_build_tasks:
            updated_split_definitions, self._matrix_build_tasks =\
                self.architect.generate_plans(
                    self.split_definitions,
                    self.feature_dicts
                )
            self.update_split_definitions(updated_split_definitions)
        return self._matrix_build_tasks

    def generate_labels(self):
        self.label_generator.generate_all_labels(
            self.labels_table_name,
            self.all_as_of_times,
            self.config['temporal_config']['prediction_window']
        )

    def update_split_definitions(self, new_split_definitions):
        self._split_definitions = new_split_definitions

    @abstractmethod
    def build_matrices(self):
        pass

    @abstractmethod
    def catwalk(self):
        pass

    def run(self):
        self.build_matrices()
        self.catwalk()
