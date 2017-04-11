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


def dt_from_str(dt_str):
    return datetime.strptime(dt_str, '%Y-%m-%d')


class PipelineBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, db_engine, model_storage_class=None, project_path=None):
        self.config = config
        self.db_engine = db_engine
        if model_storage_class:
            self.model_storage_engine =\
                model_storage_class(project_path=project_path)
        self.project_path = project_path
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
        )

        self.trainer_factory = partial(
            ModelTrainer,
            project_path=self.project_path,
            experiment_hash=self.experiment_hash,
            model_storage_engine=self.model_storage_engine,
        )

        self.predictor_factory = partial(
            Predictor,
            model_storage_engine=self.model_storage_engine,
            project_path=self.project_path,
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

    @abstractmethod
    def run(self):
        pass
