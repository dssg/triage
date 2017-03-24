from triage.db import ensure_db
from triage.label_generators import BinaryLabelGenerator
from triage.features import FeatureGenerator, FeatureDictionaryCreator
from triage.model_trainers import ModelTrainer
from triage.predictors import Predictor
from triage.scoring import ModelScorer
from timechop.timechop import Inspections
from timechop.architect import Architect
import os
from datetime import datetime
from abc import ABCMeta, abstractmethod


def dt_from_str(dt_str):
    return datetime.strptime(dt_str, '%Y-%m-%d')


class PipelineBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, db_engine, model_storage_class, project_path):
        self.config = config
        self.db_engine = db_engine
        self.model_storage_engine =\
            model_storage_class(project_path=project_path)
        self.project_path = project_path
        ensure_db(self.db_engine)

        self.labels_table_name = 'labels'
        self.features_schema_name = 'features'
        self.matrices_directory = os.path.join(self.project_path, 'matrices')
        if not os.path.exists(self.matrices_directory):
            os.makedirs(self.matrices_directory)
        self.initialize_components()

    def initialize_components(self):
        split_config = self.config['temporal_config']
        self.chopper = Inspections(
            beginning_of_time=dt_from_str(split_config['beginning_of_time']),
            modeling_start_time=dt_from_str(split_config['modeling_start_time']),
            modeling_end_time=dt_from_str(split_config['modeling_end_time']),
            update_window=split_config['update_window'],
            look_back_durations=split_config['look_back_durations'],
            test_durations=split_config['test_durations'],
        )

        self.label_generator = BinaryLabelGenerator(
            events_table=self.config['events_table'],
            db_engine=self.db_engine
        )

        self.feature_generator = FeatureGenerator(
            features_schema_name=self.features_schema_name,
            db_engine=self.db_engine
        )

        self.feature_dictionary_creator = FeatureDictionaryCreator(
            features_schema_name=self.features_schema_name,
            db_engine=self.db_engine
        )

        self.architect = Architect(
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
            engine=self.db_engine
        )

        self.trainer = ModelTrainer(
            project_path=self.project_path,
            model_storage_engine=self.model_storage_engine,
            db_engine=self.db_engine,
            matrix_store=None
        )

        self.predictor = Predictor(
            project_path=self.project_path,
            model_storage_engine=self.model_storage_engine,
            db_engine=self.db_engine
        )

        self.model_scorer = ModelScorer(
            metric_groups=self.config['scoring'],
            db_engine=self.db_engine
        )

    @abstractmethod
    def run(self):
        pass
