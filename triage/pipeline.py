from triage.utils import temporal_splits
import triage.entity_feature_date_generators as efd_generators
from triage.training_label_generators import TrainingLabelGenerator
from triage.feature_generators import FeatureGenerator
from triage.model_trainers import ModelTrainer
from triage.model_results_generators import ModelResultsGenerator
import logging
import yaml


class Pipeline(object):
    def __init__(self, config, db_engine):
        self.config = config
        self.db_engine = db_engine

    def temporal_splits(self):
        return temporal_splits(
            self.config['temporal_splits']['start_time'],
            self.config['temporal_splits']['end_time'],
            self.config['temporal_splits']['update_window'],
            self.config['temporal_splits']['prediction_windows']
        )

    def entity_feature_date_generator_cls(self):
        window_strategy = self.config['window_strategy']
        if window_strategy == 'EntireWindow':
            return efd_generators.WholeWindowFeatureDateGenerator
        elif window_strategy == 'TimeOfEvent':
            return efd_generators.TimeOfEventFeatureDateGenerator
        else:
            raise 'No entity feature date generator available for window_strategy {}'.format(window_strategy)

    def run(self):
        # 1. generate temporal splits
        for split in self.temporal_splits():
            # 2. calculate entity feature dates (ie self.entity_feature_dates(splits))
            entity_feature_dates_table = self.entity_feature_date_generator_cls()(
                split=split,
                events_table=self.config['events_table'],
                db_engine=db_engine
            ).generate()
            # 3. create training set
            training_label_table = TrainingLabelGenerator(
                entity_feature_dates_table=entity_feature_dates_table,
                events_table=self.config['events_table'],
                db_engine=db_engine
            ).generate()
            # 4. generate features
            features_table = FeatureGenerator(
                training_label_table=training_label_table,
                data_table=self.config['data_table'],
                feature_aggregations=self.config['feature_aggregations']
            ).generate()
            trained_model_path = 'project/trained_models'
            # 5. train models
            model_uuids = ModelTrainer(
                features_table=features_table,
                model_config=self.config['models'],
                trained_model_path=trained_model_path
            ).train()
            # 6. generate model results
            ModelResultsGenerator(
                trained_model_path=trained_model_path,
                model_uuids=model_uuids
            ).generate()

if __name__ == '__main__':
    with open('example_experiment_config.yaml') as f:
        experiment_config = yaml.load(f)
    pipeline = Pipeline(config=experiment_config)
    pipeline.run()
