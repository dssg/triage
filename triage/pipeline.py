from triage.utils import temporal_splits
from triage.label_generators import LabelGenerator
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

    def run(self):
        # 1. generate temporal splits
        for split in self.temporal_splits():

            # 2. create labels
            labels_table = LabelGenerator(
                events_table=self.config['events_table'],
                start_date=split['train_start'],
                end_date=split['train_end'],
                split=split,
                db_engine=self.db_engine
            ).generate()

            # 3. generate features
            features_table = FeatureGenerator(
                db_engine=self.db_engine
            ).generate(
                feature_aggregations=self.config['feature_aggregations'],
                feature_dates=split['feature_dates'],
            )

            # 4. create training and test sets
            # timechop!

            # 5. train models
            #trained_model_path = 'project/trained_models'
            #model_ids = ModelTrainer(
                #training_set_path=training_set_path,
                #test_set_path=test_set_path,
                #model_config=self.config['models'],
                #trained_model_path=trained_model_path
            #).train()

            # 6. generate model results
            #ModelResultsGenerator(
                #trained_model_path=trained_model_path,
                #model_ids=model_ids
            #).generate()

if __name__ == '__main__':
    with open('example_experiment_config.yaml') as f:
        experiment_config = yaml.load(f)
    pipeline = Pipeline(config=experiment_config)
    pipeline.run()
