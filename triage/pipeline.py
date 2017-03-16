from triage.db import ensure_db
from triage.label_generators import BinaryLabelGenerator
from triage.features import FeatureGenerator, FeatureDictionaryCreator
from triage.model_trainers import ModelTrainer
from triage.predictors import Predictor
from triage.scoring import ModelScorer
from triage.storage import MettaCSVMatrixStore
from timechop.timechop import Inspections
from timechop.architect import Architect
import logging
from datetime import datetime
import uuid


def dt_from_str(dt_str):
    return datetime.strptime(dt_str, '%Y-%m-%d')


class Pipeline(object):
    def __init__(self, config, db_engine, model_storage_class, project_path):
        self.config = config
        self.db_engine = db_engine
        self.model_storage_engine =\
            model_storage_class(project_path=project_path)
        self.project_path = project_path
        ensure_db(self.db_engine)

        self.labels_table_name = 'labels'
        self.features_schema_name = 'features'
        self.initialize_components()

    def initialize_components(self):
        split_config = self.config['temporal_config']
        self.chopper = Inspections(
            beginning_of_time=dt_from_str(split_config['beginning_of_time']),
            modeling_start_time=dt_from_str(split_config['modeling_start_time']),
            modeling_end_time=dt_from_str(split_config['modeling_end_time']),
            update_window=split_config['update_window'],
            look_back_durations=split_config['look_back_durations'],
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

    def run(self):
        # 1. generate temporal splits
        split_definitions = self.chopper.chop_time()

        # 2. create labels
        logging.debug('---------------------')
        logging.debug('---------LABEL GENERATION------------')
        logging.debug('---------------------')

        all_as_of_times = []
        for split in split_definitions:
            all_as_of_times.extend(split['train_matrix']['as_of_times'])
            for test_matrix in split['test_matrices']:
                all_as_of_times.extend(test_matrix['as_of_times'])
        all_as_of_times = list(set(all_as_of_times))

        logging.info(
            'Found %s distinct as_of_times for label and feature generation',
            len(all_as_of_times)
        )
        self.label_generator.generate_all_labels(
            self.labels_table_name,
            all_as_of_times,
            self.config['temporal_config']['prediction_window']
        )

        # 3. generate features
        logging.info('Generating features for %s', all_as_of_times)
        tables_to_exclude = self.feature_generator.generate(
            feature_aggregations=self.config['feature_aggregations'],
            feature_dates=all_as_of_times,
        )

        # remove the schema and quotes from the table names
        tables_to_exclude = [
            tbl.split('.')[1].replace('"', "'")
            for tbl in tables_to_exclude
        ]

        feature_dict = self.feature_dictionary_creator.feature_dictionary(
            tables_to_exclude + [self.labels_table_name, 'tmp_entity_ids']
        )

        # 4. create training and test sets
        logging.info('Creating matrices')
        logging.debug('---------------------')
        logging.debug('---------MATRIX GENERATION------------')
        logging.debug('---------------------')

        updated_split_definitions = self.architect.chop_data(
            split_definitions,
            feature_dict
        )

        for split in updated_split_definitions:
            train_store = MettaCSVMatrixStore(
                matrix_path='{}.csv'.format(split['train_uuid']),
                metadata_path='{}.yaml'.format(split['train_uuid'])
            )
            if train_store.matrix.empty:
                logging.warning('''Train matrix for split %s was empty,
                no point in training this model. Skipping
                ''', split['train_uuid'])
                continue
            if len(train_store.labels().unique()) == 1:
                logging.warning('''Train Matrix for split %s had only one
                unique value, no point in training this model. Skipping
                ''', split['train_uuid'])
                continue

            logging.info('Training models')
            model_ids = self.trainer.train_models(
                grid_config=self.config['grid_config'],
                misc_db_parameters=dict(test=False),
                matrix_store=train_store
            )
            logging.info('Done training models')

            for split_def, test_uuid in zip(
                split['test_matrices'],
                split['test_uuids']
            ):
                as_of_times = split_def['as_of_times']
                logging.info(
                    'Testing and scoring as_of_times min: %s max: %s num: %s',
                    min(as_of_times),
                    max(as_of_times),
                    len(as_of_times)
                )
                test_store = MettaCSVMatrixStore(
                    matrix_path='{}.csv'.format(test_uuid),
                    metadata_path='{}.yaml'.format(test_uuid)
                )
                if test_store.matrix.empty:
                    logging.warning('''Test matrix for train uuid %s
                    was empty, no point in training this model. Skipping
                    ''', split['train_uuid'])
                    continue
                for model_id in model_ids:
                    logging.info('Testing model id %s', model_id)
                    predictions, predictions_proba = self.predictor.predict(
                        model_id,
                        test_store,
                        misc_db_parameters=dict()
                    )

                    self.model_scorer.score(
                        predictions_proba=predictions_proba,
                        predictions_binary=predictions,
                        labels=test_store.labels(),
                        model_id=model_id,
                        evaluation_start_time=split_def['matrix_start_time'],
                        evaluation_end_time=split_def['matrix_end_time'],
                        prediction_frequency=self.config['temporal_config']['prediction_frequency']
                    )
