from triage.db import ensure_db
from triage.label_generators import BinaryLabelGenerator
from triage.feature_generators import FeatureGenerator
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

    def run(self):
        # 1. generate temporal splits
        split_config = self.config['temporal_config']
        inspections_chopper = Inspections(
            beginning_of_time=dt_from_str(split_config['beginning_of_time']),
            modeling_start_time=dt_from_str(split_config['modeling_start_time']),
            modeling_end_time=dt_from_str(split_config['modeling_end_time']),
            update_window=split_config['update_window'],
            look_back_durations=split_config['look_back_durations'],
        )
        matrix_definitions = inspections_chopper.chop_time()

        # 2. create labels
        logging.debug('---------------------')
        logging.debug('---------LABEL GENERATION------------')
        logging.debug('---------------------')
        label_generator = BinaryLabelGenerator(
            events_table=self.config['events_table'],
            db_engine=self.db_engine
        )

        labels_table = 'labels'

        all_as_of_times = []
        for split in matrix_definitions:
            all_as_of_times.extend(split['train_matrix']['as_of_times'])
            for test_matrix in split['test_matrices']:
                all_as_of_times.extend(test_matrix['as_of_times'])
        all_as_of_times = list(set(all_as_of_times))

        logging.warning(
            'Found %s distinct as_of_times for label and feature generation',
            len(all_as_of_times)
        )
        label_generator.generate_all_labels(
            labels_table,
            all_as_of_times,
            split_config['prediction_window']
        )

        # 3. generate features
        logging.info('Generating features for %s', all_as_of_times)
        rollup_feature_tables = FeatureGenerator(
            db_engine=self.db_engine
        ).generate(
            feature_aggregations=self.config['feature_aggregations'],
            feature_dates=all_as_of_times,
        )

        # 4. create training and test sets
        logging.info('Creating matrices')
        logging.debug('---------------------')
        logging.debug('---------MATRIX GENERATION------------')
        logging.debug('---------------------')

        rollup_tables = [
            tbl.split('.')[1].replace('"', "'")
            for tbl in rollup_feature_tables
        ]
        architect = Architect(
            batch_id=str(uuid.uuid4()),
            batch_timestamp=datetime.now(),
            beginning_of_time=dt_from_str(split_config['beginning_of_time']),
            label_names=['outcome'],
            label_types=['binary'],
            db_config={
                'features_schema_name': 'features',
                'rollup_feature_tables': rollup_tables,
                'labels_schema_name': 'public',
                'labels_table_name': labels_table,
            },
            user_metadata={},
            engine=self.db_engine
        )
        updated_matrix_definitions = architect.chop_data(matrix_definitions)

        for split in updated_matrix_definitions:
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

            trainer = ModelTrainer(
                project_path=self.project_path,
                model_storage_engine=self.model_storage_engine,
                matrix_store=train_store,
                db_engine=self.db_engine
            )

            predictor = Predictor(
                project_path=self.project_path,
                model_storage_engine=self.model_storage_engine,
                db_engine=self.db_engine
            )
            model_scorer = ModelScorer(
                metric_groups=self.config['scoring'],
                db_engine=self.db_engine
            )

            logging.info('Training models')
            model_ids = trainer.train_models(
                grid_config=self.config['grid_config'],
                misc_db_parameters=dict(test=False)
            )
            logging.info('Done training models')

            for matrix_def, test_uuid in zip(
                split['test_matrices'],
                split['test_uuids']
            ):
                as_of_times = sorted(matrix_def['as_of_times'])
                min_time = as_of_times[0]
                max_time = as_of_times[-1]
                logging.info(
                    'Testing and scoring as_of_times min: %s max: %s num: %s',
                    min_time,
                    max_time,
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
                    predictions, predictions_proba = predictor.predict(
                        model_id,
                        test_store,
                        misc_db_parameters=dict()
                    )

                    if len(as_of_times) > 1:
                        logging.warning('''
                            Scoring for multiple as_of_times not implemented.
                            Predictions can be found in results.predictions
                        ''')
                        continue
                    else:
                        model_scorer.score(
                            predictions_proba,
                            predictions,
                            test_store.labels(),
                            model_id,
                            as_of_times[0]
                        )
