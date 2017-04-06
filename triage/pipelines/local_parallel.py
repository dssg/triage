from triage.storage import MettaCSVMatrixStore
from sqlalchemy import create_engine
from triage.pipelines import PipelineBase
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os


class LocalParallelPipeline(PipelineBase):
    def __init__(self, n_processes=1, *args, **kwargs):
        super(LocalParallelPipeline, self).__init__(*args, **kwargs)
        self.n_processes = n_processes

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
        logging.info(
            'Generating features for %s as_of_times',
            len(all_as_of_times)
        )
        feature_tables = self.feature_generator.generate(
            feature_aggregations=self.config['feature_aggregations'],
            feature_dates=all_as_of_times,
        )

        master_feature_dict = self.feature_dictionary_creator\
            .feature_dictionary(feature_tables)

        feature_dicts = self.feature_group_mixer.generate(
            self.feature_group_creator.subsets(master_feature_dict)
        )

        # 4. create training and test sets
        logging.info('Creating matrices')
        logging.debug('---------------------')
        logging.debug('---------MATRIX GENERATION------------')
        logging.debug('---------------------')

        updated_split_definitions, build_tasks = self.architect.generate_plans(
            split_definitions,
            feature_dicts
        )

        partial_build_matrix = partial(
            build_matrix,
            architect_factory=self.architect_factory,
            db_connection_string=self.db_engine.url
        )
        logging.info(
            'Starting parallel matrix building: %s matrices, %s processes',
            len(build_tasks.keys()),
            self.n_processes
        )
        with ProcessPoolExecutor(max_workers=self.n_processes) as pool:
            num_successes = 0
            num_failures = 0
            for successful in pool.map(
                partial_build_matrix,
                build_tasks.values()
            ):
                if successful:
                    num_successes += 1
                else:
                    num_failures += 1
            logging.info(
                'Done building. successes: %s, failures: %s',
                num_successes,
                num_failures
            )

        for split in updated_split_definitions:
            logging.info('Starting split')
            train_store = MettaCSVMatrixStore(
                matrix_path=os.path.join(
                    self.matrices_directory,
                    '{}.csv'.format(split['train_uuid'])
                ),
                metadata_path=os.path.join(
                    self.matrices_directory,
                    '{}.yaml'.format(split['train_uuid'])
                )
            )
            logging.info('Checking out train matrix')
            if train_store.matrix.empty:
                logging.warning('''Train matrix for split %s was empty,
                no point in training this model. Skipping
                ''', split['train_uuid'])
                continue
            logging.info('Checking out train labels')
            if len(train_store.labels().unique()) == 1:
                logging.warning('''Train Matrix for split %s had only one
                unique value, no point in training this model. Skipping
                ''', split['train_uuid'])
                continue

            logging.info('Training models')
            model_ids = self.trainer.train_models(
                grid_config=self.config['grid_config'],
                misc_db_parameters=dict(
                    test=False,
                    model_comment=self.config.get('model_comment', None),
                ),
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
                    matrix_path=os.path.join(
                        self.matrices_directory,
                        '{}.csv'.format(test_uuid)
                    ),
                    metadata_path=os.path.join(
                        self.matrices_directory,
                        '{}.yaml'.format(test_uuid)
                    )
                )
                if test_store.matrix.empty:
                    logging.warning('''Test matrix for train uuid %s
                    was empty, no point in training this model. Skipping
                    ''', split['train_uuid'])
                    continue
                partial_test_and_score = partial(
                    test_and_score,
                    predictor_factory=self.predictor_factory,
                    model_scorer_factory=self.model_scorer_factory,
                    test_store=test_store,
                    db_connection_string=self.db_engine.url,
                    split_def=split_def,
                    config=self.config
                )
                logging.info(
                    'Starting parallel testing with %s processes',
                    self.n_processes
                )
                with ProcessPoolExecutor(max_workers=self.n_processes) as pool:
                    num_successes = 0
                    num_failures = 0
                    for successful in pool.map(
                        partial_test_and_score,
                        model_ids
                    ):
                        if successful:
                            num_successes += 1
                        else:
                            num_failures += 1
                    logging.info(
                        'Done testing. successes: %s, failures: %s',
                        num_successes,
                        num_failures
                    )
                logging.info('Cleaned up concurrent pool')
            logging.info('Done with test matrix')
        logging.info('Done with split')


def build_matrix(
    build_task,
    architect_factory,
    db_connection_string,
):
    try:
        db_engine = create_engine(db_connection_string)
        architect = architect_factory(engine=db_engine)
        architect.build_matrix(**build_task)
        return True
    except Exception as e:
        logging.error('Child error: %s', e)
        return False


def test_and_score(
    model_id,
    predictor_factory,
    model_scorer_factory,
    test_store,
    db_connection_string,
    split_def,
    config
):
    try:
        db_engine = create_engine(db_connection_string)
        logging.info('Generating predictions for model id %s', model_id)
        predictor = predictor_factory(db_engine=db_engine)
        model_scorer = model_scorer_factory(db_engine=db_engine)

        predictions, predictions_proba = predictor.predict(
            model_id,
            test_store,
            misc_db_parameters=dict()
        )
        logging.info('Generating evaluations for model id %s', model_id)
        model_scorer.score(
            predictions_proba=predictions_proba,
            predictions_binary=predictions,
            labels=test_store.labels(),
            model_id=model_id,
            evaluation_start_time=split_def['matrix_start_time'],
            evaluation_end_time=split_def['matrix_end_time'],
            prediction_frequency=config['temporal_config']['prediction_frequency']
        )
        return True
    except Exception as e:
        logging.error('Child error: %s', e)
        return False
