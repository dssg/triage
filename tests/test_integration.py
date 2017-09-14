from catwalk.model_trainers import ModelTrainer
from catwalk.predictors import Predictor
from catwalk.evaluation import ModelEvaluator
from catwalk.utils import save_experiment_and_get_hash

import boto3
import testing.postgresql

from moto import mock_s3
from sqlalchemy import create_engine
from catwalk.db import ensure_db

from catwalk.storage import S3ModelStorageEngine, InMemoryMatrixStore
import datetime
import pandas


def test_integration():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)

        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')
            project_path = 'econ-dev/inspections'

            # create train and test matrices
            train_matrix = pandas.DataFrame.from_dict({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': [7, 8]
            }).set_index('entity_id')
            train_metadata = {
                'beginning_of_time': datetime.date(2012, 12, 20),
                'end_time': datetime.date(2016, 12, 20),
                'label_name': 'label',
                'label_window': '1y',
                'feature_names': ['ft1', 'ft2'],
                'metta-uuid': '1234',
                'indices': ['entity_id'],
            }

            train_store = InMemoryMatrixStore(train_matrix, train_metadata)

            as_of_dates = [
                datetime.date(2016, 12, 21),
                datetime.date(2017, 1, 21)
            ]

            test_stores = [
                InMemoryMatrixStore(
                    pandas.DataFrame.from_dict({
                        'entity_id': [3],
                        'feature_one': [8],
                        'feature_two': [5],
                        'label': [5]
                    }),
                    {
                        'label_name': 'label',
                        'label_window': '1y',
                        'end_time': as_of_date,
                        'metta-uuid': '1234',
                        'indices': ['entity_id'],
                    }
                )
                for as_of_date in as_of_dates
            ]

            model_storage_engine = S3ModelStorageEngine(s3_conn, project_path)

            experiment_hash = save_experiment_and_get_hash({}, db_engine)
            # instantiate pipeline objects
            trainer = ModelTrainer(
                project_path=project_path,
                experiment_hash=experiment_hash,
                model_storage_engine=model_storage_engine,
                db_engine=db_engine,
                model_group_keys=['label_name', 'label_window']
            )
            predictor = Predictor(
                project_path,
                model_storage_engine,
                db_engine
            )
            model_evaluator = ModelEvaluator(
                [{'metrics': ['precision@'], 'thresholds': {'top_n': [5]}}],
                db_engine
            )

            # run the pipeline
            grid_config = {
                'sklearn.linear_model.LogisticRegression': {
                    'C': [0.00001, 0.0001],
                    'penalty': ['l1', 'l2'],
                    'random_state': [2193]
                }
            }
            model_ids = trainer.train_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=train_store
            )

            for model_id in model_ids:
                for as_of_date, test_store in zip(as_of_dates, test_stores):
                    predictions_proba = predictor.predict(
                        model_id,
                        test_store,
                        misc_db_parameters=dict(),
                        train_matrix_columns=['feature_one', 'feature_two']
                    )

                    model_evaluator.evaluate(
                        predictions_proba,
                        test_store.labels(),
                        model_id,
                        as_of_date,
                        as_of_date,
                        '6month'
                    )

            # assert
            # 1. that the predictions table entries are present and
            # can be linked to the original models
            records = [
                row for row in
                db_engine.execute('''select entity_id, model_id, as_of_date
                from results.predictions
                join results.models using (model_id)
                order by 3, 2''')
            ]
            assert records == [
                (3, 1, datetime.datetime(2016, 12, 21)),
                (3, 2, datetime.datetime(2016, 12, 21)),
                (3, 3, datetime.datetime(2016, 12, 21)),
                (3, 4, datetime.datetime(2016, 12, 21)),
                (3, 1, datetime.datetime(2017, 1, 21)),
                (3, 2, datetime.datetime(2017, 1, 21)),
                (3, 3, datetime.datetime(2017, 1, 21)),
                (3, 4, datetime.datetime(2017, 1, 21)),
            ]

            # that evaluations are there
            records = [
                row for row in
                db_engine.execute('''
                    select model_id, evaluation_start_time, metric, parameter
                    from results.evaluations order by 2, 1''')
            ]
            assert records == [
                (1, datetime.datetime(2016, 12, 21), 'precision@', '5_abs'),
                (2, datetime.datetime(2016, 12, 21), 'precision@', '5_abs'),
                (3, datetime.datetime(2016, 12, 21), 'precision@', '5_abs'),
                (4, datetime.datetime(2016, 12, 21), 'precision@', '5_abs'),
                (1, datetime.datetime(2017, 1, 21), 'precision@', '5_abs'),
                (2, datetime.datetime(2017, 1, 21), 'precision@', '5_abs'),
                (3, datetime.datetime(2017, 1, 21), 'precision@', '5_abs'),
                (4, datetime.datetime(2017, 1, 21), 'precision@', '5_abs'),
            ]
