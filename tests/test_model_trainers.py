import boto3
import pandas
import pickle
import testing.postgresql

from moto import mock_s3
from sqlalchemy import create_engine
from triage.db import ensure_db
from triage.utils import model_cache_key

from triage.model_trainers import ModelTrainer
from triage.storage import S3ModelStorageEngine, InMemoryMatrixStore


def test_model_trainer():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)

        grid_config = {
            'sklearn.linear_model.LogisticRegression': {
                'C': [0.00001, 0.0001],
                'penalty': ['l1', 'l2'],
                'random_state': [2193]
            }
        }

        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')

            # create training set
            matrix = pandas.DataFrame.from_dict({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': ['good', 'bad']
            })
            metadata = {'label_name': 'label'}
            project_path = 'econ-dev/inspections'
            trainer = ModelTrainer(
                project_path=project_path,
                model_storage_engine=S3ModelStorageEngine(s3_conn, project_path),
                matrix_store=InMemoryMatrixStore(matrix, metadata),
                db_engine=engine,
            )
            trainer.train_models(grid_config=grid_config, misc_db_parameters=dict())

            # assert
            # 1. that the models and feature importances table entries are present
            records = [
                row for row in
                engine.execute('select * from results.feature_importances')
            ]
            assert len(records) == 4 * 3  # maybe exclude entity_id?

            records = [
                row for row in
                engine.execute('select model_hash from results.models')
            ]
            assert len(records) == 4

            cache_keys = [
                model_cache_key(project_path, model_row[0], s3_conn)
                for model_row in records
            ]

            # 2. that all four models are cached
            model_pickles = [
                pickle.loads(cache_key.get()['Body'].read())
                for cache_key in cache_keys
            ]
            assert len(model_pickles) == 4
            assert len([x for x in model_pickles if x is not None]) == 4

            # 3. that their results can have predictions made on it
            test_matrix = pandas.DataFrame.from_dict({
                'entity_id': [3, 4],
                'feature_one': [4, 4],
                'feature_two': [6, 5],
            })
            for model_pickle in model_pickles:
                predictions = model_pickle.predict(test_matrix)
                assert len(predictions) == 2
