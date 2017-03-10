import boto3
import pandas
import pickle
import testing.postgresql
import datetime

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
            metadata = {
                'start_time': datetime.date(2012, 12, 20),
                'end_time': datetime.date(2016, 12, 20),
                'label_name': 'label',
                'prediction_window': '1y',
                'feature_names': ['ft1', 'ft2']
            }
            project_path = 'econ-dev/inspections'
            model_storage_engine = S3ModelStorageEngine(s3_conn, project_path)
            trainer = ModelTrainer(
                project_path=project_path,
                model_storage_engine=model_storage_engine,
                matrix_store=InMemoryMatrixStore(matrix, metadata),
                db_engine=engine,
            )
            model_ids = trainer.train_models(grid_config=grid_config, misc_db_parameters=dict())

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

            # 2. that the model groups are distinct
            records = [
                row for row in
                engine.execute('select distinct model_group_id from results.models')
            ]
            assert len(records) == 4

            # 3. that all four models are cached
            model_pickles = [
                pickle.loads(cache_key.get()['Body'].read())
                for cache_key in cache_keys
            ]
            assert len(model_pickles) == 4
            assert len([x for x in model_pickles if x is not None]) == 4

            # 4. that their results can have predictions made on it
            test_matrix = pandas.DataFrame.from_dict({
                'entity_id': [3, 4],
                'feature_one': [4, 4],
                'feature_two': [6, 5],
            })
            for model_pickle in model_pickles:
                predictions = model_pickle.predict(test_matrix)
                assert len(predictions) == 2

            # 5. when run again, same models are returned
            new_model_ids = trainer.train_models(grid_config=grid_config, misc_db_parameters=dict())
            assert len([
                row for row in
                engine.execute('select model_hash from results.models')
            ]) == 4
            assert model_ids == new_model_ids

            # 6. if metadata is deleted but the cache is still there,
            # retrains that one and replaces the feature importance records
            engine.execute('delete from results.feature_importances where model_id = 3')
            engine.execute('delete from results.models where model_id = 3')
            new_model_ids = trainer.train_models(grid_config=grid_config, misc_db_parameters=dict())
            expected_model_ids = [1, 2, 4, 5]
            assert expected_model_ids == sorted(new_model_ids)
            assert [
                row['model_id'] for row in
                engine.execute('select model_id from results.models order by 1 asc')
            ] == expected_model_ids

            records = [
                row for row in
                engine.execute('select * from results.feature_importances')
            ]
            assert len(records) == 4 * 3  # maybe exclude entity_id?

            # 7. if the cache is missing but the metadata is still there, reuse the metadata
            for row in engine.execute('select model_hash from results.models'):
                model_storage_engine.get_store(row[0]).delete()
            expected_model_ids = [1, 2, 4, 5]
            new_model_ids = trainer.train_models(grid_config=grid_config, misc_db_parameters=dict())
            assert expected_model_ids == sorted(new_model_ids)

            # 8. that the generator interface works the same way
            new_model_ids = trainer.generate_trained_models(
                grid_config=grid_config,
                misc_db_parameters=dict()
            )
            assert expected_model_ids == \
                sorted([model_id for model_id in new_model_ids])
