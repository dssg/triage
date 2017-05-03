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
                'beginning_of_time': datetime.date(2012, 12, 20),
                'end_time': datetime.date(2016, 12, 20),
                'label_name': 'label',
                'label_window': '1y',
                'metta-uuid': '1234',
                'feature_names': ['ft1', 'ft2']
            }
            project_path = 'econ-dev/inspections'
            model_storage_engine = S3ModelStorageEngine(s3_conn, project_path)
            trainer = ModelTrainer(
                project_path=project_path,
                experiment_hash=None,
                model_storage_engine=model_storage_engine,
                db_engine=engine,
                model_group_keys=['label_name', 'label_window']
            )
            matrix_store = InMemoryMatrixStore(matrix, metadata)
            model_ids = trainer.train_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store
            )

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
            new_model_ids = trainer.train_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store
            )
            assert len([
                row for row in
                engine.execute('select model_hash from results.models')
            ]) == 4
            assert model_ids == new_model_ids

            # 6. if metadata is deleted but the cache is still there,
            # retrains that one and replaces the feature importance records
            engine.execute('delete from results.feature_importances where model_id = 3')
            engine.execute('delete from results.models where model_id = 3')
            new_model_ids = trainer.train_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store
            )
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
            new_model_ids = trainer.train_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store
            )
            assert expected_model_ids == sorted(new_model_ids)

            # 8. that the generator interface works the same way
            new_model_ids = trainer.generate_trained_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store
            )
            assert expected_model_ids == \
                sorted([model_id for model_id in new_model_ids])

def test_n_jobs_not_new_model():
    grid_config = {
        'sklearn.ensemble.AdaBoostClassifier': {
            'n_estimators': [10, 100, 1000]
        },
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [10, 100],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [5, 10, 15, 20],
            'criterion': ['gini', 'entropy'],
            'n_jobs': [12, 24],
        }
    }

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')
            trainer = ModelTrainer(
                project_path='econ-dev/inspections',
                experiment_hash=None,
                model_storage_engine=S3ModelStorageEngine(s3_conn, 'econ-dev/inspections'),
                db_engine=engine,
                model_group_keys=['label_name', 'label_window']
            )

            matrix = pandas.DataFrame.from_dict({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': ['good', 'bad']
            })
            train_tasks = trainer.generate_train_tasks(
                grid_config,
                dict(),
                InMemoryMatrixStore(matrix, {
                    'label_window': '1d',
                    'end_time': datetime.datetime.now(),
                    'beginning_of_time': datetime.date(2012, 12, 20),
                    'label_name': 'label',
                    'metta-uuid': '1234',
                    'feature_names': ['ft1', 'ft2']
                })
            )
            assert len(train_tasks) == 35 # 32+3, would be (32*2)+3 if we didn't remove
            assert len([
                task for task in train_tasks
                if 'n_jobs' in task['parameters']
            ]) == 32

            for train_task in train_tasks:
                trainer.process_train_task(**train_task)

            for row in engine.execute(
                'select model_parameters from results.model_groups'
            ):
                assert 'n_jobs' not in row[0]
