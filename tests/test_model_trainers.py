import boto3
import pandas
import pickle
import tempfile
import testing.postgresql
import yaml

from contextlib import contextmanager
from moto import mock_s3
from sqlalchemy import create_engine
from triage.db import ensure_db
from triage.utils import model_cache_key

from triage.model_trainers import SimpleModelTrainer
from triage.storage import S3ModelStorageEngine


@contextmanager
def fake_metta(matrix_dict, metadata):
    matrix = pandas.DataFrame.from_dict(matrix_dict)
    with tempfile.NamedTemporaryFile() as matrix_file:
        with tempfile.NamedTemporaryFile('w') as metadata_file:
            hdf = pandas.HDFStore(matrix_file.name)
            hdf.put('title', matrix, data_columns=True)
            matrix_file.seek(0)

            yaml.dump(metadata, metadata_file)
            metadata_file.seek(0)
            yield (matrix_file.name, metadata_file.name)


def test_simple_model_trainer():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)

        model_config = {
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
            with fake_metta({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': ['good', 'bad']
            }, {'label_name': 'label'}) as (matrix_path, metadata_path):

                project_path = 'econ-dev/inspections'
                trainer = SimpleModelTrainer(
                    project_path=project_path,
                    model_storage_engine=S3ModelStorageEngine(s3_conn, project_path),
                    db_engine=engine
                )
                trainer.train_models(
                    training_set_path=matrix_path,
                    training_metadata_path=metadata_path,
                    model_config=model_config
                )

                # assert
                # 1. that the models table entries are present
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
