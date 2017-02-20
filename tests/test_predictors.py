import boto3
import testing.postgresql

from moto import mock_s3
from sqlalchemy import create_engine
from triage.db import ensure_db
import pandas

from triage.predictors import Predictor
from tests.utils import fake_trained_model
from triage.storage import S3ModelStorageEngine, InMemoryMatrixStore
import datetime

AS_OF_DATE = datetime.date(2016, 12, 21)


def test_predictor():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)

        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')
            project_path = 'econ-dev/inspections'
            model_storage_engine = S3ModelStorageEngine(s3_conn, project_path)
            _, model_id = \
                fake_trained_model(project_path, model_storage_engine, db_engine)
            predictor = Predictor(project_path, model_storage_engine, db_engine)
            # create prediction set
            matrix = pandas.DataFrame.from_dict({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': [7, 8]
            }).set_index('entity_id')
            metadata = {
                'label_name': 'label',
                'end_time': AS_OF_DATE
            }

            matrix_store = InMemoryMatrixStore(matrix, metadata)
            predictions = predictor.predict(model_id, matrix_store, misc_db_parameters=dict())

            # assert
            # 1. that the returned predictions are of the desired length
            assert len(predictions) == 2

            # 2. that the predictions table entries are present and
            # can be linked to the original models
            records = [
                row for row in
                db_engine.execute('''select entity_id, as_of_date
                from results.predictions
                join results.models using (model_id)''')
            ]
            assert len(records) == 2

            # 3. that the contained as_of_dates match what we sent in
            for record in records:
                assert record[1].date() == AS_OF_DATE

            # 4. that the entity ids match the given dataset
            assert sorted([record[0] for record in records]) == [1, 2]

            # 5. running with same model_id, different as of date
            # then with same as of date only replaces the records
            # with the same date
            new_matrix = pandas.DataFrame.from_dict({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': [7, 8]
            }).set_index('entity_id')
            new_metadata = {
                'label_name': 'label',
                'end_time': AS_OF_DATE + datetime.timedelta(days=1)
            }
            new_matrix_store = InMemoryMatrixStore(new_matrix, new_metadata)
            predictor.predict(model_id, new_matrix_store, misc_db_parameters=dict())
            predictor.predict(model_id, matrix_store, misc_db_parameters=dict())
            records = [
                row for row in
                db_engine.execute('''select entity_id, as_of_date
                from results.predictions
                join results.models using (model_id)''')
            ]
            assert len(records) == 4

            # 6. That we can delete the model when done prediction on it
            predictor.delete_model(model_id)
            assert predictor.load_model(model_id) == None
