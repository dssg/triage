import boto3

from moto import mock_s3
from triage.db import ensure_db

from triage.predictors import Predictor
from tests.utils import fake_metta, fake_trained_model, fake_db
from triage.storage import S3ModelStorageEngine,\
    MettaMatrixStore,\
    InMemoryModelStorageEngine,\
    InMemoryMatrixStore
import datetime
import pandas

AS_OF_DATE = datetime.date(2016, 12, 21)


def test_predictor():
    with fake_db() as db_engine:
        ensure_db(db_engine)
        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')
            project_path = 'econ-dev/inspections'
            model_storage_engine = S3ModelStorageEngine(s3_conn, project_path)
            model_id = fake_trained_model(
                project_path,
                model_storage_engine,
                db_engine
            )
            predictor = Predictor(
                project_path,
                model_storage_engine,
                db_engine
            )
            # create prediction set
            with fake_metta({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': [7, 8]
            }, {
                'label_name': 'label',
                'end_time': AS_OF_DATE
            }) as (matrix_path, metadata_path):

                matrix_store = MettaMatrixStore(matrix_path, metadata_path)
                predictions = predictor.predict(model_id, matrix_store)

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
                    assert record[1] == AS_OF_DATE

                # that the entity ids match the given dataset
                assert sorted([record[0] for record in records]) == [1, 2]


def test_predictor_custom_colname():
    with fake_db() as db_engine:
        ensure_db(db_engine, entity_column_name='thing_id')
        project_path = 'tmp'
        model_storage_engine = InMemoryModelStorageEngine(project_path)
        model_id = fake_trained_model(
            project_path,
            model_storage_engine,
            db_engine
        )
        predictor = Predictor(
            project_path,
            model_storage_engine,
            db_engine,
            entity_column_name='thing_id'
        )
        # create prediction set
        matrix = pandas.DataFrame.from_dict({
            'thing_id': [1, 2],
            'feature_one': [3, 4],
            'feature_two': [5, 6],
            'label': [7, 8]
        }).set_index('thing_id')
        metadata = {
            'label_name': 'label',
            'end_time': AS_OF_DATE
        }

        matrix_store = InMemoryMatrixStore(matrix, metadata)
        predictions = predictor.predict(model_id, matrix_store)

        # assert
        # 1. that the returned predictions are of the desired length
        assert len(predictions) == 2

        # 2. that the predictions table entries are present and
        # can be linked to the original models
        records = [
            row for row in
            db_engine.execute('''select thing_id, as_of_date
            from results.predictions
            join results.models using (model_id)''')
        ]
        assert len(records) == 2

        # 3. that the contained as_of_dates match what we sent in
        for record in records:
            assert record[1] == AS_OF_DATE

        # that the entity ids match the given dataset
        assert sorted([record[0] for record in records]) == [1, 2]
