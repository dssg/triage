import boto3
import testing.postgresql

from moto import mock_s3
from sqlalchemy import create_engine
from triage.db import ensure_db

from triage.predictors import Predictor
from tests.utils import fake_metta, fake_trained_model
import datetime

AS_OF_DATE = datetime.date(2016, 12, 21)


def test_predictor():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)

        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')
            project_path = 'econ-dev/inspections'
            model_id = fake_trained_model(project_path, s3_conn, engine)
            predictor = Predictor(project_path, s3_conn, engine)
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

                predictions = predictor.predict(model_id, matrix_path, metadata_path)

                # assert
                # 1. that the returned predictions are of the desired length
                assert len(predictions) == 2

                # 2. that the predictions table entries are present and
                # can be linked to the original models
                records = [
                    row for row in
                    engine.execute('select entity_id, as_of_date from results.predictions join results.models using (model_id)')
                ]
                assert len(records) == 2

                # 3. that the contained as_of_dates match what we sent in
                for record in records:
                    assert record[1] == AS_OF_DATE

                # that the entity ids match the given dataset
                assert sorted([record[0] for record in records]) == [1, 2]
