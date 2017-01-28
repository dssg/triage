from triage.model_trainers import ModelTrainer
from triage.predictors import Predictor

import boto3
import logging
import testing.postgresql

from moto import mock_s3
from sqlalchemy import create_engine
from triage.db import ensure_db

from triage.storage import S3ModelStorageEngine, InMemoryMatrixStore
import datetime
import pandas

AS_OF_DATE = datetime.date(2016, 12, 21)


def test_predictor():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)

        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')
            project_path = 'econ-dev/inspections'

            train_matrix = pandas.DataFrame.from_dict({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': [7, 8]
            }).set_index('entity_id')
            train_metadata = {
                'label_name': 'label',
            }

            train_store = InMemoryMatrixStore(train_matrix, train_metadata)
            model_storage_engine = S3ModelStorageEngine(s3_conn, project_path)
            trainer = ModelTrainer(
                project_path,
                model_storage_engine,
                train_store,
                db_engine,
            )
            predictor = Predictor(project_path, model_storage_engine, db_engine)

            grid_config = {
                'sklearn.linear_model.LogisticRegression': {
                    'C': [0.00001, 0.0001],
                    'penalty': ['l1', 'l2'],
                    'random_state': [2193]
                }
            }
            model_ids = trainer.train_models(grid_config=grid_config, misc_db_parameters=dict())

            test_matrix = pandas.DataFrame.from_dict({
                'entity_id': [3],
                'feature_one': [8],
                'feature_two': [5],
                'label': [5]
            }).set_index('entity_id')
            test_metadata = {
                'label_name': 'label',
                'end_time': AS_OF_DATE
            }
            test_store = InMemoryMatrixStore(test_matrix, test_metadata)

            for model_id in model_ids:
                predictor.predict(model_id, test_store)

            # assert
            # 1. that the predictions table entries are present and
            # can be linked to the original models
            records = [
                row for row in
                db_engine.execute('select entity_id, as_of_date from results.predictions join results.models using (model_id)')
            ]
            assert len(records) == 4 # one entity, four models

            # 2. that the contained as_of_dates and entity_ids match what we sent in
            for record in records:
                assert record[1] == AS_OF_DATE
                assert record[0] == 3
