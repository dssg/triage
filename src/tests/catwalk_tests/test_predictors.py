import boto3
import testing.postgresql
from moto import mock_s3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import make_transient
import datetime
from unittest.mock import Mock
from numpy.testing import assert_array_equal
import tempfile
import pandas

from triage.component.results_schema import TestPrediction, TrainPrediction
from triage.component.catwalk.db import ensure_db

from triage.component.catwalk.predictors import Predictor
from tests.utils import fake_trained_model, sample_metta_csv_diff_order
from triage.component.catwalk.storage import (
    InMemoryModelStorageEngine,
    S3ModelStorageEngine,
    MatrixStore
    )
from tests.results_tests.factories import init_engine, session, MatrixFactory


AS_OF_DATE = datetime.date(2016, 12, 21)


def test_predictor():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        init_engine(db_engine)

        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')
            project_path = 'econ-dev/inspections'
            model_storage_engine = S3ModelStorageEngine(project_path)

            _, model_id = \
                fake_trained_model(project_path, model_storage_engine, db_engine, train_matrix_uuid='1234')

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
                'end_time': AS_OF_DATE,
                'label_timespan': '3month',
                'metta-uuid': '1234',
                'indices': ['entity_id'],
            }

            train_matrix_columns = ['feature_one', 'feature_two']

            # Runs the same test for training and testing predictions
            for mat_type in ("train", "test"):
                # Create the matrix to be tested and store in db
                metadata['matrix_type'] = mat_type

                matrix_store = MatrixStore()
                matrix_store.matrix = matrix
                matrix_store.metadata = metadata

                # Note, the first time 'matrix' is used, the label column is popped.
                # It must be added back in to 'matrix' to create another matrix_store.
                matrix['label'] = [7, 8]

                predict_proba = predictor.predict(
                    model_id,
                    matrix_store,
                    misc_db_parameters=dict(),
                    train_matrix_columns=train_matrix_columns
                )

                # assert
                # 1. that the returned predictions are of the desired length
                assert len(predict_proba) == 2

                # 2. that the predictions table entries are present and
                # can be linked to the original models
                records = [
                    row for row in
                    db_engine.execute('''select entity_id, as_of_date
                    from {}_results.predictions
                    join model_metadata.models using (model_id)'''.format(mat_type, mat_type))
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
                'end_time': AS_OF_DATE + datetime.timedelta(days=1),
                'label_timespan': '3month',
                'metta-uuid': '1234',
                'indices': ['entity_id'],
            }

            # Runs the same test for training and testing predictions
            for mat_type in ("train", "test"):

                # Create the matrix to be tested and store in db
                new_metadata['matrix_type'] = mat_type

                new_matrix_store = MatrixStore()
                new_matrix_store.matrix = new_matrix
                new_matrix_store.metadata = new_metadata

                # Adding 'label' column back into new_matrix
                new_matrix['label'] = [7, 8]

                predictor.predict(
                    model_id,
                    new_matrix_store,
                    misc_db_parameters=dict(),
                    train_matrix_columns=train_matrix_columns
                )
                predictor.predict(
                    model_id,
                    matrix_store,
                    misc_db_parameters=dict(),
                    train_matrix_columns=train_matrix_columns
                )
                records = [
                    row for row in
                    db_engine.execute('''select entity_id, as_of_date
                    from {}_results.predictions
                    join model_metadata.models using (model_id)'''.format(mat_type, mat_type))
                ]
                assert len(records) == 4

            # 6. That we can delete the model when done prediction on it
            predictor.delete_model(model_id)
            assert predictor.load_model(model_id) == None


def test_predictor_composite_index():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        init_engine(db_engine)

        project_path = 'econ-dev/inspections'
        model_storage_engine = InMemoryModelStorageEngine(project_path)

        _, model_id = \
            fake_trained_model(project_path, model_storage_engine, db_engine, train_matrix_uuid='1234')

        predictor = Predictor(project_path, model_storage_engine, db_engine)

        dayone = datetime.datetime(2011, 1, 1)
        daytwo = datetime.datetime(2011, 1, 2)

        # create prediction set
        matrix = pandas.DataFrame.from_dict({
            'entity_id': [1, 2, 1, 2],
            'as_of_date': [dayone, dayone, daytwo, daytwo],
            'feature_one': [3, 4, 5, 6],
            'feature_two': [5, 6, 7, 8],
            'label': [7, 8, 8, 7]
        }).set_index(['entity_id', 'as_of_date'])
        metadata = {
            'label_name': 'label',
            'end_time': AS_OF_DATE,
            'label_timespan': '3month',
            'metta-uuid': '1234',
            'indices': ['entity_id', 'as_of_date'],
        }

        # Runs the same test for training and testing predictions
        for mat_type in ("train", "test"):

            # Create the matrix to be tested and store in db
            metadata['matrix_type'] = mat_type
            matrix_store = MatrixStore()
            matrix_store.matrix = matrix
            matrix_store.metadata = metadata

            # Adding 'label' column back into matrix
            matrix['label'] = [7, 8, 8, 7]

            predict_proba = predictor.predict(
                model_id,
                matrix_store,
                misc_db_parameters=dict(),
                train_matrix_columns=['feature_one', 'feature_two']
            )

            # assert
            # 1. that the returned predictions are of the desired length
            assert len(predict_proba) == 4

            # 2. that the predictions table entries are present and
            # can be linked to the original models
            records = [
                row for row in
                db_engine.execute('''select entity_id, as_of_date
                from {}_results.predictions
                join model_metadata.models using (model_id)'''.format(mat_type, mat_type))
            ]
            assert len(records) == 4


def test_predictor_get_train_columns():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        init_engine(db_engine)

        project_path = 'econ-dev/inspections'
        with tempfile.TemporaryDirectory() as temp_dir:
            train_store, test_store = sample_metta_csv_diff_order(temp_dir)

            model_storage_engine = InMemoryModelStorageEngine(project_path)
            _, model_id = \
                fake_trained_model(
                    project_path,
                    model_storage_engine,
                    db_engine,
                    train_matrix_uuid=train_store.uuid
                )
            predictor = Predictor(project_path, model_storage_engine, db_engine)

            # The train_store uuid is stored in fake_trained_model. Storing the other
            MatrixFactory(matrix_uuid = test_store.uuid)
            session.commit()

            # Runs the same test for training and testing predictions
            for store, mat_type in zip((train_store,test_store),("train", "test")):
                predict_proba = predictor.predict(
                    model_id,
                    store,
                    misc_db_parameters=dict(),
                    train_matrix_columns=train_store.columns()
                )
                # assert
                # 1. that we calculated predictions
                assert len(predict_proba) > 0

                # 2. that the predictions table entries are present and
                # can be linked to the original models
                records = [
                    row for row in
                    db_engine.execute('''select entity_id, as_of_date
                    from {}_results.predictions
                    join model_metadata.models using (model_id)'''.format(mat_type, mat_type))
                ]
                assert len(records) > 0


def test_predictor_retrieve():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        init_engine(db_engine)

        project_path = 'econ-dev/inspections'
        model_storage_engine = InMemoryModelStorageEngine(project_path)

        _, model_id = \
            fake_trained_model(project_path, model_storage_engine, db_engine, train_matrix_uuid='1234')

        predictor = Predictor(project_path, model_storage_engine, db_engine, replace=False)

        dayone = datetime.date(2011, 1, 1).strftime(predictor.expected_matrix_ts_format)
        daytwo = datetime.date(2011, 1, 2).strftime(predictor.expected_matrix_ts_format)

        # create prediction set
        matrix_data = {
            'entity_id': [1, 2, 1, 2],
            'as_of_date': [dayone, dayone, daytwo, daytwo],
            'feature_one': [3, 4, 5, 6],
            'feature_two': [5, 6, 7, 8],
            'label': [7, 8, 8, 7]
        }
        matrix = pandas.DataFrame.from_dict(matrix_data)\
            .set_index(['entity_id', 'as_of_date'])
        metadata = {
            'label_name': 'label',
            'end_time': AS_OF_DATE,
            'label_timespan': '3month',
            'metta-uuid': '1234',
            'indices': ['entity_id', 'as_of_date'],
            'matrix_type': 'test'
        }

        matrix_store = MatrixStore()
        matrix_store.matrix = matrix
        matrix_store.metadata = metadata

        predict_proba = predictor.predict(
            model_id,
            matrix_store,
            misc_db_parameters=dict(),
            train_matrix_columns=['feature_one', 'feature_two']
        )

        # When run again, the predictions retrieved from the database
        # should match.
        #
        # Some trickiness here. Let's explain:
        #
        # If we are not careful, retrieving predictions from the database and
        # presenting them as a numpy array can result in a bad ordering,
        # since the given matrix may not be 'ordered' by some criteria
        # that can be easily represented by an ORDER BY clause.
        #
        # It will sometimes work, because without ORDER BY you will get
        # it back in the table's physical order, which unless something has
        # happened to the table will be the order you inserted it,
        # which could very well be the order in the matrix.
        # So it's not a bug that would necessarily immediately show itself,
        # but when it does go wrong your scores will be garbage.
        #
        # So we simulate a table order mutation that can happen over time:
        # Remove the first row and put it at the end.
        # If the Predictor doesn't explicitly reorder the results, this will fail
        # Only running on TestPrediction because TrainPrediction behaves the exact same way
        reorder_session = sessionmaker(bind=db_engine)()
        obj = reorder_session.query(TestPrediction).first()
        reorder_session.delete(obj)
        reorder_session.commit()

        make_transient(obj)
        reorder_session = sessionmaker(bind=db_engine)()
        reorder_session.add(obj)
        reorder_session.commit()

        predictor.load_model = Mock()
        new_predict_proba = predictor.predict(
            model_id,
            matrix_store,
            misc_db_parameters=dict(),
            train_matrix_columns=['feature_one', 'feature_two']
        )
        assert_array_equal(new_predict_proba, predict_proba)
        assert not predictor.load_model.called
