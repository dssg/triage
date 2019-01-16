from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import make_transient
import datetime
from unittest.mock import Mock
from numpy.testing import assert_array_equal
import pandas

from triage.component.results_schema import TestPrediction, Matrix, Model
from triage.database_reflection import table_has_data

from triage.component.catwalk.predictors import Predictor
from tests.utils import (
    MockTrainedModel,
    matrix_creator,
    matrix_metadata_creator,
    get_matrix_store,
    rig_engines,
)
import pytest


AS_OF_DATE = datetime.date(2016, 12, 21)

with_matrix_types = pytest.mark.parametrize(
    ('matrix_type',),
    [
        ('train',),
        ('test',),
    ],
)


@contextmanager
def prepare():
    with rig_engines() as (db_engine, project_storage):
        train_matrix_uuid = "1234"
        try:
            session = sessionmaker(db_engine)()
            session.add(Matrix(matrix_uuid=train_matrix_uuid))

            # Create the fake trained model and store in db
            trained_model = MockTrainedModel()
            model_hash = "abcd"
            project_storage.model_storage_engine().write(trained_model, model_hash)
            db_model = Model(model_hash=model_hash, train_matrix_uuid=train_matrix_uuid)
            session.add(db_model)
            session.commit()
            yield project_storage, db_engine, db_model.model_id
        finally:
            session.close()


@pytest.fixture(name='predict_setup_args', scope='function')
def fixture_predict_setup_args():
    with prepare() as predict_setup_args:
        yield predict_setup_args


@with_matrix_types
def test_predictor_entity_index(matrix_type, predict_setup_args):
    (project_storage, db_engine, model_id) = predict_setup_args
    predictor = Predictor(project_storage.model_storage_engine(), db_engine)

    # Runs the same test for training and testing predictions
    matrix = matrix_creator(index="entity_id")
    metadata = matrix_metadata_creator(
        end_time=AS_OF_DATE, matrix_type=matrix_type, indices=["entity_id"]
    )

    matrix_store = get_matrix_store(project_storage, matrix, metadata)
    train_matrix_columns = matrix.columns[0:-1].tolist()

    predict_proba = predictor.predict(
        model_id,
        matrix_store,
        misc_db_parameters=dict(),
        train_matrix_columns=train_matrix_columns,
    )

    # assert
    # 1. that the returned predictions are of the desired length
    assert len(predict_proba) == 2

    # 2. that the predictions table entries are present and
    # can be linked to the original models
    records = [
        row
        for row in db_engine.execute(
            """select entity_id, as_of_date
        from {}_results.predictions
        join model_metadata.models using (model_id)""".format(
                matrix_type, matrix_type
            )
        )
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

    # Runs the same test for training and testing predictions
    new_matrix = matrix_creator(index="entity_id")
    new_metadata = matrix_metadata_creator(
        end_time=AS_OF_DATE + datetime.timedelta(days=1),
        matrix_type=matrix_type,
        indices=["entity_id"],
    )
    new_matrix_store = get_matrix_store(
        project_storage, new_matrix, new_metadata
    )

    predictor.predict(
        model_id,
        new_matrix_store,
        misc_db_parameters=dict(),
        train_matrix_columns=train_matrix_columns,
    )
    predictor.predict(
        model_id,
        matrix_store,
        misc_db_parameters=dict(),
        train_matrix_columns=train_matrix_columns,
    )
    records = [
        row
        for row in db_engine.execute(
            """select entity_id, as_of_date
        from {}_results.predictions
        join model_metadata.models using (model_id)""".format(
                matrix_type, matrix_type
            )
        )
    ]
    assert len(records) == 4

    # 6. That we can delete the model when done prediction on it
    predictor.delete_model(model_id)
    assert predictor.load_model(model_id) is None


@with_matrix_types
def test_predictor_composite_index(matrix_type, predict_setup_args):
    (project_storage, db_engine, model_id) = predict_setup_args
    predictor = Predictor(project_storage.model_storage_engine(), db_engine)

    dayone = datetime.datetime(2011, 1, 1)
    daytwo = datetime.datetime(2011, 1, 2)
    source_dict = {
        "entity_id": [1, 2, 1, 2],
        "as_of_date": [dayone, dayone, daytwo, daytwo],
        "feature_one": [3, 4, 5, 6],
        "feature_two": [5, 6, 7, 8],
        "label": [7, 8, 8, 7],
    }

    matrix = pandas.DataFrame.from_dict(source_dict).set_index(
        ["entity_id", "as_of_date"]
    )
    metadata = matrix_metadata_creator(matrix_type=matrix_type)
    matrix_store = get_matrix_store(project_storage, matrix, metadata)

    predict_proba = predictor.predict(
        model_id,
        matrix_store,
        misc_db_parameters=dict(),
        train_matrix_columns=["feature_one", "feature_two"],
    )

    # assert
    # 1. that the returned predictions are of the desired length
    assert len(predict_proba) == 4

    # 2. that the predictions table entries are present and
    # can be linked to the original models
    records = [
        row
        for row in db_engine.execute(
            """select entity_id, as_of_date
        from {}_results.predictions
        join model_metadata.models using (model_id)""".format(
                matrix_type, matrix_type
            )
        )
    ]
    assert len(records) == 4


@with_matrix_types
def test_predictor_save_predictions(matrix_type, predict_setup_args):
    (project_storage, db_engine, model_id) = predict_setup_args
    # if save_predictions is sent as False, don't save
    predictor = Predictor(project_storage.model_storage_engine(), db_engine, save_predictions=False)

    matrix = matrix_creator(index="entity_id")
    metadata = matrix_metadata_creator(
        end_time=AS_OF_DATE, matrix_type=matrix_type, indices=["entity_id"]
    )

    matrix_store = get_matrix_store(project_storage, matrix, metadata)
    train_matrix_columns = matrix.columns[0:-1].tolist()

    predict_proba = predictor.predict(
        model_id,
        matrix_store,
        misc_db_parameters=dict(),
        train_matrix_columns=train_matrix_columns,
    )

    # assert
    # 1. that the returned predictions are of the desired length
    assert len(predict_proba) == 2

    # 2. that the predictions table entries are present and
    # can be linked to the original models
    assert not table_has_data(f"{matrix_type}_predictions", db_engine)


@with_matrix_types
def test_predictor_needs_predictions(matrix_type, predict_setup_args):
    (project_storage, db_engine, model_id) = predict_setup_args
    # if not all of the predictions for the given model id and matrix are present in the db,
    # needs_predictions should return true. else, false
    predictor = Predictor(project_storage.model_storage_engine(), db_engine)

    matrix = matrix_creator(index="entity_id")
    metadata = matrix_metadata_creator(
        end_time=AS_OF_DATE, matrix_type=matrix_type, indices=["entity_id"]
    )

    matrix_store = get_matrix_store(project_storage, matrix, metadata)
    train_matrix_columns = matrix.columns[0:-1].tolist()

    # we haven't done anything yet, this should definitely need predictions
    assert predictor.needs_predictions(matrix_store, model_id)
    predictor.predict(
        model_id,
        matrix_store,
        misc_db_parameters=dict(),
        train_matrix_columns=train_matrix_columns,
    )
    # now that predictions have been made, this should no longer need predictions
    assert not predictor.needs_predictions(matrix_store, model_id)


def test_predictor_get_train_columns(predict_setup_args):
    (project_storage, db_engine, model_id) = predict_setup_args
    predictor = Predictor(project_storage.model_storage_engine(), db_engine)
    train_store = get_matrix_store(
        project_storage=project_storage,
        matrix=matrix_creator(),
        metadata=matrix_metadata_creator(matrix_type="train"),
    )

    # flip the order of some feature columns in the test matrix
    other_order_matrix = matrix_creator()
    order = other_order_matrix.columns.tolist()
    order[0], order[1] = order[1], order[0]
    other_order_matrix = other_order_matrix[order]
    test_store = get_matrix_store(
        project_storage=project_storage,
        matrix=other_order_matrix,
        metadata=matrix_metadata_creator(matrix_type="test"),
    )

    # Runs the same test for training and testing predictions
    for store, mat_type in zip((train_store, test_store), ("train", "test")):
        predict_proba = predictor.predict(
            model_id,
            store,
            misc_db_parameters=dict(),
            train_matrix_columns=train_store.columns(),
        )
        # assert
        # 1. that we calculated predictions
        assert len(predict_proba) > 0

        # 2. that the predictions table entries are present and
        # can be linked to the original models
        records = [
            row
            for row in db_engine.execute(
                """select entity_id, as_of_date
            from {}_results.predictions
            join model_metadata.models using (model_id)""".format(
                    mat_type, mat_type
                )
            )
        ]
        assert len(records) > 0


def test_predictor_retrieve(predict_setup_args):
    (project_storage, db_engine, model_id) = predict_setup_args
    predictor = Predictor(
        project_storage.model_storage_engine(), db_engine, replace=False
    )

    # create prediction set
    matrix = matrix_creator()
    metadata = matrix_metadata_creator()
    matrix_store = get_matrix_store(project_storage, matrix, metadata)

    predict_proba = predictor.predict(
        model_id,
        matrix_store,
        misc_db_parameters=dict(),
        train_matrix_columns=matrix.columns[0:-1].tolist(),
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
    try:
        reorder_session = sessionmaker(bind=db_engine)()
        obj = reorder_session.query(TestPrediction).first()
        reorder_session.delete(obj)
        reorder_session.commit()
    finally:
        reorder_session.close()

    make_transient(obj)
    try:
        reorder_session = sessionmaker(bind=db_engine)()
        reorder_session.add(obj)
        reorder_session.commit()
    finally:
        reorder_session.close()

    predictor.load_model = Mock()
    new_predict_proba = predictor.predict(
        model_id,
        matrix_store,
        misc_db_parameters=dict(),
        train_matrix_columns=matrix.columns[0:-1].tolist(),
    )
    assert_array_equal(new_predict_proba, predict_proba)
    assert not predictor.load_model.called
