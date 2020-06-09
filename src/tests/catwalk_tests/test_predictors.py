from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import make_transient
import datetime
from unittest.mock import Mock
from numpy.testing import assert_array_almost_equal
import pandas

from triage.component.results_schema import TestPrediction, Matrix, Model
from triage.component.catwalk.storage import TestMatrixType
from triage.component.catwalk.db import ensure_db
from tests.results_tests.factories import (
    MatrixFactory,
    ModelFactory,
    PredictionFactory,
    init_engine,
    session as factory_session
)
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


with_matrix_types = pytest.mark.parametrize(
    ('matrix_type',),
    [
        ('train',),
        ('test',),
    ],
)

MODEL_RANDOM_SEED = 123456


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
            db_model = Model(
                model_hash=model_hash,
                train_matrix_uuid=train_matrix_uuid,
                random_seed=MODEL_RANDOM_SEED
            )
            session.add(db_model)
            session.commit()
            yield project_storage, db_engine, db_model.model_id
        finally:
            session.close()


@pytest.fixture(name='predict_setup_args', scope='function')
def fixture_predict_setup_args():
    with prepare() as predict_setup_args:
        yield predict_setup_args


@pytest.fixture(name='predictor', scope='function')
def predictor(predict_setup_args):
    (project_storage, db_engine, model_id) = predict_setup_args
    return Predictor(project_storage.model_storage_engine(), db_engine, rank_order='worst')


@pytest.fixture(name='predict_proba', scope='function')
def prediction_results(matrix_type, predictor, predict_setup_args):
    (project_storage, db_engine, model_id) = predict_setup_args

    dayone = datetime.datetime(2011, 1, 1)
    daytwo = datetime.datetime(2011, 1, 2)
    source_dict = {
        "entity_id": [1, 2, 3, 1, 2, 3],
        "as_of_date": [dayone, dayone, dayone, daytwo, daytwo, daytwo],
        "feature_one": [3] * 6,
        "feature_two": [5] * 6,
        "label": [True, False] * 3
    }

    matrix = pandas.DataFrame.from_dict(source_dict)
    metadata = matrix_metadata_creator(matrix_type=matrix_type)
    matrix_store = get_matrix_store(project_storage, matrix, metadata)

    predict_proba = predictor.predict(
        model_id,
        matrix_store,
        misc_db_parameters=dict(),
        train_matrix_columns=["feature_one", "feature_two"],
    )
    return predict_proba


@with_matrix_types
def test_predictor(predict_proba):
    """assert that the returned predictions are of the desired length"""
    assert len(predict_proba) == 6


@with_matrix_types
def test_predictions_table(predictor, predict_proba, matrix_type):
    """assert that the predictions table entries are present, linked to the original models"""
    records = [
        row
        for row in predictor.db_engine.execute(
            """select entity_id, as_of_date
        from {}_results.predictions
        join triage_metadata.models using (model_id)""".format(
                matrix_type, matrix_type
            )
        )
    ]
    assert len(records) == 6


def update_ranks_test(
        predictor,
        entities_scores_labels,
        rank_col,
        expected_result,
        model_random_seed=12345,
        need_seed_data=True
):
    """Not a test in itself but rather a utility called by many of the ranking tests"""
    ensure_db(predictor.db_engine)
    init_engine(predictor.db_engine)
    model_id = 5
    matrix_uuid = "4567"
    matrix_type = "test"
    as_of_date = datetime.datetime(2012, 1, 1)
    if need_seed_data:
        matrix = MatrixFactory(matrix_uuid=matrix_uuid)
        model = ModelFactory(model_id=model_id, random_seed=model_random_seed)
        for entity_id, score, label in entities_scores_labels:
            PredictionFactory(
                model_rel=model,
                matrix_rel=matrix,
                as_of_date=as_of_date,
                entity_id=entity_id,
                score=score,
                label_value=int(label)
            )
        factory_session.commit()
    predictor.update_db_with_ranks(
        model_id=model_id,
        matrix_uuid=matrix_uuid,
        matrix_type=TestMatrixType,
    )
    ranks = tuple(
        row for row in predictor.db_engine.execute(f'''
select entity_id, {rank_col}::float
from {matrix_type}_results.predictions
where as_of_date = %s and model_id = %s and matrix_uuid = %s order by {rank_col} asc''',
                                                   (as_of_date, model_id, matrix_uuid)
                                                   )
    )
    assert ranks == expected_result

    # Test that the predictions metadata table is populated
    metadata_records = [row for row in predictor.db_engine.execute(
        f"""select tiebreaker_ordering, prediction_metadata.random_seed, models.random_seed
        from {matrix_type}_results.prediction_metadata
        join triage_metadata.models using (model_id)
        join triage_metadata.matrices using (matrix_uuid)
        """
    )]
    assert len(metadata_records) == 1
    tiebreaker_ordering, random_seed, received_model_random_seed = metadata_records[0]
    if tiebreaker_ordering == 'random':
        assert random_seed is model_random_seed
    else:
        assert not random_seed
    assert tiebreaker_ordering == predictor.rank_order
    assert received_model_random_seed == model_random_seed


def test_predictions_abs_ranks_no_ties(project_storage, db_engine):
    "test the rank_abs column, which should have no ties"
    update_ranks_test(
        predictor=Predictor(project_storage.model_storage_engine(), db_engine, 'worst'),
        entities_scores_labels=(
            (23, 0.95, True),
            (34, 0.94, False),
            (45, 0.92, True),
            (56, 0.92, False), # since 'worst' is picked, this negative label should be first
            (67, 0.89, True)
        ),
        rank_col='rank_abs_no_ties',
        expected_result=(
            (23, 1),
            (34, 2),
            (56, 3),
            (45, 4),
            (67, 5),
        )
    )


def test_predictions_pct_ranks_no_ties(project_storage, db_engine):
    "test the percentile rank column, which should be based on the without-ties absolute rank"
    update_ranks_test(
        predictor=Predictor(project_storage.model_storage_engine(), db_engine, 'worst'),
        entities_scores_labels=(
            (23, 0.95, True),
            (34, 0.94, False),
            (45, 0.92, True),
            (56, 0.92, False), # with 'worst' this one should show up first
            (67, 0.89, True)
        ),
        rank_col='rank_pct_no_ties',
        expected_result=(
            (67, 0.2),
            (45, 0.4),
            (56, 0.6),
            (34, 0.8),
            (23, 1.0),
        )
    )


def test_predictions_pct_ranks_with_ties(project_storage, db_engine):
    """test the pct_ranks_with_ties column
    """
    update_ranks_test(
        predictor=Predictor(project_storage.model_storage_engine(), db_engine, 'worst'),
        entities_scores_labels=(
            (23, 0.95, True),
            (34, 0.94, False),
            (45, 0.92, True),
            (56, 0.92, False),
            (67, 0.89, True)
        ),
        rank_col='rank_pct_with_ties',
        expected_result=(
            (67, 0.2),
            (45, 0.4),
            (56, 0.4),
            (34, 0.8),
            (23, 1.0),
        )
    )


def test_predictions_abs_ranks_with_ties(project_storage, db_engine):
    """test the abs_ranks_with_ties column, which should rank entities with
    the same scores at the same rank and then skip those ranks for the next one
    """
    update_ranks_test(
        predictor=Predictor(project_storage.model_storage_engine(), db_engine, 'worst'),
        entities_scores_labels=(
            (23, 0.95, True),
            (34, 0.94, False),
            (45, 0.92, True),
            (56, 0.92, False),
            (67, 0.89, True)
        ),
        rank_col='rank_abs_with_ties',
        expected_result=(
            (23, 1),
            (34, 2),
            (45, 3),
            (56, 3),
            (67, 5),
        )
    )


def test_predictions_abs_ranks_no_ties_best_order(project_storage, db_engine):
    """here the user tells us to give us the best ranking
    """
    update_ranks_test(
        predictor=Predictor(
            project_storage.model_storage_engine(),
            db_engine,
            rank_order='best'
        ),
        entities_scores_labels=(
            (23, 0.95, True),
            (34, 0.94, False),
            (45, 0.92, True), # with 'best' this one should show up first
            (56, 0.92, False),
            (67, 0.89, True)
        ),
        rank_col='rank_abs_no_ties',
        expected_result=(
            (23, 1),
            (34, 2),
            (45, 3),
            (56, 4),
            (67, 5),
        )
    )


def test_predictions_abs_ranks_no_ties_user_seed(project_storage, db_engine):
    "testing the 'abs_rank_no_ties' column, respecting a user-supplied seed"
    predictor = Predictor(
        project_storage.model_storage_engine(),
        db_engine,
        rank_order="random",
    )
    expected_result = (
        (23, 1),
        (34, 2),
        (56, 3),
        (45, 4),
        (67, 5),
    )
    update_ranks_test(
        predictor=predictor,
        entities_scores_labels=(
            (23, 0.95, True),
            (34, 0.94, False),
            (45, 0.92, True),
            (56, 0.92, False),
            (67, 0.89, True)
        ),
        rank_col='rank_abs_no_ties',
        model_random_seed=15,
        expected_result=expected_result
    )
    update_ranks_test(
        predictor=predictor,
        entities_scores_labels=(
            (23, 0.95, True),
            (34, 0.94, False),
            (45, 0.92, True),
            (56, 0.92, False),
            (67, 0.89, True)
        ),
        rank_col='rank_abs_no_ties',
        expected_result=expected_result,
        model_random_seed=15,
        need_seed_data=False
    )


def test_prediction_ranks_multiple_dates(project_storage, db_engine):
    """make sure that multiple as-of-dates in a single matrix are handled correctly.
    keep the other variables simple by making no within-date ties that would end up
    testing the tiebreaker logic, just data for two dates with data that could theoretically
    confound a bad ranking method:
    - a different order for entities in both dates
    - each date has some not in the other
    """
    ensure_db(db_engine)
    init_engine(db_engine)
    predictor = Predictor(project_storage.model_storage_engine(), db_engine, 'worst')
    model_id = 5
    matrix_uuid = "4567"
    matrix_type = "test"
    entities_dates_and_scores = (
        (23, datetime.datetime(2012, 1, 1), 0.95),
        (34, datetime.datetime(2012, 1, 1), 0.94),
        (45, datetime.datetime(2013, 1, 1), 0.92),
        (23, datetime.datetime(2013, 1, 1), 0.45),
    )
    expected_result = (
        (23, datetime.datetime(2012, 1, 1), 1),
        (34, datetime.datetime(2012, 1, 1), 2),
        (45, datetime.datetime(2013, 1, 1), 3),
        (23, datetime.datetime(2013, 1, 1), 4),
    )
    matrix = MatrixFactory(matrix_uuid=matrix_uuid)
    model = ModelFactory(model_id=model_id)
    for entity_id, as_of_date, score in entities_dates_and_scores:
        PredictionFactory(
            model_rel=model,
            matrix_rel=matrix,
            as_of_date=as_of_date,
            entity_id=entity_id,
            score=score
        )
    factory_session.commit()
    predictor.update_db_with_ranks(
        model_id=model_id,
        matrix_uuid=matrix_uuid,
        matrix_type=TestMatrixType,
    )
    ranks = tuple(
        row for row in predictor.db_engine.execute(f'''
select entity_id, as_of_date, rank_abs_no_ties
from {matrix_type}_results.predictions
where model_id = %s and matrix_uuid = %s order by rank_abs_no_ties''',
                                                   (model_id, matrix_uuid)
                                                   )
    )
    assert ranks == expected_result


@with_matrix_types
def test_predictor_save_predictions(matrix_type, predict_setup_args):
    """Test the save_predictions flag being set to False

    We still want to return predict_proba, but not save data to the DB
    """
    (project_storage, db_engine, model_id) = predict_setup_args
    # if save_predictions is sent as False, don't save
    predictor = Predictor(project_storage.model_storage_engine(), db_engine, rank_order='worst', save_predictions=False)

    matrix_store = get_matrix_store(project_storage)
    train_matrix_columns = matrix_store.columns()

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
    """Test that the logic that figures out if predictions are needed for a given model/matrix"""
    (project_storage, db_engine, model_id) = predict_setup_args
    # if not all of the predictions for the given model id and matrix are present in the db,
    # needs_predictions should return true. else, false
    predictor = Predictor(project_storage.model_storage_engine(), db_engine, 'worst')

    metadata = matrix_metadata_creator(matrix_type=matrix_type)
    matrix_store = get_matrix_store(project_storage, metadata=metadata)
    train_matrix_columns = matrix_store.columns()

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
    """Test behavior when train/test matrices are created with different column orders
    """
    (project_storage, db_engine, model_id) = predict_setup_args
    predictor = Predictor(project_storage.model_storage_engine(), db_engine, 'worst')
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
            join triage_metadata.models using (model_id)""".format(
                    mat_type, mat_type
                )
            )
        ]
        assert len(records) > 0


def test_predictor_retrieve(predict_setup_args):
    """Test the predictions retrieved from the database match the output from predict_proba"""
    (project_storage, db_engine, model_id) = predict_setup_args
    predictor = Predictor(
        project_storage.model_storage_engine(), db_engine, 'worst', replace=False
    )

    # create prediction set
    matrix_store = get_matrix_store(project_storage)

    predict_proba = predictor.predict(
        model_id,
        matrix_store,
        misc_db_parameters=dict(),
        train_matrix_columns=matrix_store.columns()
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
        train_matrix_columns=matrix_store.columns()
    )
    assert_array_almost_equal(new_predict_proba, predict_proba, decimal=5)
    assert not predictor.load_model.called
