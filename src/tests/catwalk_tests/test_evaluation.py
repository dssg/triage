from triage.component.catwalk.evaluation import ModelEvaluator, generate_binary_at_x
from triage.component.catwalk.metrics import Metric
import testing.postgresql
import datetime

import numpy
from sqlalchemy import create_engine
from triage.component.catwalk.db import ensure_db
from tests.utils import fake_labels, fake_trained_model, MockMatrixStore
from tests.results_tests.factories import ModelFactory, EvaluationFactory, init_engine, session


@Metric(greater_is_better=True)
def always_half(predictions_proba, predictions_binary, labels, parameters):
    return 0.5


def test_evaluating_early_warning():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        testing_metric_groups = [
            {
                "metrics": [
                    "precision@",
                    "recall@",
                    "true positives@",
                    "true negatives@",
                    "false positives@",
                    "false negatives@",
                ],
                "thresholds": {"percentiles": [5.0, 10.0], "top_n": [5, 10]},
            },
            {
                "metrics": [
                    "f1",
                    "mediocre",
                    "accuracy",
                    "roc_auc",
                    "average precision score",
                ]
            },
            {"metrics": ["fbeta@"], "parameters": [{"beta": 0.75}, {"beta": 1.25}]},
        ]

        training_metric_groups = [{"metrics": ["accuracy", "roc_auc"]}]

        custom_metrics = {"mediocre": always_half}

        model_evaluator = ModelEvaluator(
            testing_metric_groups,
            training_metric_groups,
            db_engine,
            custom_metrics=custom_metrics,
        )

        labels = fake_labels(5)
        fake_train_matrix_store = MockMatrixStore("train", "efgh", 5, db_engine, labels)
        fake_test_matrix_store = MockMatrixStore("test", "1234", 5, db_engine, labels)

        trained_model, model_id = fake_trained_model(db_engine)

        # Evaluate the testing metrics and test for all of them.
        model_evaluator.evaluate(
            trained_model.predict_proba(labels)[:, 1], fake_test_matrix_store, model_id
        )
        records = [
            row[0]
            for row in db_engine.execute(
                """select distinct(metric || parameter)
                from test_results.evaluations
                where model_id = %s and
                evaluation_start_time = %s
                order by 1""",
                (model_id, fake_test_matrix_store.as_of_dates[0]),
            )
        ]
        assert records == [
            "accuracy",
            "average precision score",
            "f1",
            "false negatives@10.0_pct",
            "false negatives@10_abs",
            "false negatives@5.0_pct",
            "false negatives@5_abs",
            "false positives@10.0_pct",
            "false positives@10_abs",
            "false positives@5.0_pct",
            "false positives@5_abs",
            "fbeta@0.75_beta",
            "fbeta@1.25_beta",
            "mediocre",
            "precision@10.0_pct",
            "precision@10_abs",
            "precision@5.0_pct",
            "precision@5_abs",
            "recall@10.0_pct",
            "recall@10_abs",
            "recall@5.0_pct",
            "recall@5_abs",
            "roc_auc",
            "true negatives@10.0_pct",
            "true negatives@10_abs",
            "true negatives@5.0_pct",
            "true negatives@5_abs",
            "true positives@10.0_pct",
            "true positives@10_abs",
            "true positives@5.0_pct",
            "true positives@5_abs",
        ]

        # ensure that the matrix uuid is present
        matrix_uuids = [
            row[0]
            for row in db_engine.execute("select matrix_uuid from test_results.evaluations")
        ]
        assert all(matrix_uuid == "1234" for matrix_uuid in matrix_uuids)

        # Evaluate the training metrics and test
        model_evaluator.evaluate(
            trained_model.predict_proba(labels)[:, 1], fake_train_matrix_store, model_id
        )
        records = [
            row[0]
            for row in db_engine.execute(
                """select distinct(metric || parameter)
                from train_results.evaluations
                where model_id = %s and
                evaluation_start_time = %s
                order by 1""",
                (model_id, fake_train_matrix_store.as_of_dates[0]),
            )
        ]
        assert records == ["accuracy", "roc_auc"]

        # ensure that the matrix uuid is present
        matrix_uuids = [
            row[0]
            for row in db_engine.execute("select matrix_uuid from train_results.evaluations")
        ]
        assert all(matrix_uuid == "efgh" for matrix_uuid in matrix_uuids)


def test_model_scoring_inspections():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        testing_metric_groups = [
            {
                "metrics": ["precision@", "recall@", "fpr@"],
                "thresholds": {"percentiles": [50.0], "top_n": [3]},
            },
            {
                # ensure we test a non-thresholded metric as well
                "metrics": ["accuracy"]
            },
        ]
        training_metric_groups = [
            {"metrics": ["accuracy"], "thresholds": {"percentiles": [50.0]}}
        ]

        model_evaluator = ModelEvaluator(
            testing_metric_groups, training_metric_groups, db_engine
        )

        testing_labels = numpy.array([True, False, numpy.nan, True, False])
        testing_prediction_probas = numpy.array([0.56, 0.4, 0.55, 0.5, 0.3])

        training_labels = numpy.array(
            [False, False, True, True, True, False, True, True]
        )
        training_prediction_probas = numpy.array(
            [0.6, 0.4, 0.55, 0.70, 0.3, 0.2, 0.8, 0.6]
        )

        fake_train_matrix_store = MockMatrixStore(
            "train", "efgh", 5, db_engine, training_labels
        )
        fake_test_matrix_store = MockMatrixStore(
            "test", "1234", 5, db_engine, testing_labels
        )

        trained_model, model_id = fake_trained_model(db_engine)

        # Evaluate testing matrix and test the results
        model_evaluator.evaluate(
            testing_prediction_probas, fake_test_matrix_store, model_id
        )
        for record in db_engine.execute(
            """select * from test_results.evaluations
            where model_id = %s and evaluation_start_time = %s
            order by 1""",
            (model_id, fake_test_matrix_store.as_of_dates[0]),
        ):
            assert record["num_labeled_examples"] == 4
            assert record["num_positive_labels"] == 2
            if record["parameter"] == "":
                assert record["num_labeled_above_threshold"] == 4
            elif "pct" in record["parameter"]:
                assert record["num_labeled_above_threshold"] == 1
            else:
                assert record["num_labeled_above_threshold"] == 2

        # Evaluate the training matrix and test the results
        model_evaluator.evaluate(
            training_prediction_probas, fake_train_matrix_store, model_id
        )
        for record in db_engine.execute(
            """select * from train_results.evaluations
            where model_id = %s and evaluation_start_time = %s
            order by 1""",
            (model_id, fake_train_matrix_store.as_of_dates[0]),
        ):
            assert record["num_labeled_examples"] == 8
            assert record["num_positive_labels"] == 5
            assert record["value"] == 0.625


def test_ModelEvaluator_needs_evaluation(db_engine):
    ensure_db(db_engine)
    init_engine(db_engine)
    # TEST SETUP:

    # create two models: one that has zero evaluations,
    # one that has an evaluation for precision@100_abs
    model_with_evaluations = ModelFactory()
    model_without_evaluations = ModelFactory()

    eval_time = datetime.datetime(2016, 1, 1)
    as_of_date_frequency = "3d"
    EvaluationFactory(
        model_rel=model_with_evaluations,
        evaluation_start_time=eval_time,
        evaluation_end_time=eval_time,
        as_of_date_frequency=as_of_date_frequency,
        metric="precision@",
        parameter="100_abs"
    )
    session.commit()

    # make a test matrix to pass in
    metadata_overrides = {
        'as_of_date_frequency': as_of_date_frequency,
        'end_time': eval_time,
    }
    test_matrix_store = MockMatrixStore(
        "test", "1234", 5, db_engine, metadata_overrides=metadata_overrides
    )
    train_matrix_store = MockMatrixStore(
        "train", "2345", 5, db_engine, metadata_overrides=metadata_overrides
    )

    # the evaluated model has test evaluations for precision, but not recall,
    # so this needs evaluations
    assert ModelEvaluator(
        testing_metric_groups=[{
            "metrics": ["precision@", "recall@"],
            "thresholds": {"top_n": [100]},
        }],
        training_metric_groups=[],
        db_engine=db_engine
    ).needs_evaluations(
        matrix_store=test_matrix_store,
        model_id=model_with_evaluations.model_id,
    )

    # the evaluated model has test evaluations for precision,
    # so this should not need evaluations
    assert not ModelEvaluator(
        testing_metric_groups=[{
            "metrics": ["precision@"],
            "thresholds": {"top_n": [100]},
        }],
        training_metric_groups=[],
        db_engine=db_engine
    ).needs_evaluations(
        matrix_store=test_matrix_store,
        model_id=model_with_evaluations.model_id,
    )

    # the non-evaluated model has no evaluations,
    # so this should need evaluations
    assert ModelEvaluator(
        testing_metric_groups=[{
            "metrics": ["precision@"],
            "thresholds": {"top_n": [100]},
        }],
        training_metric_groups=[],
        db_engine=db_engine
    ).needs_evaluations(
        matrix_store=test_matrix_store,
        model_id=model_without_evaluations.model_id,
    )

    # the evaluated model has no *train* evaluations,
    # so the train matrix should need evaluations
    assert ModelEvaluator(
        testing_metric_groups=[{
            "metrics": ["precision@"],
            "thresholds": {"top_n": [100]},
        }],
        training_metric_groups=[{
            "metrics": ["precision@"],
            "thresholds": {"top_n": [100]},
        }],
        db_engine=db_engine
    ).needs_evaluations(
        matrix_store=train_matrix_store,
        model_id=model_with_evaluations.model_id,
    )
    session.close()
    session.remove()


def test_generate_binary_at_x():
    input_list = [0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6]

    # bug can arise when the same value spans both sides of threshold
    assert generate_binary_at_x(input_list, 50, "percentile") == [
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]

    assert generate_binary_at_x(input_list, 2) == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
