from triage.component.catwalk.evaluation import ModelEvaluator, generate_binary_at_x
from triage.component.catwalk.metrics import Metric
import testing.postgresql

import numpy
from sqlalchemy import create_engine
from triage.component.catwalk.db import ensure_db
from tests.utils import fake_labels, fake_trained_model
from triage.component.catwalk.storage import InMemoryModelStorageEngine
import datetime


@Metric(greater_is_better=True)
def always_half(predictions_proba, predictions_binary, labels, parameters):
    return 0.5


def test_evaluating_early_warning():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        metric_groups = [{
            'metrics': ['precision@',
                        'recall@',
                        'true positives@',
                        'true negatives@',
                        'false positives@',
                        'false negatives@'],
            'thresholds': {
                'percentiles': [5.0, 10.0],
                'top_n': [5, 10]
            }
        }, {
            'metrics': ['f1',
                        'mediocre',
                        'accuracy',
                        'roc_auc',
                        'average precision score'],
        }, {
            'metrics': ['fbeta@'],
            'parameters': [{'beta': 0.75}, {'beta': 1.25}]
        }]

        training_metric_groups = [{'metrics': ['accuracy', 'roc_auc']}]

        custom_metrics = {'mediocre': always_half}

        model_evaluator = ModelEvaluator(metric_groups, training_metric_groups, db_engine,
            custom_metrics=custom_metrics
        )

        trained_model, model_id = fake_trained_model(
            'myproject',
            InMemoryModelStorageEngine('myproject'),
            db_engine
        )

        labels = fake_labels(5)
        as_of_date = datetime.date(2016, 5, 5)

        # Evaluate the testing metrics and test for all of them. Note that the test version
        # of the evaluations table can hold only 1 set of results at a time.
        model_evaluator.evaluate(
            trained_model.predict_proba(labels)[:, 1],
            labels,
            model_id,
            as_of_date,
            as_of_date,
            '1y',
            "Test"
        )
        records = [
            row[0] for row in
            db_engine.execute(
                '''select distinct(metric || parameter || matrix_type)
                from results.evaluations
                where model_id = %s and
                evaluation_start_time = %s
                order by 1''',
                (model_id, as_of_date)
            )
        ]
        assert records == [
            'accuracyTest',
            'average precision scoreTest',
            'f1Test',
            'false negatives@10.0_pctTest',
            'false negatives@10_absTest',
            'false negatives@5.0_pctTest',
            'false negatives@5_absTest',
            'false positives@10.0_pctTest',
            'false positives@10_absTest',
            'false positives@5.0_pctTest',
            'false positives@5_absTest',
            'fbeta@0.75_betaTest',
            'fbeta@1.25_betaTest',
            'mediocreTest',
            'precision@10.0_pctTest',
            'precision@10_absTest',
            'precision@5.0_pctTest',
            'precision@5_absTest',
            'recall@10.0_pctTest',
            'recall@10_absTest',
            'recall@5.0_pctTest',
            'recall@5_absTest',
            'roc_aucTest',
            'true negatives@10.0_pctTest',
            'true negatives@10_absTest',
            'true negatives@5.0_pctTest',
            'true negatives@5_absTest',
            'true positives@10.0_pctTest',
            'true positives@10_absTest',
            'true positives@5.0_pctTest',
            'true positives@5_absTest'
        ]

        # Evaluate the training metrics and test
        model_evaluator.evaluate(
            trained_model.predict_proba(labels)[:, 1],
            labels,
            model_id,
            as_of_date,
            as_of_date,
            '1y',
            "Train"
        )
        records = [
            row[0] for row in
            db_engine.execute(
                '''select distinct(metric || parameter || matrix_type)
                from results.evaluations
                where model_id = %s and
                evaluation_start_time = %s
                order by 1''',
                (model_id, as_of_date)
            )
        ]
        assert records == [ 'accuracyTrain', 'roc_aucTrain']


def test_model_scoring_inspections():
    print('\n')
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        metric_groups = [{
            'metrics': ['precision@', 'recall@', 'fpr@'],
            'thresholds': {'percentiles': [50.0], 'top_n': [3]}
        }]
        training_metric_groups = [{'metrics': ['accuracy'], 'thresholds': {'percentiles': [50.0]}}]

        model_evaluator = ModelEvaluator(metric_groups, training_metric_groups, db_engine)

        trained_model, model_id = fake_trained_model(
            'myproject',
            InMemoryModelStorageEngine('myproject'),
            db_engine
        )

        testing_labels = numpy.array([True, False, numpy.nan, True, False])
        testing_prediction_probas = numpy.array([0.56, 0.4, 0.55, 0.5, 0.3])

        training_labels = numpy.array([False, False, True, True, True, False, True, True])
        training_prediction_probas = numpy.array([0.6, 0.4, 0.55, 0.70, 0.3, 0.2, 0.8, 0.6])

        evaluation_start = datetime.datetime(2016, 4, 1)
        evaluation_end = datetime.datetime(2016, 7, 1)
        example_as_of_date_frequency = '1d'

        # Evaluate testing matrix and test the results
        model_evaluator.evaluate(
            testing_prediction_probas,
            testing_labels,
            model_id,
            evaluation_start,
            evaluation_end,
            example_as_of_date_frequency,
            matrix_type="Test"
        )
        for record in db_engine.execute(
            '''select * from results.evaluations
            where model_id = %s and evaluation_start_time = %s and matrix_type = 'Test'
            order by 1''',
            (model_id, evaluation_start)
        ):
            assert record['num_labeled_examples'] == 4
            assert record['num_positive_labels'] == 2
            if 'pct' in record['parameter']:
                assert record['num_labeled_above_threshold'] == 1
            else:
                assert record['num_labeled_above_threshold'] == 2

        # Evaluate the training matrix and test the results
        model_evaluator.evaluate(
                    training_prediction_probas,
                    training_labels,
                    model_id,
                    evaluation_start,
                    evaluation_end,
                    example_as_of_date_frequency,
                    matrix_type="Train"
        )
        for record in db_engine.execute(
            '''select * from results.evaluations
            where model_id = %s and evaluation_start_time = %s and matrix_type = 'Train'
            order by 1''',
            (model_id, evaluation_start)
        ):
            assert record['num_labeled_examples'] == 8
            assert record['num_positive_labels'] == 5
            assert record['value'] == 0.625


def test_generate_binary_at_x():
    input_list = [0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6]

    # bug can arise when the same value spans both sides of threshold
    assert generate_binary_at_x(input_list, 50, 'percentile') == \
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    assert generate_binary_at_x(input_list, 2) == \
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
