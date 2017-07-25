from catwalk.evaluation import ModelEvaluator, generate_binary_at_x
from catwalk.metrics import Metric
import testing.postgresql

import numpy
from sqlalchemy import create_engine
from catwalk.db import ensure_db
from tests.utils import fake_labels, fake_trained_model
from catwalk.storage import InMemoryModelStorageEngine
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

        custom_metrics = {'mediocre': always_half}

        model_evaluator = ModelEvaluator(
            metric_groups,
            db_engine,
            custom_metrics=custom_metrics
        )

        trained_model, model_id = fake_trained_model(
            'myproject',
            InMemoryModelStorageEngine('myproject'),
            db_engine
        )

        labels = fake_labels(5)
        as_of_date = datetime.date(2016, 5, 5)
        model_evaluator.evaluate(
            trained_model.predict_proba(labels)[:, 1],
            labels,
            model_id,
            as_of_date,
            as_of_date,
            '1y'
        )

        # assert
        # that all of the records are there
        records = [
            row[0] for row in
            db_engine.execute(
                '''select distinct(metric || parameter)
                from results.evaluations
                where model_id = %s and
                evaluation_start_time = %s order by 1''',
                (model_id, as_of_date)
            )
        ]
        assert records == [
            'accuracy',
            'average precision score',
            'f1',
            'false negatives@10.0_pct',
            'false negatives@10_abs',
            'false negatives@5.0_pct',
            'false negatives@5_abs',
            'false positives@10.0_pct',
            'false positives@10_abs',
            'false positives@5.0_pct',
            'false positives@5_abs',
            'fbeta@0.75_beta',
            'fbeta@1.25_beta',
            'mediocre',
            'precision@10.0_pct',
            'precision@10_abs',
            'precision@5.0_pct',
            'precision@5_abs',
            'recall@10.0_pct',
            'recall@10_abs',
            'recall@5.0_pct',
            'recall@5_abs',
            'roc_auc',
            'true negatives@10.0_pct',
            'true negatives@10_abs',
            'true negatives@5.0_pct',
            'true negatives@5_abs',
            'true positives@10.0_pct',
            'true positives@10_abs',
            'true positives@5.0_pct',
            'true positives@5_abs'
        ]


def test_model_scoring_inspections():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        metric_groups = [{
            'metrics': ['precision@', 'recall@', 'fpr@'],
            'thresholds': {'percentiles': [50.0], 'top_n': [3]}
        }]

        model_evaluator = ModelEvaluator(metric_groups, db_engine)

        _, model_id = fake_trained_model(
            'myproject',
            InMemoryModelStorageEngine('myproject'),
            db_engine
        )

        labels = numpy.array([True, False, numpy.nan, True, False])
        prediction_probas = numpy.array([0.56, 0.4, 0.55, 0.5, 0.3])
        evaluation_start = datetime.datetime(2016, 4, 1)
        evaluation_end = datetime.datetime(2016, 7, 1)
        example_frequency = '1d'
        model_evaluator.evaluate(
            prediction_probas,
            labels,
            model_id,
            evaluation_start,
            evaluation_end,
            example_frequency
        )

        for record in db_engine.execute(
            '''select * from results.evaluations
            where model_id = %s and evaluation_start_time = %s order by 1''',
            (model_id, evaluation_start)
        ):
            assert record['num_labeled_examples'] == 4
            assert record['num_positive_labels'] == 2
            if 'pct' in record['parameter']:
                assert record['num_labeled_above_threshold'] == 1
            else:
                assert record['num_labeled_above_threshold'] == 2


def test_generate_binary_at_x():
    input_list = [0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6]

    # bug can arise when the same value spans both sides of threshold
    assert generate_binary_at_x(input_list, 50, 'percentile') == \
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    assert generate_binary_at_x(input_list, 2) == \
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


