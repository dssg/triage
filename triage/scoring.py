import numpy
from sklearn import metrics
from triage.db import Evaluation
from triage.utils import sort_predictions_and_labels
from sqlalchemy.orm import sessionmaker
import logging
import time


"""Metric definitions

Mostly just wrappers around sklearn.metrics functions, these functions
implement a generalized interface to metric calculations that can be stored
as a scalar in the database.

All functions should take four parameters:
predictions_proba (1d array-like) Prediction probabilities
predictions_binary (1d array-like) Binarized predictions
labels (1d array-like) Ground truth target values
parameters (dict) Any needed hyperparameters in the implementation

All functions should return: (float) the resulting score

Functions defined here are meant to be used in ModelScorer.available_metrics
"""


def precision(_, predictions_binary, labels, parameters):
    return metrics.precision_score(labels, predictions_binary, **parameters)


def recall(_, predictions_binary, labels, parameters):
    return metrics.recall_score(labels, predictions_binary, **parameters)


def fbeta(_, predictions_binary, labels, parameters):
    return metrics.fbeta_score(labels, predictions_binary, **parameters)


def f1(_, predictions_binary, labels, parameters):
    return metrics.f1_score(labels, predictions_binary, **parameters)


def accuracy(_, predictions_binary, labels, parameters):
    return metrics.accuracy_score(labels, predictions_binary, **parameters)


def roc_auc(predictions_proba, _, labels, parameters):
    return metrics.roc_auc_score(labels, predictions_proba)


def avg_precision(predictions_proba, _, labels, parameters):
    return metrics.average_precision_score(labels, predictions_proba)


def true_positives(_, predictions_binary, labels, parameters):
     tp = [1 if x == 1 and y == 1 else 0 
             for (x, y) in zip(predictions_binary, labels)]
     return int(numpy.sum(tp))


def false_positives(_, predictions_binary, labels, parameters):
    fp = [1 if x == 1 and y == 0 else 0
            for (x, y) in zip(predictions_binary, labels)]
    return int(numpy.sum(fp))


def true_negatives(_, predictions_binary, labels, parameters):
    tn = [1 if x == 0 and y == 0 else 0
            for (x, y) in zip(predictions_binary, labels)]
    return int(numpy.sum(tn))


def false_negatives(_, predictions_binary, labels, parameters):
    fn = [1 if x == 0 and y == 1 else 0
            for (x, y) in zip(predictions_binary, labels)]
    return int(numpy.sum(fn))


def fpr(_, predictions_binary, labels, parameters):
    fp = false_positives(_, predictions_binary, labels, parameters)
    return float(fp / labels.count(0))


class UnknownMetricError(ValueError):
    """Signifies that a metric name was passed, but no matching computation
    function is available
    """
    pass


def generate_binary_at_x(test_predictions, x_value, unit='top_n'):
    """Generate subset of predictions based on top% or absolute

    Args:
        test_predictions (list) A list of predictions, sorted by risk desc
        test_labels (list) A list of labels, sorted by risk desc
        x_value (int) The percentile or absolute value desired
        unit (string, default 'top_n') The subsetting method desired,
            either percentile or top_n

    Returns: (list) The predictions subset
    """
    if unit == 'percentile':
        cutoff_index = int(len(test_predictions) * (x_value / 100.00))
    else:
        cutoff_index = x_value
    test_predictions_binary = [
        1 if x < cutoff_index else 0
        for x in range(len(test_predictions))
    ]
    return test_predictions_binary


class ModelScorer(object):
    """An object that can score models based on its known metrics"""

    """Available metric calculation functions

    Each value is expected to be a function that takes in the following params
    (predictions_proba, predictions_binary, labels, parameters)
    and return a numeric score
    """
    available_metrics = {
        'precision@': precision,
        'recall@': recall,
        'fbeta@': fbeta,
        'f1': f1,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'average precision score': avg_precision,
        'true positives@': true_positives,
        'true negatives@': true_negatives,
        'false positives@': false_positives,
        'false negatives@': false_negatives,
        'fpr@': fpr,
    }

    def __init__(self, metric_groups, db_engine, sort_seed=None, custom_metrics=None):
        """
        Args:
            metric_groups (list) A list of groups of metric/configurations
                to use for score all given models

                Each entry is a dict, with a list of metrics, and potentially
                    thresholds and parameter lists. Each metric is expected to
                    be a key in self.available_metrics

                Examples:

                metric_groups = [{
                    'metrics': ['precision@', 'recall@'],
                    'thresholds': {
                        'percentiles': [5.0, 10.0],
                        'top_n': [5, 10]
                    }
                }, {
                    'metrics': ['f1'],
                }, {
                    'metrics': ['fbeta@'],
                    'parameters': [{'beta': 0.75}, {'beta': 1.25}]
                }]

            db_engine (sqlalchemy.engine)
            custom_metrics (dict) Functions to generate metrics
                not available by default
                Each function is expected take in the following params:
                (predictions_proba, predictions_binary, labels, parameters)
                and return a numeric score
        """
        self.metric_groups = metric_groups
        self.db_engine = db_engine
        self.sort_seed = sort_seed or int(time.time())
        if custom_metrics:
            self.available_metrics.update(custom_metrics)
        if self.db_engine:
            self.sessionmaker = sessionmaker(bind=self.db_engine)

    def _generate_evaluations(
        self,
        metrics,
        parameters,
        threshold_config,
        predictions_proba,
        predictions_binary,
        labels,
    ):
        """Generate scores based on config and create ORM objects to hold them

        Args:
            metrics (list) names of metric to compute
            parameters (list) dicts holding parameters to pass to metrics
            threshold_config (dict) Unit type and value referring to how any
                thresholds were computed. Combined with parameter string
                to make a unique identifier for the parameter in the database
            predictions_proba (list) Probability predictions
            predictions_binary (list) Binary predictions
            labels (list) True labels

        Returns: (list) triage.db.Evaluation objects
        Raises: UnknownMetricError if a given metric is not present in
            self.available_metrics
        """
        evaluations = []
        num_labeled_examples = len(labels)
        num_labeled_above_threshold = predictions_binary.count(1)
        num_positive_labels = labels.count(1)
        for metric in metrics:
            if metric in self.available_metrics:
                for parameter_combination in parameters:
                    value = self.available_metrics[metric](
                        predictions_proba,
                        predictions_binary,
                        labels,
                        parameter_combination
                    )

                    full_params = parameter_combination.copy()
                    full_params.update(threshold_config)
                    parameter_string = '/'.join([
                        '{}_{}'.format(val, key)
                        for key, val in full_params.items()
                    ])
                    evaluations.append(Evaluation(
                        metric=metric,
                        parameter=parameter_string,
                        value=value,
                        num_labeled_examples=num_labeled_examples,
                        num_labeled_above_threshold=num_labeled_above_threshold,
                        num_positive_labels=num_positive_labels,
                        sort_seed=self.sort_seed
                    ))
            else:
                raise UnknownMetricError()
        return evaluations

    def score(
        self,
        predictions_proba,
        labels,
        model_id,
        evaluation_start_time,
        evaluation_end_time,
        example_frequency
    ):
        """Score a model based on predictions, and save the results

        Args:
            predictions_proba (numpy.array) List of prediction probabilities
            labels (numpy.array) The true labels for the prediction set
            model_id (int) The database identifier of the model
            evaluation_start_time (datetime.datetime) The time of the first prediction being evaluated
            evaluation_end_time (datetime.datetime) The time of the last prediction being evaluated
            example_frequency (string) How frequently predictions were generated
        """
        predictions_proba_sorted, labels_sorted = sort_predictions_and_labels(
            predictions_proba,
            labels,
            self.sort_seed
        )
        labels_sorted = numpy.array(labels_sorted)

        evaluations = []
        for group in self.metric_groups:
            parameters = group.get('parameters', [{}])
            if 'thresholds' not in group:
                evaluations = evaluations + self._generate_evaluations(
                    group['metrics'],
                    parameters,
                    {},
                    predictions_proba,
                    generate_binary_at_x(
                        predictions_proba_sorted,
                        100,
                        unit='percentile'
                    ),
                    labels_sorted.tolist(),
                )

            logging.info('thresholding predictions for group')
            for pct_thresh in group.get('thresholds', {}).get('percentiles', []):
                predicted_classes = numpy.array(generate_binary_at_x(
                    predictions_proba_sorted,
                    pct_thresh,
                    unit='percentile'
                ))
                nan_mask = numpy.isfinite(labels_sorted)
                predicted_classes = (predicted_classes[nan_mask]).tolist()
                present_labels_sorted = (labels_sorted[nan_mask]).tolist()
                evaluations = evaluations + self._generate_evaluations(
                    group['metrics'],
                    parameters,
                    {'pct': pct_thresh},
                    None,
                    predicted_classes,
                    present_labels_sorted,
                )

            for abs_thresh in group.get('thresholds', {}).get('top_n', []):
                predicted_classes = numpy.array(generate_binary_at_x(
                    predictions_proba_sorted,
                    abs_thresh,
                    unit='top_n'
                ))
                nan_mask = numpy.isfinite(labels_sorted)
                predicted_classes = (predicted_classes[nan_mask]).tolist()
                present_labels_sorted = (labels_sorted[nan_mask]).tolist()
                evaluations = evaluations + self._generate_evaluations(
                    group['metrics'],
                    parameters,
                    {'abs': abs_thresh},
                    None,
                    predicted_classes,
                    present_labels_sorted,
                )

        logging.info('Writing metrics to db')
        self._write_to_db(
            model_id,
            evaluation_start_time,
            evaluation_end_time,
            example_frequency,
            evaluations
        )
        logging.info('Done writing metrics to db')

    def _write_to_db(
        self,
        model_id,
        evaluation_start_time,
        evaluation_end_time,
        example_frequency,
        evaluations
    ):
        """Write evaluation objects to the database

        Binds the model_id as as_of_date to the given ORM objects
        and writes them to the database

        Args:
            model_id (int) primary key of the model
            as_of_date (datetime.date) Date the predictions were made as of
            evaluations (list) triage.db.Evaluation objects
        """
        session = self.sessionmaker()
        session.query(Evaluation)\
            .filter_by(
                model_id=model_id,
                evaluation_start_time=evaluation_start_time,
                evaluation_end_time=evaluation_end_time,
                example_frequency=example_frequency
            ).delete()

        for evaluation in evaluations:
            evaluation.model_id = model_id
            evaluation.evaluation_start_time = evaluation_start_time
            evaluation.evaluation_end_time = evaluation_end_time
            evaluation.example_frequency = example_frequency
            session.add(evaluation)
        session.commit()
        session.close()
