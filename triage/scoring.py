from sklearn import metrics
from triage.db import Evaluation
from sqlalchemy.orm import sessionmaker


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


class UnknownMetricError(ValueError):
    """Signifies that a metric name was passed, but no matching computation
    function is available
    """
    pass


def generate_binary_at_x(test_predictions, x_value, unit='top_n'):
    """Generate subset of predictions based on top% or absolute

    Args:
        test_predictions (list) The whole list of predictions
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
        'f1': f1
    }

    def __init__(self, metric_groups, db_engine, custom_metrics=None):
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
                        value=value
                    ))
            else:
                raise UnknownMetricError()
        return evaluations

    def score(
        self,
        predictions_proba,
        predictions_binary,
        labels,
        model_id,
        as_of_date
    ):
        """Score a model based on predictions, and save the results

        Args:
            predictions_proba (numpy.array) List of prediction probabilities
            predictions_binary (numpy.array) List of binarized predictions
            labels (numpy.array) The true labels for the prediction set
            model_id (int) The database identifier of the model
            as_of_date (datetime.date) The date predictions were made as of
        """
        predictions_proba = predictions_proba.tolist()
        predictions_binary = predictions_binary.tolist()
        labels = labels.tolist()

        predictions_proba_sorted, labels_sorted = \
            zip(*sorted(zip(predictions_proba, labels), reverse=True))
        evaluations = []
        for group in self.metric_groups:
            parameters = group.get('parameters', [{}])
            if 'thresholds' not in group:
                evaluations = evaluations + self._generate_evaluations(
                    group['metrics'],
                    parameters,
                    {},
                    predictions_proba,
                    predictions_binary,
                    labels,
                )

            for pct_thresh in group.get('thresholds', {}).get('percentiles', []):
                binary_subset = generate_binary_at_x(
                    predictions_proba_sorted,
                    pct_thresh,
                    unit='pct'
                )
                evaluations = evaluations + self._generate_evaluations(
                    group['metrics'],
                    parameters,
                    {'pct': pct_thresh},
                    None,
                    binary_subset,
                    labels,
                )

            for abs_thresh in group.get('thresholds', {}).get('top_n', []):
                binary_subset = generate_binary_at_x(
                    predictions_proba_sorted,
                    abs_thresh,
                    unit='abs'
                )
                evaluations = evaluations + self._generate_evaluations(
                    group['metrics'],
                    parameters,
                    {'abs': abs_thresh},
                    None,
                    binary_subset,
                    labels,
                )

        self._write_to_db(model_id, as_of_date, evaluations)

    def _write_to_db(self, model_id, as_of_date, evaluations):
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
            .filter_by(model_id=model_id, as_of_date=as_of_date)\
            .delete()

        for evaluation in evaluations:
            evaluation.model_id = model_id
            evaluation.as_of_date = as_of_date
            session.add(evaluation)
        session.commit()
