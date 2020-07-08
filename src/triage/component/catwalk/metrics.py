"""Metric definitions

Mostly just wrappers around sklearn.metrics functions, these functions
implement a generalized interface to metric calculations that can be stored
as a scalar in the database.

All functions should take four parameters:
predictions_proba (1d array-like) Prediction probabilities
predictions_binary (1d array-like) Binarized predictions
labels (1d array-like) Ground truth target values
parameters (dict) Any needed hyperparameters in the implementation

All functions should be wrapped with @Metric to define the optimal direction

All functions should return: (float) the resulting score

Functions defined here are meant to be used in ModelEvaluator.available_metrics

"""
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np


class Metric:
    """decorator for metrics: result will be a callable metric with an
    `greater_is_better` parameter defined as either True or False
    depending on whether larger or smaller metric values indicate
    better models.
    """

    def __init__(self, greater_is_better):
        if greater_is_better not in (True, False):
            raise ValueError("greater_is_better must be True or False")
        self.greater_is_better = greater_is_better

    def __call__(self, function, *params, **kwparams):
        class DecoratedMetric:
            def __init__(self, greater_is_better, function):
                self.greater_is_better = greater_is_better
                self.function = function
                self.__name__ = function.__name__
                self.__doc__ = function.__doc__

            def __call__(self, *params, **kwparams):
                return self.function(*params, **kwparams)

        return DecoratedMetric(self.greater_is_better, function)


@Metric(greater_is_better=True)
def precision(_, predictions_binary, labels, parameters):
    return metrics.precision_score(labels, predictions_binary, **parameters)


@Metric(greater_is_better=True)
def recall(_, predictions_binary, labels, parameters):
    return metrics.recall_score(labels, predictions_binary, **parameters)


@Metric(greater_is_better=True)
def fbeta(_, predictions_binary, labels, parameters):
    return metrics.fbeta_score(labels, predictions_binary, **parameters)


@Metric(greater_is_better=True)
def f1(_, predictions_binary, labels, parameters):
    return metrics.f1_score(labels, predictions_binary, **parameters)


@Metric(greater_is_better=True)
def accuracy(_, predictions_binary, labels, parameters):
    return metrics.accuracy_score(labels, predictions_binary, **parameters)


@Metric(greater_is_better=True)
def roc_auc(predictions_proba, _, labels, parameters):
    return metrics.roc_auc_score(labels, predictions_proba)


@Metric(greater_is_better=True)
def avg_precision(predictions_proba, _, labels, parameters):
    return metrics.average_precision_score(labels, predictions_proba)


@Metric(greater_is_better=True)
def true_positives(_, predictions_binary, labels, parameters):
    # If all labels false
    if not any(labels):
        return 0
    # If all labels true and all predictions 1
    elif all(labels) and all(i == 1 for i in predictions_binary):
        return 1
    else:
        return int(confusion_matrix(labels, predictions_binary)[1, 1])


@Metric(greater_is_better=False)
def false_positives(_, predictions_binary, labels, parameters):
    # If all labels false
    if not any(labels):
        return 0
    # If all labels true and all predictions 1
    elif all(labels) and all(i == 1 for i in predictions_binary):
        return 0
    else:
        return int(confusion_matrix(labels, predictions_binary)[0, 1])


@Metric(greater_is_better=True)
def true_negatives(_, predictions_binary, labels, parameters):
    # If all labels false and all predictions 0
    if not any(labels) and all(i == 0 for i in predictions_binary):
        return 1
    # If all labels true
    elif all(labels):
        return 0
    else:
        return int(confusion_matrix(labels, predictions_binary)[0, 0])


@Metric(greater_is_better=False)
def false_negatives(_, predictions_binary, labels, parameters):
    # If all labels false and all predictions 0
    if not any(labels) and all(i == 0 for i in predictions_binary):
        return 0
    # If all labels true
    elif all(labels):
        return 0
    else:
        return int(confusion_matrix(labels, predictions_binary)[1, 0])


@Metric(greater_is_better=False)
def fpr(_, predictions_binary, labels, parameters):
    fp = false_positives(_, predictions_binary, labels, parameters)
    return float(fp / (len(labels) - np.count_nonzero(labels)))


class UnknownMetricError(ValueError):
    """Signifies that a metric name was passed, but no matching computation
    function is available
    """
