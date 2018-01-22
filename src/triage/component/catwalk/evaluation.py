import functools
import logging
import time

import numpy
from sqlalchemy.orm import sessionmaker

from triage.component.results_schema import Evaluation

from . import metrics
from .utils import db_retry, sort_predictions_and_labels


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


class ModelEvaluator(object):
    """An object that can score models based on its known metrics"""

    """Available metric calculation functions

    Each value is expected to be a function that takes in the following params
    (predictions_proba, predictions_binary, labels, parameters)
    and return a numeric score
    """
    available_metrics = {
        'precision@': metrics.precision,
        'recall@': metrics.recall,
        'fbeta@': metrics.fbeta,
        'f1': metrics.f1,
        'accuracy': metrics.accuracy,
        'roc_auc': metrics.roc_auc,
        'average precision score': metrics.avg_precision,
        'true positives@': metrics.true_positives,
        'true negatives@': metrics.true_negatives,
        'false positives@': metrics.false_positives,
        'false negatives@': metrics.false_negatives,
        'fpr@': metrics.fpr,
    }

    def __init__(self, metric_groups, db_engine, sort_seed=None, custom_metrics=None):
        """
        Args:
            metric_groups (list) A list of groups of metric/configurations
                to use for evaluating all given models

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
            self._validate_metrics(custom_metrics)
            self.available_metrics.update(custom_metrics)
        if self.db_engine:
            self.sessionmaker = sessionmaker(bind=self.db_engine)

    def _validate_metrics(
        self,
        custom_metrics
    ):
        for name, met in custom_metrics.items():
            if not hasattr(met, 'greater_is_better'):
                raise ValueError("Custom metric {} missing greater_is_better "
                                 "attribute".format(name))
            elif met.greater_is_better not in (True, False):
                raise ValueError("For custom metric {} greater_is_better must be "
                                 "boolean True or False".format(name))

    def _build_parameter_string(
        self,
        threshold_unit,
        threshold_value,
        parameter_combination
    ):
        """Encode the metric parameters and threshold into a short, human-parseable string

        Examples are: '100_abs', '5_pct'

        Args:
            threshold_unit (string) the type of threshold, either 'percentile' or 'top_n'
            threshold_value (int) the numeric threshold,
            parameter_combination (dict) The non-threshold parameter keys and values used

        Returns: (string) A short, human-parseable string
        """
        full_params = parameter_combination.copy()
        if not (threshold_unit == 'percentile' and threshold_value == 100):
            short_threshold_unit = 'pct' if threshold_unit == 'percentile' else 'abs'
            full_params[short_threshold_unit] = threshold_value
        parameter_string = '/'.join([
            '{}_{}'.format(val, key)
            for key, val in full_params.items()
        ])
        return parameter_string

    def _filter_nan_labels(self, predicted_classes, labels):
        """Filter missing labels and their corresponding predictions

        Args:
            predicted_classes (list) Predicted binary classes, of same length as labels
            labels (list) Labels, maybe containing NaNs

        Returns: (tuple) Copies of the input lists, with NaN labels removed
        """
        labels = numpy.array(labels)
        predicted_classes = numpy.array(predicted_classes)
        nan_mask = numpy.isfinite(labels)
        return (
            (predicted_classes[nan_mask]).tolist(),
            (labels[nan_mask]).tolist()
        )

    def _evaluations_for_threshold(
        self,
        metrics,
        parameters,
        predictions_proba,
        labels,
        threshold_unit,
        threshold_value,
    ):
        """Generate evaluations for a given threshold in a metric group,
        and create ORM objects to hold them

        Args:
            metrics (list) names of metric to compute
            parameters (list) dicts holding parameters to pass to metrics
            threshold_unit (string) the type of threshold, either 'percentile' or 'top_n'
            threshold_value (int) the numeric threshold,
            predictions_proba (list) Probability predictions
            labels (list) True labels (may have NaNs)

        Returns: (list) results_schema.Evaluation objects
        Raises: UnknownMetricError if a given metric is not present in
            self.available_metrics
        """

        # using threshold configuration, convert probabilities to predicted classes
        predicted_classes = generate_binary_at_x(
            predictions_proba,
            threshold_value,
            unit=threshold_unit
        )
        # filter out null labels
        predicted_classes_with_labels, present_labels = self._filter_nan_labels(
            predicted_classes,
            labels,
        )
        num_labeled_examples = len(present_labels)
        num_labeled_above_threshold = predicted_classes_with_labels.count(1)
        num_positive_labels = present_labels.count(1)
        evaluations = []
        for metric in metrics:
            if metric not in self.available_metrics:
                raise metrics.UnknownMetricError()

            for parameter_combination in parameters:
                value = self.available_metrics[metric](
                    predictions_proba,
                    predicted_classes_with_labels,
                    present_labels,
                    parameter_combination
                )

                # convert the thresholds/parameters into something
                # more readable
                parameter_string = self._build_parameter_string(
                    threshold_unit=threshold_unit,
                    threshold_value=threshold_value,
                    parameter_combination=parameter_combination
                )

                logging.info(
                    'Evaluations for %s%s, labeled examples %s '
                    'above threshold %s, positive labels %s, value %s',
                    metric,
                    parameter_string,
                    num_labeled_examples,
                    num_labeled_above_threshold,
                    num_positive_labels,
                    value
                )
                evaluations.append(Evaluation(
                    metric=metric,
                    parameter=parameter_string,
                    value=value,
                    num_labeled_examples=num_labeled_examples,
                    num_labeled_above_threshold=num_labeled_above_threshold,
                    num_positive_labels=num_positive_labels,
                    sort_seed=self.sort_seed
                ))
        return evaluations

    def _evaluations_for_group(
        self,
        group,
        predictions_proba_sorted,
        labels_sorted
    ):
        """Generate evaluations for a given metric group, and create ORM objects to hold them

        Args:
            group (dict) A configuration dictionary for the group.
                Should contain the key 'metrics', and optionally 'parameters' or 'thresholds'
            predictions_proba (list) Probability predictions
            labels (list) True labels (may have NaNs)

        Returns: (list) results_schema.Evaluation objects
        """
        logging.info('Creating evaluations for metric group %s', group)
        parameters = group.get('parameters', [{}])
        generate_evaluations = functools.partial(
            self._evaluations_for_threshold,
            metrics=group['metrics'],
            parameters=parameters,
            predictions_proba=predictions_proba_sorted,
            labels=labels_sorted
        )
        evaluations = []
        if 'thresholds' not in group:
            logging.info('Not a thresholded group, generating evaluation based on all predictions')
            evaluations = evaluations + generate_evaluations(
                threshold_unit='percentile',
                threshold_value=100
            )

        for pct_thresh in group.get('thresholds', {}).get('percentiles', []):
            logging.info('Processing percent threshold %s', pct_thresh)
            evaluations = evaluations + generate_evaluations(
                threshold_unit='percentile',
                threshold_value=pct_thresh
            )

        for abs_thresh in group.get('thresholds', {}).get('top_n', []):
            logging.info('Processing absolute threshold %s', abs_thresh)
            evaluations = evaluations + generate_evaluations(
                threshold_unit='top_n',
                threshold_value=abs_thresh
            )
        return evaluations

    def evaluate(
        self,
        predictions_proba,
        labels,
        model_id,
        evaluation_start_time,
        evaluation_end_time,
        as_of_date_frequency
    ):
        """Evaluate a model based on predictions, and save the results

        Args:
            predictions_proba (numpy.array) List of prediction probabilities
            labels (numpy.array) The true labels for the prediction set
            model_id (int) The database identifier of the model
            evaluation_start_time (datetime.datetime) The time of the first prediction
                being evaluated
            evaluation_end_time (datetime.datetime) The time of the last prediction being evaluated
            as_of_date_frequency (string) How frequently predictions were generated
        """
        logging.info(
            'Generating evaluations for model id %s, evaluation range %s-%s, '
            'as_of_date frequency %s',
            model_id,
            evaluation_start_time,
            evaluation_end_time,
            as_of_date_frequency
        )
        predictions_proba_sorted, labels_sorted = sort_predictions_and_labels(
            predictions_proba,
            labels,
            self.sort_seed
        )

        evaluations = []
        for group in self.metric_groups:
            evaluations += self._evaluations_for_group(
                group,
                predictions_proba_sorted,
                labels_sorted
            )

        logging.info('Writing metrics to db')
        self._write_to_db(
            model_id,
            evaluation_start_time,
            evaluation_end_time,
            as_of_date_frequency,
            evaluations
        )
        logging.info('Done writing metrics to db')

    @db_retry
    def _write_to_db(
        self,
        model_id,
        evaluation_start_time,
        evaluation_end_time,
        as_of_date_frequency,
        evaluations
    ):
        """Write evaluation objects to the database

        Binds the model_id as as_of_date to the given ORM objects
        and writes them to the database

        Args:
            model_id (int) primary key of the model
            as_of_date (datetime.date) Date the predictions were made as of
            evaluations (list) results_schema.Evaluation objects
        """
        session = self.sessionmaker()
        session.query(Evaluation)\
            .filter_by(
                model_id=model_id,
                evaluation_start_time=evaluation_start_time,
                evaluation_end_time=evaluation_end_time,
                as_of_date_frequency=as_of_date_frequency
            ).delete()

        for evaluation in evaluations:
            evaluation.model_id = model_id
            evaluation.evaluation_start_time = evaluation_start_time
            evaluation.evaluation_end_time = evaluation_end_time
            evaluation.as_of_date_frequency = as_of_date_frequency
            session.add(evaluation)
        session.commit()
        session.close()
