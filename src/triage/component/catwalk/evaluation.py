import functools
import itertools
import verboselogs, logging

logger = verboselogs.VerboseLogger(__name__)
import math

import numpy as np
import ohio.ext.pandas
import pandas as pd
import statistics
import typing
from collections import defaultdict
from sqlalchemy.orm import sessionmaker

from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df

from . import metrics
from .utils import (
    db_retry,
    sort_predictions_and_labels,
    get_subset_table_name,
    filename_friendly_hash,
)
from triage.util.db import scoped_session
from triage.util.random import generate_python_random_seed
from triage.component.catwalk.storage import MatrixStore

RELATIVE_TOLERANCE = 0.01
SORT_TRIALS = 30


def subset_labels_and_predictions(
    subset_df,
    labels,
    predictions_proba,
    protected_df=None,
):
    """Reduce the labels and predictions to only those relevant to the current
       subset.

    Args:
        subset_df (pandas.DataFrame) A dataframe whose index is the entity_ids
            and as_of_dates in a subset
        labels (pandas.Series) A series of labels with entity_id and as_of_date
            as the index
        predictions_proba (np.array) An array of predictions for the same
            entity_date pairs as the labels and in the same order
        protected_df (pandas.DataFrame) A dataframe of protected group attributes

    Returns: (pandas.Series, np.array, pandas.DataFrame) The labels, predictions, and protected
        group attributes that refer to entity-date pairs in the subset
    """
    indexed_predictions = pd.Series(predictions_proba, index=labels.index)
    if protected_df is None:
        protected_df = pd.DataFrame()

    # The subset isn't specific to the cohort, so inner join to the labels/predictions
    labels_subset = labels.align(subset_df, join="inner")[0]
    predictions_subset = indexed_predictions.align(subset_df, join="inner")[0].values
    protected_df_subset = (
        protected_df
        if protected_df.empty
        else protected_df.align(subset_df, join="inner")[0]
    )
    logger.spam(
        f"{len(labels_subset)} entities in subset out of {len(labels)} in matrix.",
    )

    return (labels_subset, predictions_subset, protected_df_subset)


def query_subset_table(db_engine, as_of_dates, subset_table_name):
    """Queries the subset table to find the entities active at the given
       as_of_dates

    Args:
        db_engine (sqlalchemy.engine) a database engine
        as_of_dates (list) the as_of_Dates to query
        subset_table_name (str) the name of the table to query

    Returns: (pandas.DataFrame) a dataframe indexed by the entity-date pairs
        active in the subset
    """
    as_of_dates_sql = "[{}]".format(
        ", ".join(
            "'{}'".format(date.strftime("%Y-%m-%d %H:%M:%S.%f")) for date in as_of_dates
        )
    )
    query_string = f"""
        with dates as (
            select unnest(array{as_of_dates_sql}::timestamp[]) as as_of_date
        )
        select entity_id, as_of_date, active
        from {subset_table_name}
        join dates using(as_of_date)
    """
    df = pd.DataFrame.pg_copy_from(
        query_string,
        connectable=db_engine,
        parse_dates=["as_of_date"],
        index_col=MatrixStore.indices,
    )
    return df


def generate_binary_at_x(test_predictions, x_value, unit="top_n"):
    """Assign predicted classes based based on top% or absolute rank of score

    Args:
        test_predictions (np.array) A predictions, sorted by risk score descending
        x_value (int) The percentile or absolute value desired
        unit (string, default 'top_n') The thresholding method desired,
            either percentile or top_n

    Returns: (np.array) The predicted classes
    """
    len_predictions = len(test_predictions)
    if len_predictions == 0:
        return np.array([])
    if unit == "percentile":
        cutoff_index = int(len_predictions * (x_value / 100.00))
    else:
        cutoff_index = int(x_value)
    num_ones = cutoff_index if cutoff_index <= len_predictions else len_predictions
    num_zeroes = (
        len_predictions - cutoff_index if cutoff_index <= len_predictions else 0
    )
    test_predictions_binary = np.concatenate(
        (np.ones(num_ones, np.int8), np.zeros(num_zeroes, np.int8))
    )
    return test_predictions_binary


class MetricDefinition(typing.NamedTuple):
    """A single metric, bound to a particular threshold and parameter combination"""

    metric: str
    threshold_unit: str
    threshold_value: int
    parameter_combination: dict
    parameter_string: str


class MetricEvaluationResult(typing.NamedTuple):
    """A metric and parameter combination alongside preliminary results.

    The 'value' could represent the worst, best, or a random version of tiebreaking.
    """

    metric: str
    parameter: str
    value: float
    num_labeled_examples: int
    num_labeled_above_threshold: int
    num_positive_labels: int


class ModelEvaluator:
    """An object that can score models based on its known metrics"""

    # Available metric calculation functions

    # Each value is expected to be a function that takes in the following params
    # (predictions_proba, predictions_binary, labels, parameters)
    # and return a numeric score
    available_metrics = {
        "precision@": metrics.precision,
        "recall@": metrics.recall,
        "fbeta@": metrics.fbeta,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "roc_auc": metrics.roc_auc,
        "average precision score": metrics.avg_precision,
        "true positives@": metrics.true_positives,
        "true negatives@": metrics.true_negatives,
        "false positives@": metrics.false_positives,
        "false negatives@": metrics.false_negatives,
        "fpr@": metrics.fpr,
    }

    def __init__(
        self,
        testing_metric_groups,
        training_metric_groups,
        db_engine,
        custom_metrics=None,
        bias_config=None,
    ):
        """
        Args:
            testing_metric_groups (list) A list of groups of metric/configurations
                to use for evaluating all given models

                Each entry is a dict, with a list of metrics, and potentially
                    thresholds and parameter lists. Each metric is expected to
                    be a key in self.available_metrics

                Examples:

                testing_metric_groups = [{
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
            training_metric_groups (list) metrics to be calculated on training set,
                in the same form as testing_metric_groups
            db_engine (sqlalchemy.engine)
            custom_metrics (dict) Functions to generate metrics
                not available by default
                Each function is expected take in the following params:
                (predictions_proba, predictions_binary, labels, parameters)
                and return a numeric score
        """
        self.testing_metric_groups = testing_metric_groups
        self.training_metric_groups = training_metric_groups
        self.db_engine = db_engine
        self.bias_config = bias_config
        if custom_metrics:
            self._validate_metrics(custom_metrics)
            self.available_metrics.update(custom_metrics)

    @property
    def sessionmaker(self):
        return sessionmaker(bind=self.db_engine)

    def _validate_metrics(self, custom_metrics):
        for name, met in custom_metrics.items():
            if not hasattr(met, "greater_is_better"):
                raise ValueError(
                    f"Custom metric {name} missing greater_is_better " f"attribute"
                )
            elif met.greater_is_better not in (True, False):
                raise ValueError(
                    "For custom metric {name} greater_is_better must be "
                    "boolean True or False"
                )

    def _build_parameter_string(
        self,
        threshold_unit,
        threshold_value,
        parameter_combination,
        threshold_specified_by_user,
    ):
        """Encode the metric parameters and threshold into a short, human-parseable string

        Examples are: '100_abs', '5_pct'

        Args:
            threshold_unit (string) the type of threshold, either 'percentile' or 'top_n'
            threshold_value (int) the numeric threshold,
            parameter_combination (dict) The non-threshold parameter keys and values used
                Usually this will be empty, but an example would be {'beta': 0.25}

        Returns: (string) A short, human-parseable string
        """
        full_params = parameter_combination.copy()
        if threshold_specified_by_user:
            short_threshold_unit = "pct" if threshold_unit == "percentile" else "abs"
            full_params[short_threshold_unit] = threshold_value
        parameter_string = "/".join(
            ["{}_{}".format(val, key) for key, val in full_params.items()]
        )
        return parameter_string

    def _filter_nan_labels(self, predicted_classes: np.array, labels: np.array):
        """Filter missing labels and their corresponding predictions

        Args:
            predicted_classes (list) Predicted binary classes, of same length as labels
            labels (list) Labels, maybe containing NaNs

        Returns: (tuple) Copies of the input lists, with NaN labels removed
        """
        nan_mask = np.isfinite(labels)
        return (predicted_classes[nan_mask], labels[nan_mask])

    def _flatten_metric_threshold(
        self,
        metrics,
        parameters,
        threshold_unit,
        threshold_value,
        threshold_specified_by_user=True,
    ):
        """Flatten lists of metrics and parameters for an individual threshold
        into individual metric definitions.

        Args:
            metrics (list) names of metric to compute
            parameters (list) dicts holding parameters to pass to metrics
            threshold_unit (string) the type of threshold, either 'percentile' or 'top_n'
            threshold_value (int) the numeric threshold,
            threshold_specified_by_user (bool) Whether or not there was any threshold
                specified by the user. Defaults to True

        Returns: (list) MetricDefinition objects
        Raises: UnknownMetricError if a given metric is not present in
            self.available_metrics
        """

        metric_definitions = []
        for metric in metrics:
            if metric not in self.available_metrics:
                raise metrics.UnknownMetricError()

            for parameter_combination in parameters:
                # convert the thresholds/parameters into something
                # more readable
                parameter_string = self._build_parameter_string(
                    threshold_unit=threshold_unit,
                    threshold_value=threshold_value,
                    parameter_combination=parameter_combination,
                    threshold_specified_by_user=threshold_specified_by_user,
                )

                result = MetricDefinition(
                    metric=metric,
                    parameter_string=parameter_string,
                    parameter_combination=parameter_combination,
                    threshold_unit=threshold_unit,
                    threshold_value=threshold_value,
                )
                metric_definitions.append(result)
        return metric_definitions

    def _flatten_metric_config_group(self, group):
        """Flatten lists of metrics, parameters, and thresholds into individual metric definitions

        Args:
            group (dict) A configuration dictionary for the group.
                Should contain the key 'metrics', and optionally 'parameters' or 'thresholds'
        Returns: (list) MetricDefinition objects
        """
        logger.debug(f"Creating evaluations for metric group {group}")
        parameters = group.get("parameters", [{}])
        generate_metrics = functools.partial(
            self._flatten_metric_threshold,
            metrics=group["metrics"],
            parameters=parameters,
        )
        metrics = []
        if "thresholds" not in group:
            logger.notice(
                "Not a thresholded group, generating evaluation based on all predictions"
            )
            metrics = metrics + generate_metrics(
                threshold_unit="percentile",
                threshold_value=100,
                threshold_specified_by_user=False,
            )

        for pct_thresh in group.get("thresholds", {}).get("percentiles", []):
            logger.debug(f"Processing percent threshold {pct_thresh}")
            metrics = metrics + generate_metrics(
                threshold_unit="percentile", threshold_value=pct_thresh
            )

        for abs_thresh in group.get("thresholds", {}).get("top_n", []):
            logger.debug(f"Processing absolute threshold {abs_thresh}")
            metrics = metrics + generate_metrics(
                threshold_unit="top_n", threshold_value=abs_thresh
            )
        return metrics

    def _flatten_metric_config_groups(self, metric_config_groups):
        """Flatten lists of metrics, parameters, and thresholds into individual metric definitions

        Args:
            metric_config_groups (list) A list of metric group configuration dictionaries
                Each dict should contain the key 'metrics', and optionally 'parameters' or 'thresholds'
        Returns:
            (list) MetricDefinition objects
        """
        return [
            item
            for group in metric_config_groups
            for item in self._flatten_metric_config_group(group)
        ]

    def metric_definitions_from_matrix_type(self, matrix_type):
        """Retrieve the correct metric config groups for the matrix type and flatten them into metric definitions

        Args:
            matrix_type (catwalk.storage.MatrixType) A matrix type definition

        Returns:
            (list) MetricDefinition objects
        """
        if matrix_type.is_test:
            return self._flatten_metric_config_groups(self.testing_metric_groups)
        else:
            return self._flatten_metric_config_groups(self.training_metric_groups)

    def needs_evaluations(self, matrix_store, model_id, subset_hash=""):
        """Returns whether or not all the configured metrics are present in the
        database for the given matrix and model.
        Args:
            matrix_store (triage.component.catwalk.storage.MatrixStore)
            model_id (int) A model id
            subset_hash (str) An identifier for the subset to be evaluated

        Returns:
            (bool) whether or not this matrix and model are missing any evaluations in the db
        """

        # assemble a list of evaluation objects from the config
        # by running the evaluation code with an empty list of predictions and labels
        eval_obj = matrix_store.matrix_type.evaluation_obj
        matrix_type = matrix_store.matrix_type
        metric_definitions = self.metric_definitions_from_matrix_type(matrix_type)

        # assemble a list of evaluation objects from the database
        # by querying the unique metrics and parameters relevant to the passed-in matrix
        session = self.sessionmaker()

        evaluation_objects_in_db = (
            session.query(eval_obj)
            .filter_by(
                model_id=model_id,
                evaluation_start_time=matrix_store.as_of_dates[0],
                evaluation_end_time=matrix_store.as_of_dates[-1],
                as_of_date_frequency=matrix_store.metadata["as_of_date_frequency"],
                subset_hash=subset_hash,
            )
            .distinct(eval_obj.metric, eval_obj.parameter)
            .all()
        )

        # The list of needed metrics and parameters are all the unique metric/params from the config
        # not present in the unique metric/params from the db

        evals_needed = bool(
            {(met.metric, met.parameter_string) for met in metric_definitions}
            - {(obj.metric, obj.parameter) for obj in evaluation_objects_in_db}
        )
        session.close()
        if evals_needed:
            logger.notice(
                f"Needed evaluations for model {model_id} on matrix {matrix_store.uuid} are missing"
            )
            return True

        # now check bias config if there
        # if no bias config, no aequitas audits needed, so just return False at this point
        if not self.bias_config:
            logger.notice(f"No aequitas audit configured, so no evaluation needed")
            return False

        # if we do have bias config, return True. Too complicated with aequitas' visibility
        # at present to check whether all the needed records are needed.
        return True

    def _compute_evaluations(self, predictions_proba, labels, metric_definitions):
        """Compute evaluations for a set of predictions and labels

        Args:
            predictions_proba (np.array) predictions, sorted by score descending
            labels (np.array) labels, sorted however the caller wishes to break ties
            metric_definitions (list of MetricDefinition objects) metrics to compute

        Returns: (list of MetricEvaluationResult objects) One result for each metric definition
        """
        evals = []
        for (
            (threshold_unit, threshold_value),
            metrics_for_threshold,
        ) in itertools.groupby(
            metric_definitions, lambda m: (m.threshold_unit, m.threshold_value)
        ):
            predicted_classes = generate_binary_at_x(
                predictions_proba, threshold_value, unit=threshold_unit
            )
            # filter out null labels
            predicted_classes_with_labels, present_labels = self._filter_nan_labels(
                predicted_classes, labels
            )
            num_labeled_examples = len(present_labels)
            num_labeled_above_threshold = np.count_nonzero(
                predicted_classes_with_labels
            )
            num_positive_labels = np.count_nonzero(present_labels)
            for metric_def in metrics_for_threshold:
                # using threshold configuration, convert probabilities to predicted classes
                if len(predictions_proba) == 0:
                    logger.warning(
                        f"{metric_def.metric} not defined for parameter {metric_def.parameter_combination} because no entities "
                        "are in the subset for this matrix. Inserting NULL for value."
                    )
                    value = None
                else:
                    try:
                        value = self.available_metrics[metric_def.metric](
                            predictions_proba,
                            predicted_classes_with_labels,
                            present_labels,
                            metric_def.parameter_combination,
                        )

                    except ValueError:
                        logger.warning(
                            f"{metric_def.metric} not defined for parameter {metric_def.parameter_combination} because all labels "
                            "are the same. Inserting NULL for value."
                        )
                        value = None

                result = MetricEvaluationResult(
                    metric=metric_def.metric,
                    parameter=metric_def.parameter_string,
                    value=value,
                    num_labeled_examples=num_labeled_examples,
                    num_labeled_above_threshold=num_labeled_above_threshold,
                    num_positive_labels=num_positive_labels,
                )
                evals.append(result)
        return evals

    def evaluate(
        self, predictions_proba, matrix_store, model_id, protected_df=None, subset=None
    ):
        """Evaluate a model based on predictions, and save the results

        Args:
            predictions_proba (np.array) List of prediction probabilities
            matrix_store (catwalk.storage.MatrixStore) a wrapper for the
                prediction matrix and metadata
            model_id (int) The database identifier of the model
            subset (dict) A dictionary containing a query and a
                name for the subset to evaluate on, if any
            protected_df (pandas.DataFrame) A dataframe with protected group attributes
        """
        # If we are evaluating on a subset, we want to get just the labels and
        # predictions for the included entity-date pairs
        if subset:
            logger.verbose(
                f"Subsetting labels and predictions of model {model_id} on matrix {matrix_store.uuid}"
            )
            labels, predictions_proba, protected_df = subset_labels_and_predictions(
                subset_df=query_subset_table(
                    self.db_engine,
                    matrix_store.as_of_dates,
                    get_subset_table_name(subset),
                ),
                predictions_proba=predictions_proba,
                labels=matrix_store.labels,
                protected_df=protected_df,
            )
            subset_hash = filename_friendly_hash(subset)
        else:
            logger.debug(
                f"Using all the predictions of model {model_id} on matrix {matrix_store.uuid} for evaluation (i.e. no subset)"
            )
            labels = matrix_store.labels
            subset_hash = ""

        # confirm protected_df and labels have same set and count of values
        if (protected_df is not None) and (not protected_df.empty):
            if (protected_df.index.shape != labels.index.shape) or (
                not protected_df.index.symmetric_difference(labels.index).empty
            ):
                raise ValueError("Mismatch between protected_df and labels indices")

        df_index = labels.index
        labels = np.array(labels)

        matrix_type = matrix_store.matrix_type
        metric_defs = self.metric_definitions_from_matrix_type(matrix_type)

        logger.spam(f"Found {len(metric_defs)} metric definitions total")

        # 1. get worst sorting
        (
            predictions_proba_worst,
            labels_worst,
            df_index_worst,
        ) = sort_predictions_and_labels(
            predictions_proba=predictions_proba,
            labels=labels,
            df_index=df_index,
            tiebreaker="worst",
        )
        worst_lookup = {
            (eval.metric, eval.parameter): eval
            for eval in self._compute_evaluations(
                predictions_proba_worst, labels_worst, metric_defs
            )
        }
        logger.debug(
            f"Predictions from {model_id} sorted by worst case scenario, i.e. all negative and NULL labels first"
        )

        # 2. get best sorting
        (
            predictions_proba_best,
            labels_best,
            df_index_best,
        ) = sort_predictions_and_labels(
            predictions_proba=predictions_proba_worst,
            labels=labels_worst,
            df_index=df_index_worst,
            tiebreaker="best",
        )
        best_lookup = {
            (eval.metric, eval.parameter): eval
            for eval in self._compute_evaluations(
                predictions_proba_best, labels_best, metric_defs
            )
        }
        logger.debug(
            f"Predictions from {model_id} sorted by best case scenario, i.e. all positive labels first, NULL labels at the end"
        )

        evals_without_trials = dict()

        # 3. figure out which metrics have too far of a distance between best and worst
        # and need random trials
        metric_defs_to_trial = []
        for metric_def in metric_defs:
            worst_eval = worst_lookup[(metric_def.metric, metric_def.parameter_string)]
            best_eval = best_lookup[(metric_def.metric, metric_def.parameter_string)]
            if (
                worst_eval.value is None
                or best_eval.value is None
                or math.isclose(
                    worst_eval.value, best_eval.value, rel_tol=RELATIVE_TOLERANCE
                )
            ):
                evals_without_trials[
                    (worst_eval.metric, worst_eval.parameter)
                ] = worst_eval.value
            else:
                metric_defs_to_trial.append(metric_def)

        # 4. get average of n random trials
        logger.debug(
            f"For model {model_id}, {len(metric_defs_to_trial)} metric definitions need {SORT_TRIALS} random trials each as best/worst evals were different"
        )

        random_eval_accumulator = defaultdict(list)
        for _ in range(0, SORT_TRIALS):
            sort_seed = generate_python_random_seed()
            (
                predictions_proba_random,
                labels_random,
                df_index_random,
            ) = sort_predictions_and_labels(
                predictions_proba=predictions_proba_worst,
                labels=labels_worst,
                df_index=df_index_worst,
                tiebreaker="random",
                sort_seed=sort_seed,
            )
            for random_eval in self._compute_evaluations(
                predictions_proba_random, labels_random, metric_defs_to_trial
            ):
                random_eval_accumulator[
                    (random_eval.metric, random_eval.parameter)
                ].append(random_eval.value)

        # 5. flatten best, worst, stochastic results for each metric definition
        # into database records
        evaluation_start_time = matrix_store.as_of_dates[0]
        evaluation_end_time = matrix_store.as_of_dates[-1]
        as_of_date_frequency = matrix_store.metadata["as_of_date_frequency"]
        matrix_uuid = matrix_store.uuid
        evaluations = []
        for metric_def in metric_defs:
            metric_key = (metric_def.metric, metric_def.parameter_string)
            if metric_key in evals_without_trials:
                stochastic_value = evals_without_trials[metric_key]
                standard_deviation = 0
                num_sort_trials = 0
            else:
                trial_results = [
                    value
                    for value in random_eval_accumulator[metric_key]
                    if value is not None
                ]
                stochastic_value = statistics.mean(trial_results)
                try:
                    standard_deviation = statistics.stdev(trial_results)
                except ValueError:
                    logger.warning(
                        f"{metric_def.metric} not defined for parameter {metric_def.parameter_combination} because all values "
                        "are NaN. Inserting NULL for value."
                    )
                    standard_deviation = None

                num_sort_trials = len(trial_results)

            evaluation = matrix_type.evaluation_obj(
                metric=metric_def.metric,
                parameter=metric_def.parameter_string,
                num_labeled_examples=worst_lookup[metric_key].num_labeled_examples,
                num_labeled_above_threshold=worst_lookup[
                    metric_key
                ].num_labeled_above_threshold,
                num_positive_labels=worst_lookup[metric_key].num_positive_labels,
                worst_value=worst_lookup[metric_key].value,
                best_value=best_lookup[metric_key].value,
                stochastic_value=stochastic_value,
                num_sort_trials=num_sort_trials,
                standard_deviation=standard_deviation,
            )
            evaluations.append(evaluation)

        self._write_to_db(
            model_id,
            subset_hash,
            evaluation_start_time,
            evaluation_end_time,
            as_of_date_frequency,
            matrix_store.uuid,
            evaluations,
            matrix_type.evaluation_obj,
        )
        if protected_df is not None:
            self._write_audit_to_db(
                model_id=model_id,
                protected_df=protected_df.reindex(df_index_worst),
                predictions_proba=predictions_proba_worst,
                labels=labels_worst,
                tie_breaker="worst",
                subset_hash=subset_hash,
                matrix_type=matrix_type,
                evaluation_start_time=evaluation_start_time,
                evaluation_end_time=evaluation_end_time,
                matrix_uuid=matrix_store.uuid,
            )
            self._write_audit_to_db(
                model_id=model_id,
                protected_df=protected_df.reindex(df_index_best),
                predictions_proba=predictions_proba_best,
                labels=labels_best,
                tie_breaker="best",
                subset_hash=subset_hash,
                matrix_type=matrix_type,
                evaluation_start_time=evaluation_start_time,
                evaluation_end_time=evaluation_end_time,
                matrix_uuid=matrix_store.uuid,
            )

    def _write_audit_to_db(
        self,
        model_id,
        protected_df,
        predictions_proba,
        labels,
        tie_breaker,
        subset_hash,
        matrix_type,
        evaluation_start_time,
        evaluation_end_time,
        matrix_uuid,
    ):
        """
        Runs the bias audit and saves the result in the bias table.

        Args:
            model_id (int) primary key of the model
            protected_df (pandas.DataFrame) A dataframe with protected group attributes:
            predictions_proba (np.array) List of prediction probabilities
            labels (pandas.Series): List of labels
            tie_breaker: 'best' or 'worst' case tiebreaking rule that the predictions and labels were sorted by
            subset_hash (str) the hash of the subset, if any, that the
                evaluation is made on
            matrix_type (triage.component.catwalk.storage.MatrixType)
                The type of matrix used
            evaluation_start_time (pandas._libs.tslibs.timestamps.Timestamp)
                first as_of_date included in the evaluation period
            evaluation_end_time (pandas._libs.tslibs.timestamps.Timestamp) last
                as_of_date included in the evaluation period
            matrix_uuid: the uuid of the matrix
        Returns:

        """
        if protected_df.empty:
            return

        # to preprocess aequitas requires the following columns:
        # score, label value, model_id, protected attributes
        # fill out the protected_df, which just has protected attributes at this point
        protected_df = protected_df.copy()
        protected_df["model_id"] = model_id
        protected_df["score"] = predictions_proba
        protected_df["label_value"] = labels
        aequitas_df, attr_cols_input = preprocess_input_df(protected_df)

        # create group crosstabs
        g = Group()
        score_thresholds = {}
        score_thresholds["rank_abs"] = self.bias_config["thresholds"].get("top_n", [])
        # convert 0-100 percentile to 0-1 that Aequitas expects
        score_thresholds["rank_pct"] = [
            value / 100.0
            for value in self.bias_config["thresholds"].get("percentiles", [])
        ]
        groups_model, attr_cols = g.get_crosstabs(
            aequitas_df, score_thresholds=score_thresholds, attr_cols=attr_cols_input
        )
        # analyze bias from reference groups
        bias = Bias()
        ref_groups_method = self.bias_config.get("ref_groups_method", None)
        if ref_groups_method == "predefined" and self.bias_config["ref_groups"]:
            bias_df = bias.get_disparity_predefined_groups(
                groups_model, aequitas_df, self.bias_config["ref_groups"]
            )
        elif ref_groups_method == "majority":
            bias_df = bias.get_disparity_major_group(groups_model, aequitas_df)
        else:
            bias_df = bias.get_disparity_min_metric(groups_model, aequitas_df)

        # analyze fairness for each group
        f = Fairness(tau=0.8)  # the default fairness threshold is 0.8
        group_value_df = f.get_group_value_fairness(bias_df)
        group_value_df["subset_hash"] = subset_hash
        group_value_df["tie_breaker"] = tie_breaker
        group_value_df["evaluation_start_time"] = evaluation_start_time
        group_value_df["evaluation_end_time"] = evaluation_end_time
        group_value_df["matrix_uuid"] = matrix_uuid
        group_value_df = group_value_df.rename(
            index=str, columns={"score_threshold": "parameter", "for": "for_"}
        )
        if group_value_df.empty:
            raise ValueError(
                f"""
            Bias audit: aequitas_audit() failed.
            Returned empty dataframe for model_id = {model_id}, and subset_hash = {subset_hash}
            and matrix_type = {matrix_type}"""
            )
        with scoped_session(self.db_engine) as session:
            for index, row in group_value_df.iterrows():
                session.query(matrix_type.aequitas_obj).filter_by(
                    model_id=row["model_id"],
                    evaluation_start_time=row["evaluation_start_time"],
                    evaluation_end_time=row["evaluation_end_time"],
                    subset_hash=row["subset_hash"],
                    parameter=row["parameter"],
                    tie_breaker=row["tie_breaker"],
                    matrix_uuid=row["matrix_uuid"],
                    attribute_name=row["attribute_name"],
                    attribute_value=row["attribute_value"],
                ).delete()
            session.bulk_insert_mappings(
                matrix_type.aequitas_obj, group_value_df.to_dict(orient="records")
            )

    @db_retry
    def _write_to_db(
        self,
        model_id,
        subset_hash,
        evaluation_start_time,
        evaluation_end_time,
        as_of_date_frequency,
        matrix_uuid,
        evaluations,
        evaluation_table_obj,
    ):
        """Write evaluation objects to the database
        Binds the model_id as as_of_date to the given ORM objects
        and writes them to the database
        Args:
            model_id (int) primary key of the model
            subset_hash (str) the hash of the subset, if any, that the
                evaluation is made on
            evaluation_start_time (pandas._libs.tslibs.timestamps.Timestamp)
                first as_of_date included in the evaluation period
            evaluation_end_time (pandas._libs.tslibs.timestamps.Timestamp) last
                as_of_date included in the evaluation period
            as_of_date_frequency (str) the frequency with which as_of_dates
                occur between the evaluation_start_time and evaluation_end_time
            evaluations (list) results_schema.TestEvaluation or TrainEvaluation
                objects
            evaluation_table_obj (schema.TestEvaluation or TrainEvaluation)
                specifies to which table to add the evaluations
        """
        with scoped_session(self.db_engine) as session:
            session.query(evaluation_table_obj).filter_by(
                model_id=model_id,
                evaluation_start_time=evaluation_start_time,
                evaluation_end_time=evaluation_end_time,
                as_of_date_frequency=as_of_date_frequency,
                subset_hash=subset_hash,
            ).delete()

            for evaluation in evaluations:
                evaluation.model_id = model_id
                evaluation.as_of_date_frequency = as_of_date_frequency
                evaluation.subset_hash = subset_hash
                evaluation.evaluation_start_time = evaluation_start_time
                evaluation.evaluation_end_time = evaluation_end_time
                evaluation.as_of_date_frequency = as_of_date_frequency
                evaluation.matrix_uuid = matrix_uuid
                evaluation.subset_hash = subset_hash
                session.add(evaluation)
