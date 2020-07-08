import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)
import operator

from triage.component.catwalk.evaluation import ModelEvaluator


def greater_is_better(metric):
    """Whether or not a metric wants higher values

    Args:
        metric (str): The name of a metric, ie 'precision@'
    Returns: (bool) Whether or not greater is better for the metric
    """
    if metric in ModelEvaluator.available_metrics:
        return ModelEvaluator.available_metrics[metric].greater_is_better
    else:
        logger.warning(
            "Metric %s not found in available metrics, assuming greater is better",
            metric,
        )
        return True


def sql_rank_order(metric):
    """SQL Rank Order for a metric

    Args:
        metric (str): The name of a metric, ie 'precision@'
    Returns: (str) A SQL ORDER BY clause that will rank the best values first
    """
    if greater_is_better(metric):
        return "desc"
    else:
        return "asc"


def is_better_operator(metric):
    """Operator to decide which of two values is better

    Args:
        metric (str): The name of a metric, ie 'precision@'
    Returns: (function) An operator function that will compare two values
        and return whether or not the first one is better
    """
    if greater_is_better(metric):
        return operator.ge
    else:
        return operator.le


def best_in_series(metric):
    """The best value in a series

    Args:
        metric (str): The name of a metric, ie 'precision@'
    Returns: (str) The name of a pandas Series function that will provide
        the best value
    """
    if greater_is_better(metric):
        return "max"
    else:
        return "min"


def idxbest(metric):
    """Index of first occurrence of the best value

    Args:
        metric (str): The name of a metric, ie 'precision@'
    Returns: (str) The name of a pandas function that will provide
        the index of the first occurrence of the best value
    """
    if greater_is_better(metric):
        return "idxmax"
    else:
        return "idxmin"
