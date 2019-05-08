import logging
import numpy
import random


AVAILABLE_TIEBREAKERS = {'random', 'best', 'worst'}

def sort_predictions_and_labels(predictions_proba, labels, tiebreaker='random', sort_seed=None, parallel_arrays=()):
    """Sort predictions and labels with a configured tiebreaking rule

    Args:
        predictions_proba (numpy.array) The predicted scores
        labels (numpy.array) The numeric labels (1/0, not True/False)
        tiebreaker (string) The tiebreaking method ('best', 'worst', 'random')
        sort_seed (signed int) The sort seed. Needed if 'random' tiebreaking is picked.
        parallel_arrays (tuple of numpy.array) Any other arrays, understood to be the same size
            as the predictions and labels, that should be sorted alongside them.

    Returns:
        (tuple) (predictions_proba, labels), sorted
    """
    if len(labels) == 0:
        logging.debug("No labels present, skipping sorting.")
        if parallel_arrays:
            return (predictions_proba, labels, parallel_arrays)
        else:
            return (predictions_proba, labels)
    mask = None
    if tiebreaker == 'random':
        if not sort_seed:
            raise ValueError("If random tiebreaker is used, a sort seed must be given")
        random.seed(sort_seed)
        numpy.random.seed(sort_seed)
        random_arr = numpy.random.rand(*predictions_proba.shape)
        mask = numpy.lexsort((random_arr, predictions_proba))
    elif tiebreaker == 'worst':
        mask = numpy.lexsort((-labels, predictions_proba))
    elif tiebreaker == 'best':
        mask = numpy.lexsort((labels, predictions_proba))
    else:
        raise ValueError("Unknown tiebreaker")

    return_value = [
        numpy.flip(predictions_proba[mask]),
        numpy.flip(labels[mask]),
    ]
    if parallel_arrays:
        return_value.append(tuple(numpy.flip(arr[mask]) for arr in parallel_arrays))
    return return_value
