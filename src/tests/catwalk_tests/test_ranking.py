import numpy
from numpy.testing import assert_array_equal
import pytest

from triage.component.catwalk.ranking import sort_predictions_and_labels


def test_sort_predictions_and_labels():
    predictions = numpy.array([0.5, 0.4, 0.6, 0.5])

    labels = numpy.array([0, 0, 1, 1])

    # best sort
    sorted_predictions, sorted_labels = sort_predictions_and_labels(
        predictions, labels, tiebreaker='best'
    )
    assert_array_equal(sorted_predictions, numpy.array([0.6, 0.5, 0.5, 0.4]))
    assert_array_equal(sorted_labels, numpy.array([1, 1, 0, 0]))

    # worst wort
    sorted_predictions, sorted_labels = sort_predictions_and_labels(
        predictions, labels, tiebreaker='worst'
    )
    assert_array_equal(sorted_predictions, numpy.array([0.6, 0.5, 0.5, 0.4]))
    assert_array_equal(sorted_labels, numpy.array([1, 0, 1, 0]))

    # random tiebreaker needs a seed
    with pytest.raises(ValueError):
        sort_predictions_and_labels(predictions, labels, tiebreaker='random')

    # random tiebreaker respects the seed
    sorted_predictions, sorted_labels = sort_predictions_and_labels(
        predictions,
        labels,
        tiebreaker='random',
        sort_seed=1234
    )
    assert_array_equal(sorted_predictions, numpy.array([0.6, 0.5, 0.5, 0.4]))
    assert_array_equal(sorted_labels, numpy.array([1, 1, 0, 0]))

    sorted_predictions, sorted_labels = sort_predictions_and_labels(
        predictions,
        labels,
        tiebreaker='random',
        sort_seed=24376234
    )
    assert_array_equal(sorted_predictions, numpy.array([0.6, 0.5, 0.5, 0.4]))
    assert_array_equal(sorted_labels, numpy.array([1, 0, 1, 0]))
