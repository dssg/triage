import numpy as np
import pandas as pd
import pandas.api.types as ptypes

import pytest
from unittest import TestCase
from numpy.testing import assert_array_equal

from triage.component.catwalk.baselines.rankers import PercentileRankOneFeature
from triage.component.catwalk.baselines.rankers import BaselineRankMultiFeature
from triage.component.catwalk.baselines.rankers import LinearRanker
from triage.component.catwalk.baselines.thresholders import SimpleThresholder
from triage.component.catwalk.baselines.thresholders import get_operator_method
from triage.component.catwalk.baselines.thresholders import OPERATOR_METHODS
from triage.component.catwalk.exceptions import BaselineFeatureNotInMatrix


@pytest.fixture(scope="class")
def data(request):
    X_train = pd.DataFrame( 
        {
            "x1": [0, 1, 2, 56, 25, 8, -3, 89],
            "x2": [0, 23, 1, 6, 5, 3, 18, 7],
            "x3": [1, 12, 5, -6, 2, 5, 3, -3],
            "x4": [6, 13, 4, 5, 35, 6, 43, 74],
            "x6": [400, 400, 400, 300, 300, 300, 300, 300],
            "x7": [1, 12, 12, -6, -6, 5, 5, -3],
        }
    )
    y_train = [0, 1, 0, 1, 1, 1, 3, 0]
    X_test = pd.DataFrame(
        {
            "x1": [4, 14, 0, 6, 25, 8, -3, 4],
            "x2": [6, -1, 1, 24, 5, 3, 18, 39],
            "x3": [1, 7, 4, 57, 2, 5, 3, 2],
            "x4": [7, 3, 6, 39, 35, 6, 43, -6],
            "x6": [305, 305, 305, 305, 401, 401, 401, 401],
            "x7": [1, 7, 7, 57, 2, 5, 3, 2],
        }
    )
    y_test = [1, 3, 0, 0, 0, 0, 0, 1]

    request.cls.data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

@pytest.fixture(scope="class")
def rules(request):
    request.cls.rules = ["x1 > 0", "x2 <= 1"]

@pytest.fixture(scope="class")
def features(request):
    request.cls.features = ['x1', 'x2', 'x3', 'x4', 'x6', 'x7']

@pytest.fixture(scope="class")
def weights(request):
    request.cls.weights = [0.6, 0.5, 0.4, 0.7, 0.6, 0.3]


def predict_proba_deprecated(ranker, x):
        """ Generate the rank scores and return these.
        """
        # reduce x to the selected set of features
        x = x[ranker.all_feature_names].reset_index(drop=True)

        x = x.sort_values(ranker.all_feature_names, ascending=ranker.all_sort_directions)

        # initialize curr_rank to -1 so the first record will have rank 0 (hence "score"
        # will range from 0 to 1)
        ranks = []
        curr_rank = -1
        prev = []

        # calculate ranks over sorted records, giving ties the same rank
        for rec in x.values:
            if not np.array_equal(prev, rec):
                curr_rank += 1
            ranks.append(curr_rank)
            prev = rec

        # normalize to 0 to 1 range
        x['score'] = [r/max(ranks) for r in ranks]

        # reset back to original sort order, calculate "score" for "0 class"
        scores_1 = x.sort_index()['score'].values
        scores_0 = np.array([1-s for s in scores_1])

        return np.array([scores_0, scores_1]).transpose()

def scores_align_with_ranks(expected_ranks, returned_scores):
    '''
    Helper function to check that scores align with ranks
    correctly for the ranking baselines (e.g., higher ranks
    get higher scores and ties have the same score)
    '''
    df = pd.DataFrame({
        'rank': expected_ranks,
        'score': returned_scores
        }).sort_values('rank', ascending=True)

    curr_rank = None
    curr_score = None

    # Loop through the sorted records to check for any inconsistencies
    for ix, rec in df.iterrows():
        if curr_rank is None:
            curr_rank = rec['rank']
            curr_score = rec['score']
            continue

        if rec['rank'] < curr_rank:
            return RuntimeError('Something has gone wrong with df.sort_values!')
        elif rec['rank'] == curr_rank and rec['score'] != curr_score:
            return False
        elif rec['rank'] > curr_rank and rec['score'] <= curr_score:
            return False

        curr_rank = rec['rank']
        curr_score = rec['score']

    # If we got through the loop without any issues, return True
    return True


def test_scores_align_with_ranks():
    # correct, no ties
    assert scores_align_with_ranks([1,2,3], [0,0.5,1.0])
    # correct, with ties
    assert scores_align_with_ranks([1,2,2,3], [0,0.5,0.5,1.0])
    # incorrect, no ties
    assert not scores_align_with_ranks([1,2,3], [1.0,0.5,0.8])
    # ties with different scores
    assert not scores_align_with_ranks([1,2,2,3], [0,0.5,0.7,1.0])



@pytest.mark.usefixtures("data")
class TestRankOneFeature(TestCase):

    def test_fit(self):
        ranker = PercentileRankOneFeature(feature="x3")
        assert ranker.feature_importances_ is None
        ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
        np.testing.assert_array_equal(
            ranker.feature_importances_, np.array([0, 0, 1, 0, 0, 0])
        )

    def test_ranking_on_unavailable_feature_raises_error(self):
        ranker = PercentileRankOneFeature(feature="x5")
        with self.assertRaises(BaselineFeatureNotInMatrix):
            ranker.fit(x=self.data["X_train"], y=self.data["y_train"])

    def test_predict_proba(self):
        for direction_value in [True, False]:
            ranker = PercentileRankOneFeature(feature="x3", low_value_high_score=direction_value)
            ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
            results = ranker.predict_proba(self.data["X_test"])
            if direction_value:
                expected_ranks = [6, 1, 3, 0, 5, 2, 4, 5]
            else:
               expected_ranks = [0, 5, 3, 6, 1, 4, 2, 1]

            assert scores_align_with_ranks(expected_ranks, results[:,1])


@pytest.mark.usefixtures("data")
class TestRankMultiFeature(TestCase):
    def test_fit(self):
        rules = {'feature': 'x3', 'low_value_high_score': False}
        ranker = BaselineRankMultiFeature(rules=rules)
        assert ranker.feature_importances_ is None
        ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
        np.testing.assert_array_equal(
            ranker.feature_importances_, np.array([0, 0, 1, 0, 0, 0])
        )

    def test_ranking_on_unavailable_feature_raises_error(self):
        rules = [{'feature': 'x5', 'low_value_high_score': False}]
        ranker = BaselineRankMultiFeature(rules=rules)
        with self.assertRaises(BaselineFeatureNotInMatrix):
            ranker.fit(x=self.data["X_train"], y=self.data["y_train"])

    def test_predict_proba_one_feature(self):
        for direction_value in [True, False]:
            rules = {'feature': 'x3', 'low_value_high_score': direction_value}
            ranker = BaselineRankMultiFeature(rules=rules)
            ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
            results = ranker.predict_proba(self.data["X_test"])
            if direction_value:
                expected_ranks = [6, 1, 3, 0, 5, 2, 4, 5]
            else:
                expected_ranks = [0, 5, 3, 6, 1, 4, 2, 1]

            assert scores_align_with_ranks(expected_ranks, results[:,1])

    def test_predict_proba_multi_feature(self):
        rules = [
            {'feature': 'x3', 'low_value_high_score': True},
            {'feature': 'x2', 'low_value_high_score': False}
        ]

        ranker = BaselineRankMultiFeature(rules=rules)
        ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
        results = ranker.predict_proba(self.data["X_test"])

        expected_ranks = [7, 1, 3, 0, 5, 2, 4, 6]

        assert scores_align_with_ranks(expected_ranks, results[:,1])

    def test_predict_proba_no_ties(self):
        for direction_value in [True, False]:
            rules = [{'feature': 'x2', 'low_value_high_score': direction_value}]

            ranker = BaselineRankMultiFeature(rules=rules)
            ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
            results_new = ranker.predict_proba(self.data["X_test"])
            results_deprecated = predict_proba_deprecated(ranker, self.data["X_test"])
            
            assert_array_equal(results_new, results_deprecated)

    
    def test_predict_proba_half_ties(self):
        for direction_value in [True, False]: 
            rules = [{'feature': 'x6', 'low_value_high_score': direction_value}]

            ranker = BaselineRankMultiFeature(rules=rules)
            ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
            results_new = ranker.predict_proba(self.data["X_test"])
            results_deprecated = predict_proba_deprecated(ranker, self.data["X_test"])

            assert_array_equal(results_new, results_deprecated)


    def test_predict_proba_some_ties(self):
        for direction_value in [True, False]: 
            rules = [{'feature': 'x7', 'low_value_high_score': direction_value}]

            ranker = BaselineRankMultiFeature(rules=rules)
            ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
            results_new = ranker.predict_proba(self.data["X_test"])
            results_deprecated = predict_proba_deprecated(ranker, self.data["X_test"])

            assert_array_equal(results_new, results_deprecated)


@pytest.mark.parametrize('operator', OPERATOR_METHODS.keys())
def test_get_operator_method(operator):
    series = pd.Series([1, 2, 3, 4, 5])
    pd_operator = get_operator_method(operator)
    result = getattr(series, pd_operator)(series)
    assert ptypes.is_bool_dtype(result)


@pytest.mark.usefixtures("data", "rules")
class TestSimpleThresholder(TestCase):
    def test__convert_string_rule_to_dict(self):
        thresholder = SimpleThresholder(self.rules, "or")
        results = thresholder._convert_string_rule_to_dict(self.rules[0])
        expected_results = {"feature_name": "x1", "operator": "gt", "threshold": 0}
        assert results == expected_results

    def test_all_feature_names_property(self):
        thresholder = SimpleThresholder(self.rules, "or")
        assert thresholder.all_feature_names == ["x1", "x2"]

    def test_fit(self):
        thresholder = SimpleThresholder(self.rules, "or")
        assert thresholder.feature_importances_ is None
        thresholder.fit(x=self.data["X_train"], y=self.data["y_train"])
        np.testing.assert_array_equal(
            thresholder.feature_importances_, np.array([1, 1, 0, 0, 0, 0])
        )

    def test_rule_with_unavailable_feature_raises_error(self):
        rules = self.rules
        rules.append("x5 == 3")
        thresholder = SimpleThresholder(rules, "or")
        with self.assertRaises(BaselineFeatureNotInMatrix):
            thresholder.fit(x=self.data["X_train"], y=self.data["y_train"])

    def test_predict_proba(self):
        for logical_operator in ["and", "or"]:
            thresholder = SimpleThresholder(self.rules, logical_operator)
            thresholder.fit(x=self.data["X_train"], y=self.data["y_train"])
            results = thresholder.predict_proba(self.data["X_test"])
            if logical_operator == "and":
                expected_results = np.array(
                    [[0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]]
                ).transpose()
            elif logical_operator == "or":
                expected_results = np.array(
                    [[1, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 0, 1]]
                ).transpose()
            np.testing.assert_array_equal(results, expected_results)


@pytest.mark.usefixtures("data", "features", "weights")
class TestLinearRanker(TestCase):
    def test_fit(self):
        ranker = LinearRanker(
            self.features,
            self.weights
        )
        assert ranker.feature_importances_ is None
        ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
        np.testing.assert_almost_equal(
            ranker.feature_importances_, 
            np.array([0.19354839, 0.16129032, 
                      0.12903226, 0.22580645, 
                      0.19354839, 0.09677419]),
            decimal=4
        )

    def test_predict_proba(self):
        ranker = LinearRanker(self.features, 
                              self.weights)
        ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
        results = ranker.predict_proba(self.data["X_test"])
        
        assert results.shape == (8, 2)
        
        expected_results = np.array([[0.75411428, 0.24588572],
                                     [0.69980399, 0.30019601],
                                     [0.79451361, 0.20548639],
                                     [0.27090783, 0.72909217],
                                     [0.39821089, 0.60178911],
                                     [0.67697988, 0.32302012],
                                     [0.51544567, 0.48455433],
                                     [0.63648246, 0.36351754]])

        np.testing.assert_almost_equal(results, 
                                       expected_results, 
                                       decimal=4)
        
