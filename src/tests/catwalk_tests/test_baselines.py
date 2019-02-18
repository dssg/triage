import numpy as np
import pandas as pd
import pandas.api.types as ptypes

import pytest
from unittest import TestCase

from triage.component.catwalk.baselines.rankers import PercentileRankOneFeature
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
        }
    )
    y_train = [0, 1, 0, 1, 1, 1, 3, 0]
    X_test = pd.DataFrame(
        {
            "x1": [4, 14, 0, 6, 25, 8, -3, 4],
            "x2": [6, -1, 1, 24, 5, 3, 18, 39],
            "x3": [1, 7, 4, 57, 2, 5, 3, 2],
            "x4": [7, 3, 6, 39, 35, 6, 43, -6],
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


@pytest.mark.usefixtures("data")
class TestRankOneFeature(TestCase):
    def test_fit(self):
        ranker = PercentileRankOneFeature(feature="x3")
        assert ranker.feature_importances_ is None
        ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
        np.testing.assert_array_equal(
            ranker.feature_importances_, np.array([0, 0, 1, 0])
        )

    def test_ranking_on_unavailable_feature_raises_error(self):
        ranker = PercentileRankOneFeature(feature="x5")
        with self.assertRaises(BaselineFeatureNotInMatrix):
            ranker.fit(x=self.data["X_train"], y=self.data["y_train"])

    def test_predict_proba(self):
        for descend_value in [True, False]:
            ranker = PercentileRankOneFeature(feature="x3", descend=descend_value)
            ranker.fit(x=self.data["X_train"], y=self.data["y_train"])
            results = ranker.predict_proba(self.data["X_test"])
            if descend_value:
                expected_results = np.array(
                    [
                        np.zeros(len(self.data["X_test"])),
                        [0.875, 0.125, 0.375, 0, 0.625, 0.25, 0.5, 0.625],
                    ]
                ).transpose()
            else:
                expected_results = np.array(
                    [
                        np.zeros(len(self.data["X_test"])),
                        [0, 0.75, 0.5, 0.875, 0.125, 0.625, 0.375, 0.125],
                    ]
                ).transpose()
            np.testing.assert_array_equal(results, expected_results)


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
            thresholder.feature_importances_, np.array([1, 1, 0, 0])
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
