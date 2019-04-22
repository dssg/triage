import unittest
from triage.component.audition.selection_rules import BoundSelectionRule
from triage.component.audition.selection_rule_grid import make_selection_rule_grid

from triage.component.audition.rules_maker import (
    SimpleRuleMaker,
    RandomGroupRuleMaker,
    TwoMetricsRuleMaker,
    create_selection_grid,
)


class TestSimpleRuleMaker(unittest.TestCase):
    def test_add_rule_best_current_value(self):
        Rule = SimpleRuleMaker()
        Rule.add_rule_best_current_value(metric="precision@", parameter="100_abs")
        assert Rule.create() == [
            {
                "selection_rules": [{"name": "best_current_value", "n": 1}],
                "shared_parameters": [{"metric": "precision@", "parameter": "100_abs"}],
            }
        ]

    def test_add_rule_best_average_value(self):
        Rule = SimpleRuleMaker()
        Rule.add_rule_best_average_value(metric="precision@", parameter="100_abs")
        assert Rule.create() == [
            {
                "selection_rules": [{"name": "best_average_value", "n": 1}],
                "shared_parameters": [{"metric": "precision@", "parameter": "100_abs"}],
            }
        ]

    def test_add_rule_lowest_metric_variance(self):
        Rule = SimpleRuleMaker()
        Rule.add_rule_lowest_metric_variance(metric="precision@", parameter="100_abs")
        assert Rule.create() == [
            {
                "selection_rules": [{"name": "lowest_metric_variance", "n": 1}],
                "shared_parameters": [{"metric": "precision@", "parameter": "100_abs"}],
            }
        ]

    def test_add_rule_most_frequent_best_dist(self):
        Rule = SimpleRuleMaker()
        Rule.add_rule_most_frequent_best_dist(
            metric="precision@",
            parameter="100_abs",
            dist_from_best_case=[0.01, 0.05, 0.1, 0.15],
        )
        assert Rule.create() == [
            {
                "selection_rules": [
                    {
                        "dist_from_best_case": [0.01, 0.05, 0.1, 0.15],
                        "name": "most_frequent_best_dist",
                        "n": 1,
                    }
                ],
                "shared_parameters": [{"metric": "precision@", "parameter": "100_abs"}],
            }
        ]

    def test_add_rule_best_avg_recency_weight(self):
        Rule = SimpleRuleMaker()
        Rule.add_rule_best_avg_recency_weight(metric="precision@", parameter="100_abs")
        assert Rule.create() == [
            {
                "selection_rules": [
                    {
                        "curr_weight": [1.5, 2.0, 5.0],
                        "decay_type": ["linear"],
                        "name": "best_avg_recency_weight",
                        "n": 1,
                    }
                ],
                "shared_parameters": [{"metric": "precision@", "parameter": "100_abs"}],
            }
        ]

    def test_add_rule_best_avg_var_penalized(self):
        Rule = SimpleRuleMaker()
        Rule.add_rule_best_avg_var_penalized(
            metric="precision@", parameter="100_abs", stdev_penalty=0.5
        )
        assert Rule.create() == [
            {
                "selection_rules": [
                    {"name": "best_avg_var_penalized", "stdev_penalty": 0.5, "n": 1}
                ],
                "shared_parameters": [{"metric": "precision@", "parameter": "100_abs"}],
            }
        ]


class TestRandomGroupRuleMaker(unittest.TestCase):
    def test_random_model_groups(self):
        Rule = RandomGroupRuleMaker()
        assert Rule.create() == [
            {
                "selection_rules": [{"name": "random_model_group", "n": 1}],
                "shared_parameters": [{}],
            }
        ]


class TestTwoMetricsRuleMaker(unittest.TestCase):
    def test_add_two_metrics_rule_maker(self):
        Rule = TwoMetricsRuleMaker()
        Rule.add_rule_best_average_two_metrics(
            metric1="precision@",
            parameter1="100_abs",
            metric2="recall@",
            parameter2="300_abs",
            metric1_weight=[0.5],
        )
        assert Rule.create() == [
            {
                "selection_rules": [
                    {
                        "metric1_weight": [0.5],
                        "name": "best_average_two_metrics",
                        "metric2": ["recall@"],
                        "parameter2": ["300_abs"],
                        "n": 1,
                    }
                ],
                "shared_parameters": [
                    {"metric1": "precision@", "parameter1": "100_abs"}
                ],
            }
        ]


class TestCreateSelectionRuleGrid(unittest.TestCase):
    def test_create_grid(self):
        """
        input_data = [{
            'shared_parameters': [
                {'metric': 'precision@', 'parameter': '100_abs'},
                {'metric': 'recall@', 'parameter': '100_abs'},
            ],
            'selection_rules': [
                {'name': 'most_frequent_best_dist', 'dist_from_best_case': [0.1, 0.2, 0.3]},
                {'name': 'best_current_value'}
            ]
        }, {
            'shared_parameters': [
                {'metric1': 'precision@', 'parameter1': '100_abs'},
            ],
            'selection_rules': [
                {
                    'name': 'best_average_two_metrics',
                    'metric2': ['recall@'],
                    'parameter2': ['100_abs'],
                    'metric1_weight': [0.4, 0.5, 0.6]
                },
            ]
        }]
        """
        Rule1 = SimpleRuleMaker()
        Rule1.add_rule_best_current_value(metric="precision@", parameter="100_abs")
        Rule1.add_rule_most_frequent_best_dist(
            metric="recall@", parameter="100_abs", dist_from_best_case=[0.1, 0.2, 0.3]
        )

        Rule2 = TwoMetricsRuleMaker()
        Rule2.add_rule_best_average_two_metrics(
            metric1="precision@",
            parameter1="100_abs",
            metric2="recall@",
            parameter2="100_abs",
            metric1_weight=[0.4, 0.5, 0.6],
        )

        expected_output = [
            BoundSelectionRule(
                descriptive_name="most_frequent_best_dist_precision@_100_abs_0.1",
                function_name="most_frequent_best_dist",
                args={
                    "metric": "precision@",
                    "parameter": "100_abs",
                    "dist_from_best_case": 0.1,
                },
            ),
            BoundSelectionRule(
                descriptive_name="most_frequent_best_dist_precision@_100_abs_0.2",
                function_name="most_frequent_best_dist",
                args={
                    "metric": "precision@",
                    "parameter": "100_abs",
                    "dist_from_best_case": 0.2,
                },
            ),
            BoundSelectionRule(
                descriptive_name="most_frequent_best_dist_precision@_100_abs_0.3",
                function_name="most_frequent_best_dist",
                args={
                    "metric": "precision@",
                    "parameter": "100_abs",
                    "dist_from_best_case": 0.3,
                },
            ),
            BoundSelectionRule(
                descriptive_name="most_frequent_best_dist_recall@_100_abs_0.1",
                function_name="most_frequent_best_dist",
                args={
                    "metric": "recall@",
                    "parameter": "100_abs",
                    "dist_from_best_case": 0.1,
                },
            ),
            BoundSelectionRule(
                descriptive_name="most_frequent_best_dist_recall@_100_abs_0.2",
                function_name="most_frequent_best_dist",
                args={
                    "metric": "recall@",
                    "parameter": "100_abs",
                    "dist_from_best_case": 0.2,
                },
            ),
            BoundSelectionRule(
                descriptive_name="most_frequent_best_dist_recall@_100_abs_0.3",
                function_name="most_frequent_best_dist",
                args={
                    "metric": "recall@",
                    "parameter": "100_abs",
                    "dist_from_best_case": 0.3,
                },
            ),
            BoundSelectionRule(
                descriptive_name="best_current_value_precision@_100_abs",
                function_name="best_current_value",
                args={"metric": "precision@", "parameter": "100_abs"},
            ),
            BoundSelectionRule(
                descriptive_name="best_current_value_recall@_100_abs",
                function_name="best_current_value",
                args={"metric": "recall@", "parameter": "100_abs"},
            ),
            BoundSelectionRule(
                descriptive_name="best_average_two_metrics_precision@_100_abs_recall@_100_abs_0.4",
                function_name="best_average_two_metrics",
                args={
                    "metric1": "precision@",
                    "parameter1": "100_abs",
                    "metric2": "recall@",
                    "parameter2": "100_abs",
                    "metric1_weight": 0.4,
                },
            ),
            BoundSelectionRule(
                descriptive_name="best_average_two_metrics_precision@_100_abs_recall@_100_abs_0.5",
                function_name="best_average_two_metrics",
                args={
                    "metric1": "precision@",
                    "parameter1": "100_abs",
                    "metric2": "recall@",
                    "parameter2": "100_abs",
                    "metric1_weight": 0.5,
                },
            ),
            BoundSelectionRule(
                descriptive_name="best_average_two_metrics_precision@_100_abs_recall@_100_abs_0.6",
                function_name="best_average_two_metrics",
                args={
                    "metric1": "precision@",
                    "parameter1": "100_abs",
                    "metric2": "recall@",
                    "parameter2": "100_abs",
                    "metric1_weight": 0.6,
                },
            ),
        ]
        expected_output.sort(key=lambda x: x.descriptive_name)
        grid = sorted(
            make_selection_rule_grid(create_selection_grid(Rule1, Rule2)),
            key=lambda x: x.descriptive_name,
        )
        assert len(grid) == len(expected_output)
        for expected_rule, actual_rule in zip(expected_output, grid):
            assert expected_rule.descriptive_name == actual_rule.descriptive_name
