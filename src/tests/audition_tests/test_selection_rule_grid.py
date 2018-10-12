from triage.component.audition.selection_rules import BoundSelectionRule
from triage.component.audition.selection_rule_grid import make_selection_rule_grid


def test_selection_rule_grid():
    input_data = [
        {
            "shared_parameters": [
                {"metric": "precision@", "parameter": "100_abs"},
                {"metric": "recall@", "parameter": "100_abs"},
            ],
            "selection_rules": [
                {
                    "name": "most_frequent_best_dist",
                    "dist_from_best_case": [0.1, 0.2, 0.3],
                },
                {"name": "best_current_value"},
            ],
        },
        {
            "shared_parameters": [{"metric1": "precision@", "parameter1": "100_abs"}],
            "selection_rules": [
                {
                    "name": "best_average_two_metrics",
                    "metric2": ["recall@"],
                    "parameter2": ["100_abs"],
                    "metric1_weight": [0.4, 0.5, 0.6],
                }
            ],
        },
    ]

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

    # sort both lists so we can compare them without resorting to a hash
    expected_output.sort(key=lambda x: x.descriptive_name)
    grid = sorted(
        make_selection_rule_grid(input_data), key=lambda x: x.descriptive_name
    )
    assert len(grid) == len(expected_output)
    for expected_rule, actual_rule in zip(expected_output, grid):
        assert expected_rule.descriptive_name == actual_rule.descriptive_name
