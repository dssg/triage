from unittest.mock import patch

import testing.postgresql
from sqlalchemy import create_engine

from triage.component.audition.regrets import SelectionRulePicker
from triage.component.audition.selection_rule_performance import (
    SelectionRulePerformancePlotter,
)
from triage.component.audition.selection_rules import BoundSelectionRule

from .utils import create_sample_distance_table


TRAIN_END_TIMES = ["2014-01-01", "2015-01-01"]


class MockSelectionRulePicker:
    def results_for_rule(*args, **kwargs):
        return [
            {
                "train_end_time": TRAIN_END_TIMES[0],
                "dist_from_best_case_next_time": 0.15,
                "raw_value_next_time": 0.5,
                "model_group_id": 1,
            },
            {
                "train_end_time": TRAIN_END_TIMES[1],
                "dist_from_best_case_next_time": 0.30,
                "raw_value_next_time": 0.4,
                "model_group_id": 2,
            },
        ]


def test_SelectionRulePerformancePlotter_generate_plot_data():
    plotter = SelectionRulePerformancePlotter(MockSelectionRulePicker())
    df = plotter.generate_plot_data(
        bound_selection_rules=[
            BoundSelectionRule(
                function_name="best_current_value",
                args={"metric": "precision@", "parameter": "100_abs"},
            ),
            BoundSelectionRule(
                function_name="best_average_value",
                args={"metric": "precision@", "parameter": "100_abs"},
            ),
        ],
        regret_metric="precision@",
        regret_parameter="100_abs",
        model_group_ids=[1, 2],
        train_end_times=TRAIN_END_TIMES,
    )
    print(df.to_dict("list"))
    assert df.to_dict("list") == {
        "selection_rule": [
            "best_current_value_precision@_100_abs",
            "best_current_value_precision@_100_abs",
            "best_average_value_precision@_100_abs",
            "best_average_value_precision@_100_abs",
        ],
        "train_end_time": TRAIN_END_TIMES + TRAIN_END_TIMES,
        "regret": [0.15, 0.30, 0.15, 0.30],
        "raw_value_next_time": [0.5, 0.4, 0.5, 0.4],
        "model_group_id": [1, 2, 1, 2],
    }


def test_SelectionRulePerformancePlotter_plot_regrets():
    with patch(
        "triage.component.audition.selection_rule_performance.plot_cats"
    ) as plot_patch:
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            distance_table, model_groups = create_sample_distance_table(engine)
            plotter = SelectionRulePerformancePlotter(
                SelectionRulePicker(distance_table)
            )
            plotter.plot(
                bound_selection_rules=[
                    BoundSelectionRule(
                        function_name="best_current_value",
                        args={"metric": "precision@", "parameter": "100_abs"},
                    ),
                    BoundSelectionRule(
                        function_name="best_average_value",
                        args={"metric": "precision@", "parameter": "100_abs"},
                    ),
                ],
                regret_metric="precision@",
                regret_parameter="100_abs",
                model_group_ids=[1, 2],
                train_end_times=["2014-01-01", "2015-01-01"],
            )
        assert plot_patch.called
        args, kwargs = plot_patch.call_args
        assert "regret" in kwargs["frame"]
        assert "train_end_time" in kwargs["frame"]
        assert kwargs["x_col"] == "train_end_time"
        assert kwargs["y_col"] == "regret"


def test_SelectionRulePerformancePlotter_plot_metrics():
    with patch(
        "triage.component.audition.selection_rule_performance.plot_cats"
    ) as plot_patch:
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            distance_table, model_groups = create_sample_distance_table(engine)
            plotter = SelectionRulePerformancePlotter(
                SelectionRulePicker(distance_table)
            )
            plotter.plot(
                bound_selection_rules=[
                    BoundSelectionRule(
                        function_name="best_current_value",
                        args={"metric": "precision@", "parameter": "100_abs"},
                    ),
                    BoundSelectionRule(
                        function_name="best_average_value",
                        args={"metric": "precision@", "parameter": "100_abs"},
                    ),
                ],
                regret_metric="precision@",
                regret_parameter="100_abs",
                model_group_ids=[1, 2],
                train_end_times=["2014-01-01", "2015-01-01"],
                plot_type="metric",
            )
        assert plot_patch.called
        args, kwargs = plot_patch.call_args
        assert "raw_value_next_time" in kwargs["frame"]
        assert "train_end_time" in kwargs["frame"]
        assert kwargs["x_col"] == "train_end_time"
        assert kwargs["y_col"] == "raw_value_next_time"
