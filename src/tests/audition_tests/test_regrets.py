from unittest.mock import patch

import numpy as np
import testing.postgresql
from sqlalchemy import create_engine

from triage.component.audition.regrets import SelectionRulePicker, SelectionRulePlotter
from triage.component.audition.selection_rules import (
    best_current_value,
    best_average_value,
    BoundSelectionRule,
)

from .utils import create_sample_distance_table


def test_selection_rule_picker():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        distance_table, model_groups = create_sample_distance_table(engine)

        def pick_spiky(df, train_end_time):
            return [model_groups["spiky"].model_group_id]

        selection_rule_picker = SelectionRulePicker(
            distance_from_best_table=distance_table
        )

        results = selection_rule_picker.results_for_rule(
            bound_selection_rule=BoundSelectionRule(
                descriptive_name="spiky", function=pick_spiky, args={}
            ),
            model_group_ids=[mg.model_group_id for mg in model_groups.values()],
            train_end_times=["2014-01-01", "2015-01-01", "2016-01-01"],
            regret_metric="precision@",
            regret_parameter="100_abs",
        )
        assert [result["dist_from_best_case_next_time"] for result in results] == [
            0.19,
            0.3,
            0.12,
        ]
        assert [result["raw_value"] for result in results] == [0.45, 0.84, 0.45]


def test_selection_rule_picker_with_args():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        distance_table, model_groups = create_sample_distance_table(engine)

        def pick_highest_avg(df, train_end_time, metric, parameter):
            assert len(df["train_end_time"].unique()) == 2
            subsetted = df[(df["metric"] == metric) & (df["parameter"] == parameter)]
            mean = subsetted.groupby(["model_group_id"])["raw_value"].mean()
            return [mean.nlargest(1).index[0]]

        selection_rule_picker = SelectionRulePicker(
            distance_from_best_table=distance_table
        )
        regrets = [
            result["dist_from_best_case_next_time"]
            for result in selection_rule_picker.results_for_rule(
                bound_selection_rule=BoundSelectionRule(
                    descriptive_name="pick_highest_avg",
                    function=pick_highest_avg,
                    args={"metric": "recall@", "parameter": "100_abs"},
                ),
                model_group_ids=[mg.model_group_id for mg in model_groups.values()],
                train_end_times=["2015-01-01"],
                regret_metric="precision@",
                regret_parameter="100_abs",
            )
        ]
        # picking the highest avg recall will pick 'spiky' for this time
        assert regrets == [0.3]


def test_SelectionPlotter_create_plot_dataframe():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        distance_table, model_groups = create_sample_distance_table(engine)
        plotter = SelectionRulePlotter(
            selection_rule_picker=SelectionRulePicker(distance_table)
        )
        plot_df = plotter.create_plot_dataframe(
            bound_selection_rules=[
                BoundSelectionRule(
                    descriptive_name="best_current_precision",
                    function=best_current_value,
                    args={"metric": "precision@", "parameter": "100_abs"},
                ),
                BoundSelectionRule(
                    descriptive_name="best_avg_precision",
                    function=best_average_value,
                    args={"metric": "precision@", "parameter": "100_abs"},
                ),
            ],
            model_group_ids=[mg.model_group_id for mg in model_groups.values()],
            train_end_times=["2014-01-01", "2015-01-01", "2016-01-01"],
            regret_metric="precision@",
            regret_parameter="100_abs",
        )
        # assert that we have the right # of columns and a row for each % diff value
        assert plot_df.shape == (100 * 2, 3)

        # both selection rules have a regret lower than 70
        for value in plot_df[plot_df["regret"] == 0.70]["pct_of_time"].values:
            assert np.isclose(value, 1.0)

        # best avg precision rule should be within 0.14 1/3 of the time
        for value in plot_df[
            (plot_df["regret"] == 0.14)
            & (plot_df["selection_rule"] == "best_avg_precision")
        ]["pct_of_time"].values:
            assert np.isclose(value, 1.0 / 3)


def test_SelectionPlotter_plot():
    with patch("triage.component.audition.regrets.plot_cats") as plot_patch:
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            distance_table, model_groups = create_sample_distance_table(engine)
            plotter = SelectionRulePlotter(
                selection_rule_picker=SelectionRulePicker(distance_table)
            )
            plotter.plot_all_selection_rules(
                bound_selection_rules=[
                    BoundSelectionRule(
                        descriptive_name="best_current_precision",
                        function=best_current_value,
                        args={"metric": "precision@", "parameter": "100_abs"},
                    ),
                    BoundSelectionRule(
                        descriptive_name="best_avg_precision",
                        function=best_average_value,
                        args={"metric": "precision@", "parameter": "100_abs"},
                    ),
                ],
                model_group_ids=[mg.model_group_id for mg in model_groups.values()],
                train_end_times=["2014-01-01", "2015-01-01", "2016-01-01"],
                regret_metric="precision@",
                regret_parameter="100_abs",
            )
        assert plot_patch.called
        args, kwargs = plot_patch.call_args
        assert "regret" in kwargs["frame"]
        assert "pct_of_time" in kwargs["frame"]
        assert kwargs["x_col"] == "regret"
        assert kwargs["y_col"] == "pct_of_time"
