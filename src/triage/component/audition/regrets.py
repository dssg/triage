import copy
import os
import numpy as np
import pandas as pd

from .plotting import plot_cats, plot_bounds


class SelectionRulePicker:
    def __init__(self, distance_from_best_table):
        """Runs simulations of different model group selection rules

        Can look at different results of selection rules, like 'regrets'
        or raw metric values.

        A regret refers to the difference in performance between a model group
        and the best model group for the next testing window
        if a selection rule is followed.

        Args:
            distance_from_best_table (audition.DistanceFromBestTable)
                A pre-populated distance-from-best database table
        """
        self.distance_from_best_table = distance_from_best_table

    def results_for_rule(
        self,
        bound_selection_rule,
        model_group_ids,
        train_end_times,
        regret_metric,
        regret_parameter,
    ):
        """Calculate the regrets, or distance between the chosen model and
            the maximum value next test time

        Arguments:
            bound_selection_rule (function) A function that returns a model group
                given a dataframe of model group performances plus other
                arguments
            model_group_ids (list) The list of model group ids to include in
                the regret analysis
            train_end_times (list) The list of train end times to include in
                the regret analysis
            regret_metric (string) -- model evaluation metric, such as 'precision@'
            regret_parameter (string) -- model evaluation metric parameter,
                such as '300_abs'

        Returns: (list) for each train end time, a dictionary representing the
            model group chosen by the selection rule and its performance

            Should have all keys from the DistanceFromBestTable, most
            importantly
            'distance_from_best_case_next_time'
            'distance_from_best_case',
            'raw_value',
            'raw_value_next_time'
        """
        df = self.distance_from_best_table.as_dataframe(model_group_ids)
        choices = []
        for train_end_time in train_end_times:
            # When plotting rules, we use only the best one model group id
            model_group_id = self.model_group_from_rule(
                bound_selection_rule, model_group_ids, train_end_time
            )
            model_group_id = model_group_id[0]
            choice = df[
                (df["model_group_id"] == model_group_id)
                & (df["train_end_time"] == train_end_time)
                & (df["metric"] == regret_metric)
                & (df["parameter"] == regret_parameter)
            ]
            assert len(choice) == 1
            choices.append(choice.squeeze().to_dict())
        return choices

    def model_group_from_rule(
        self, bound_selection_rule, model_group_ids, train_end_time
    ):
        """Pick a model group that best selects the given selection rule

        In here, we create a subset of the distance from best table dataframe,
        with all data after the given train end time removed, both rows representing
        later time periods but also columns that have access to later data. Calculating and
        passing this allows the selection rules to be written without specific code
        to exclude the future

        Arguments:
            bound_selection_rule (function) A function that returns a model group
                given a dataframe of model group performances plus other
                arguments
            model_group_ids (list) The list of model group ids to consider
            train_end_time (timestamp) The list of train end times to include in
                the regret analysis

        Returns: (int) The model group id chosen by the input selection rule
        """
        df = self.distance_from_best_table.as_dataframe(model_group_ids)
        localized_df = copy.deepcopy(df[df["train_end_time"] <= train_end_time])
        del localized_df["dist_from_best_case_next_time"]

        return bound_selection_rule.pick(localized_df, train_end_time)


class SelectionRulePlotter:
    """Plot selection rules

    Args:
        selection_rule_picker (.SelectionRulePicker)
    """

    def __init__(self, selection_rule_picker, directory=None):
        self.selection_rule_picker = selection_rule_picker
        self.directory = directory

    def plot_bounds(self, metric, parameter):
        """Compute the plot bounds for a given metric and parameter

        Args:
            metric (string) -- model evaluation metric, such as 'precision@'
            parameter (string) -- model evaluation metric parameter,
                such as '300_abs'

        Returns: (number, number) A minimum and maximum x value for the plot
        """

        observed_min, observed_max = \
            self.selection_rule_picker.distance_from_best_table.observed_bounds[(metric, parameter)]
        return plot_bounds(observed_min, observed_max)

    def regret_threshold_dist(self, plot_min, plot_max):
        dist = plot_max - plot_min
        return dist / 100.0

    def regret_thresholds(self, regret_metric, regret_parameter):
        plot_min, plot_max = self.plot_bounds(regret_metric, regret_parameter)
        regret_threshold_dist = self.regret_threshold_dist(plot_min, plot_max)
        return np.arange(plot_min, plot_max, regret_threshold_dist)

    def create_plot_dataframe(
        self,
        bound_selection_rules,
        model_group_ids,
        train_end_times,
        regret_metric,
        regret_parameter,
    ):
        """Create a dataframe suitable for plotting selection rule regrets

        Args:
            bound_selection_rules (list of .selection_rules.BoundSelectionRule)
               Selection rule/parameter combinations to plot
            model_group_ids (list of integers) The model group ids to include
                in calculating the data
            train_end_times (list of timestamps) The timestamps to include in
                calculating the data
            regret_metric (string) The metric (i.e. precision@) to calculate
                regrets against
            regret_parameter (string) The metric parameter (i.e. 100_abs) to
                calculate regrets against

        Returns: (pandas.DataFrame) A dataframe with columns 'regret',
            'pct_of_time', and 'selection_rule'
        """
        accumulator = []
        for selection_rule in bound_selection_rules:
            regrets = [
                result["dist_from_best_case_next_time"]
                for result in self.selection_rule_picker.results_for_rule(
                    selection_rule,
                    model_group_ids,
                    train_end_times,
                    regret_metric,
                    regret_parameter,
                )
            ]
            for regret_threshold in self.regret_thresholds(
                regret_metric, regret_parameter
            ):
                distro = [int(regret < regret_threshold) for regret in regrets]
                accumulator.append(
                    {
                        "regret": regret_threshold,
                        "pct_of_time": np.mean(distro),
                        "selection_rule": selection_rule.descriptive_name,
                    }
                )
        return pd.DataFrame.from_records(accumulator)

    def plot_all_selection_rules(
        self,
        bound_selection_rules,
        model_group_ids,
        train_end_times,
        regret_metric,
        regret_parameter,
    ):
        """Plot the regrets of all given selection rules

        Args:
            bound_selection_rules (list of .selection_rules.BoundSelectionRule)
               Selection rule/parameter combinations to plot
            model_group_ids (list of integers) The model group ids to include
                in calculating the data
            train_end_times (list of timestamps) The timestamps to include in
                calculating the data
            regret_metric (string) The metric (i.e. precision@) to calculate
                regrets against
            regret_parameter (string) The metric parameter (i.e. 100_abs) to
                calculate regrets against
        """
        df_regrets = self.create_plot_dataframe(
            bound_selection_rules,
            model_group_ids,
            train_end_times,
            regret_metric,
            regret_parameter,
        )
        cat_col = "selection_rule"
        plt_title = "Fraction of models X pp worse than best {} {} next time".format(
            regret_metric, regret_parameter
        )
        (plot_min, plot_max) = self.plot_bounds(regret_metric, regret_parameter)

        if self.directory:
            path_to_save = os.path.join(
                self.directory,
                f"regret_distance_from_best_rules_{regret_metric}{regret_parameter}.png",
            )
        else:
            path_to_save = None

        plot_cats(
            frame=df_regrets,
            x_col="regret",
            y_col="pct_of_time",
            cat_col=cat_col,
            grp_col="selection_rule",
            title=plt_title,
            x_label="distance from best {} next time".format(regret_metric),
            y_label="fraction of models",
            x_lim=self.plot_bounds(regret_metric, regret_parameter),
            path_to_save=path_to_save,
            alpha=1.0,
        )
