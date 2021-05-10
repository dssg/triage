import pandas as pd
import os
from .plotting import plot_cats


class SelectionRulePerformancePlotter:
    """Plot regrets over time

    Generates a simple line chart with audition.BoundSelectionRule objects as
    lines, time periods to be analyzed as the x-axis, and the regret of that
    selection rule as the y-axis.

    Args:
        selection_rule_picker (audition.SelectionRulePicker) An object that can
            simulate the effects of picking a selection rule
    """

    def __init__(self, selection_rule_picker, directory=None):
        self.selection_rule_picker = selection_rule_picker
        self.results_for_rule = None
        self.directory = directory

    def plot(
        self,
        bound_selection_rules,
        regret_metric,
        regret_parameter,
        model_group_ids,
        train_end_times,
        plot_type="regret",
    ):
        """Generate a selection rule performance plot for one metric

        Args:
            bound_selection_rules (list of .selection_rules.BoundSelectionRule)
               Selection rule/parameter combinations to plot
            regret_metric (string) -- model evaluation metric, such as 'precision@'
            regret_parameter (string) -- model evaluation metric parameter,
                    such as '300_abs'
            model_group_ids (list of integers) The model group ids to include
                in calculating the data
            train_end_times (list of timestamps) The timestamps to include in
                calculating the data
            plot_type (string) The plot type to show (either 'regret' or 'metric')
        """
        df = self.generate_plot_data(
            bound_selection_rules=bound_selection_rules,
            model_group_ids=model_group_ids,
            train_end_times=train_end_times,
            regret_metric=regret_metric,
            regret_parameter=regret_parameter,
        )
        if plot_type == "regret":
            self.regret_plot_from_dataframe(
                metric=regret_metric, parameter=regret_parameter, df=df
            )
        elif plot_type == "metric":
            self.raw_next_time_plot_from_dataframe(
                metric=regret_metric, parameter=regret_parameter, df=df
            )
        else:
            raise ValueError("Plot type must be either regret or metric")

    def generate_plot_data(
        self,
        bound_selection_rules,
        model_group_ids,
        train_end_times,
        regret_metric,
        regret_parameter,
    ):
        """Create a dataframe suitable for plotting regrets over time

        Args:
            bound_selection_rules (list of .selection_rules.BoundSelectionRule)
               Selection rule/parameter combinations to plot
            model_group_ids (list of integers) The model group ids to include
                in calculating the data
            train_end_times (list of timestamps) The timestamps to include in
                calculating the data
            regret_metric (string) -- model evaluation metric, such as 'precision@'
            regret_parameter (string) -- model evaluation metric parameter,
                such as '300_abs'

        Returns: (pandas.DataFrame) A dataframe with columns 'regret',
            'train_end_time', and 'selection_rule'
        """
        accumulator = list()
        for selection_rule in bound_selection_rules:
            results = self.selection_rule_picker.results_for_rule(
                selection_rule,
                model_group_ids,
                train_end_times,
                regret_metric,
                regret_parameter,
            )
            for result in results:
                accumulator.append(
                    {
                        "train_end_time": result["train_end_time"],
                        "regret": result["dist_from_best_case_next_time"],
                        "selection_rule": selection_rule.descriptive_name,
                        "raw_value_next_time": result["raw_value_next_time"],
                        "model_group_id": result["model_group_id"],
                    }
                )
        return pd.DataFrame.from_records(accumulator)

    def regret_plot_from_dataframe(self, metric, parameter, df, **plt_format_args):
        """Generate a regret-over-time plot from a given dataframe

        Args:
            metric (string) -- model evaluation metric, such as 'precision@'
            parameter (string) -- model evaluation metric parameter,
                such as '300_abs'
            df (pandas.DataFrame) -- A dataframe preresenting regrets over time.
                 Should have columns 'regret', 'train_end_time', 'selection_rule'
        """
        cat_col = "selection_rule"
        plt_title = "Regret for {} {} over time".format(metric, parameter)
        if self.directory:
            path_to_save = os.path.join(
                self.directory, f"regret_over_time_{metric}{parameter}.png"
            )
        else:
            path_to_save = None
        plot_cats(
            frame=df,
            x_col="train_end_time",
            y_col="regret",
            cat_col=cat_col,
            grp_col="selection_rule",
            title=plt_title,
            x_label="train end time",
            y_label="regret in {}".format(metric),
            x_lim=(df["train_end_time"].min(), df["train_end_time"].max()),
            path_to_save=path_to_save,
            alpha=1.0,
            **plt_format_args,
        )

    def raw_next_time_plot_from_dataframe(
        self, metric, parameter, df, **plt_format_args
    ):
        """Generate a 'raw-value-next-time' plot from a given dataframe

        Args:
            metric (string) -- model evaluation metric, such as 'precision@'
            parameter (string) -- model evaluation metric parameter,
                such as '300_abs'
            df (pandas.DataFrame) -- A dataframe that contains raw metric values
                over time given selection rules.
                Should have columns 'raw_value_next_time', 'train_end_time', 'selection_rule'
        """
        cat_col = "selection_rule"
        plt_title = "{} {} next time".format(metric, parameter)
        if self.directory:
            path_to_save = os.path.join(
                self.directory, f"{metric}{parameter}_next_time.png"
            )
        else:
            path_to_save = None
        plot_cats(
            frame=df,
            x_col="train_end_time",
            y_col="raw_value_next_time",
            cat_col=cat_col,
            grp_col="selection_rule",
            title=plt_title,
            x_label="train end time",
            y_label="value of {} next time".format(metric),
            x_lim=(df["train_end_time"].min(), df["train_end_time"].max()),
            path_to_save=path_to_save,
            alpha=1.0,
            **plt_format_args,
        )
