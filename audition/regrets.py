import copy
from audition.selection_rules import *
import numpy
import pandas
from audition.plotting import plot_cats, plot_bounds


class RegretCalculator(object):
    def __init__(self, distance_from_best_table):
        """Calculates 'regrets' for different model group selection rules

        A regret refers to the difference in performance between a model group
        and the best model group for the next testing window
        if a selection rule is followed.

        Args:
            distance_from_best_table (audition.DistanceFromBestTable)
                A pre-populated distance-from-best database table
        """
        self.distance_from_best_table = distance_from_best_table

    def regrets_for_rule(
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
            selection_rule (function) A function that returns a model group
                given a dataframe of model group performances plus other
                arguments
            model_group_ids (list) The list of model group ids to include in
                the regret analysis
            train_end_times (list) The list of train end times to include in
                the regret analysis
            metric (string) -- model evaluation metric, such as 'precision@'
            parameter (string) -- model evaluation metric parameter,
                such as '300_abs'
            selection_rule_args (dict) Arguments that the given selection rule
                will accept as keyword arguments

        Returns: (list) for each train end time, the distance between the
            model group chosen by the selection rule and the potential
            maximum for the next train end time
        """
        regrets = []
        df = self.distance_from_best_table.as_dataframe(model_group_ids)

        for train_end_time in train_end_times:
            localized_df = copy.deepcopy(
                df[df['train_end_time'] <= train_end_time]
            )
            del localized_df['dist_from_best_case_next_time']

            choice = bound_selection_rule.pick(localized_df, train_end_time)
            regret_result = df[
                (df['model_group_id'] == choice) &
                (df['train_end_time'] == train_end_time) &
                (df['metric'] == regret_metric) &
                (df['parameter'] == regret_parameter)
            ]
            assert len(regret_result) == 1
            regrets.append(regret_result['dist_from_best_case_next_time'].values[0])
        return regrets


class SelectionRulePlotter(object):
    """Plot selection rules

    Args:
        regret_calculator (.RegretCalculator)
    """
    def __init__(self, regret_calculator):
        self.regret_calculator = regret_calculator

    def plot_bounds(self, metric, parameter):
        """Compute the plot bounds for a given metric and parameter

        Args:
            metric (string) -- model evaluation metric, such as 'precision@'
            parameter (string) -- model evaluation metric parameter,
                such as '300_abs'

        Returns: (number, number) A minimum and maximum x value for the plot
        """

        observed_min, observed_max = self.regret_calculator\
            .distance_from_best_table\
            .observed_bounds[(metric, parameter)]
        return plot_bounds(observed_min, observed_max)

    def regret_threshold_dist(self, plot_min, plot_max):
        dist = plot_max - plot_min
        return dist/100.0

    def regret_thresholds(self, regret_metric, regret_parameter):
        plot_min, plot_max = self.plot_bounds(regret_metric, regret_parameter)
        regret_threshold_dist = self.regret_threshold_dist(plot_min, plot_max)
        return numpy.arange(plot_min, plot_max, regret_threshold_dist)

    def create_plot_dataframe(
        self,
        bound_selection_rules,
        model_group_ids,
        train_end_times,
        regret_metric,
        regret_parameter
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
        accumulator = list()
        for selection_rule in bound_selection_rules:
            regrets = self.regret_calculator.regrets_for_rule(
                selection_rule,
                model_group_ids,
                train_end_times,
                regret_metric,
                regret_parameter
            )
            for regret_threshold in self.regret_thresholds(regret_metric, regret_parameter):
                pct_of_time = numpy.mean([1 if regret < regret_threshold else 0 for regret in regrets])
                accumulator.append({
                    'regret': regret_threshold,
                    'pct_of_time': pct_of_time,
                    'selection_rule': selection_rule.descriptive_name,
                })
        return pandas.DataFrame.from_records(accumulator)

    def plot_all_selection_rules(
        self,
        bound_selection_rules,
        model_group_ids,
        train_end_times,
        regret_metric,
        regret_parameter
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
            regret_parameter
        )
        cat_col = 'selection_rule'
        plt_title = 'Fraction of models X pp worse than best {} {} next time'.format(regret_metric, regret_parameter)
        plot_min, plot_max = self.plot_bounds(regret_metric, regret_parameter)

        plot_cats(
            frame=df_regrets,
            x_col='regret',
            y_col='pct_of_time',
            cat_col=cat_col,
            grp_col='selection_rule',
            title=plt_title,
            x_label='distance from best {} next time'.format(regret_metric),
            y_label='fraction of models',
            x_lim=self.plot_bounds(regret_metric, regret_parameter)
        )
