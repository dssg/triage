import logging

import yaml
from smart_open import smart_open

from .distance_from_best import DistanceFromBestTable, BestDistancePlotter
from .thresholding import ModelGroupThresholder
from .regrets import SelectionRulePicker, SelectionRulePlotter
from .selection_rule_performance import SelectionRulePerformancePlotter
from .model_group_performance import ModelGroupPerformancePlotter
from .selection_rule_grid import make_selection_rule_grid


class Auditioner(object):

    def __init__(
        self,
        db_engine,
        model_group_ids,
        train_end_times,
        initial_metric_filters,
        models_table=None,
        distance_table=None,
    ):
        """Filter model groups using a two-step process:

        1. Broad thresholds to filter out truly bad models
        2. A selection rule grid to find the best model groups over time
            for each of a variety of methods

        This is achieved by creating a 'best distance' table, which functions like a
        denormalized 'model group/model/evaluations', storing for each
        model group/train end time/metric/parameter:
            1. the raw evaluation value,
            2. the distance of that evaluation metric from the best model group at that train time,
            3. and the distance of the metric from the best model group the *next* train time

        Each of the steps is computed based on the data in this table, and an iterative process of
            sending thresholding/selection configuration and viewing the results.

        For step 1, the initial configuration is sent in this constructor
            (as 'initial_metric_filters', format detailed below), future iterations of this
            configuration are sent to 'update_metric_filters'.

        For step 2, all configuration is sent to the object via 'register_selection_rule_grid',
            and its format is detailed in that method's docstring

        Args:
            db_engine (sqlalchemy.engine) A database engine with access to a
                results schema of a completed modeling run
            model_group_ids (list) A large list of model groups to audition. No effort should
                be needed to pick 'good' model groups, but they should all be groups that could
                be used if they are found to perform well. They should also each have evaluations
                for any train end times you wish to include in analysis
            train_end_times (list) A list of train end times that all of the given model groups
                contain evaluations for and that you want to be deemed important in the analysis
            initial_metric_filters (list) A list of metrics to filter model
                groups on, and how to filter them. Each entry should be a dict
                with the keys:

                    metric (string) -- model evaluation metric, such as 'precision@'
                    parameter (string) -- model evaluation metric parameter,
                        such as '300_abs'
                    max_below_best (float) The maximum value that the given metric
                        can be below the best for a given train end time
                    threshold_value (float) The minimum value that the given metric can be
            models_table (string, optional) The name of the results schema
                models table that you want to use. Will default to 'models',
                which is also the default in triage.
            distance_table (string, optional) The name of the 'best distance' table to use.
                Will default to 'best_distance', but this can be sent if you want to avoid
                clobbering the results from a prior analysis.
        """
        self.metric_filters = initial_metric_filters
        # sort the train end times so we can reliably pick off the last time later
        self.train_end_times = sorted(train_end_times)

        models_table = models_table or 'models'
        distance_table = distance_table or 'best_distance'
        self.distance_from_best_table = DistanceFromBestTable(
            db_engine=db_engine,
            models_table=models_table,
            distance_table=distance_table
        )
        self.best_distance_plotter = BestDistancePlotter(self.distance_from_best_table)
        self.model_group_thresholder = ModelGroupThresholder(
            distance_from_best_table=self.distance_from_best_table,
            train_end_times=train_end_times,
            initial_model_group_ids=model_group_ids,
            initial_metric_filters=initial_metric_filters
        )
        self.model_group_performance_plotter = \
            ModelGroupPerformancePlotter(self.distance_from_best_table)

        self.selection_rule_picker = SelectionRulePicker(self.distance_from_best_table)
        self.selection_rule_plotter = SelectionRulePlotter(self.selection_rule_picker)
        self.selection_rule_performance_plotter = \
            SelectionRulePerformancePlotter(self.selection_rule_picker)

        self.distance_from_best_table.create_and_populate(
            model_group_ids,
            self.train_end_times,
            self.metrics
        )
        self.results_for_rule = {}

    @property
    def metrics(self):
        return [
            {'metric': f['metric'], 'parameter': f['parameter']}
            for f in self.metric_filters
        ]

    @property
    def thresholded_model_group_ids(self):
        """The model group thresholder will have a varying list of model group ids
        depending on its current thresholding rules, this is a reference to whatever
        that current list is.

        Returns: (list) of model group ids
        """
        return self.model_group_thresholder.model_group_ids

    @property
    def average_regret_for_rules(self):
        result = dict()
        for k in self.results_for_rule.keys():
            result[k] = self.results_for_rule[k]\
                .groupby('selection_rule')['regret']\
                .mean()\
                .to_dict()
        return result

    @property
    def selection_rule_model_group_ids(self):
        """Calculate the current winners for each selection rule and the most recent date

        Returns: (dict) keys are selection rule descriptive names, values are the model group id
            chosen by them
        """
        logging.info('Calculating selection rule picks for all rules')
        model_group_ids = dict()
        thresholded_ids = self.thresholded_model_group_ids
        for selection_rule in self.selection_rules:
            logging.info('Calculating selection rule picks for %s', selection_rule)
            model_group_ids[selection_rule.descriptive_name] =\
                self.selection_rule_picker.model_group_from_rule(
                    bound_selection_rule=selection_rule,
                    model_group_ids=thresholded_ids,
                    # evaluate the selection rules for the most recent
                    # time period and use those as candidate model groups
                    train_end_time=self.train_end_times[-1],
                )
            logging.info(
                'For rule %s, model group %s was picked',
                selection_rule,
                model_group_ids[selection_rule.descriptive_name]
            )
        return model_group_ids

    def plot_model_groups(self):
        """Display model group plots, one of the below for each configured metric.

        1. A cumulative plot showing the effect of different worse-than-best
        thresholds for the given metric across the thresholded model groups.

        2. A performance-over-time plot showing the value for the given
        metric over time for each thresholded model group
        """
        logging.info('Showing best distance plots for all metrics')
        thresholded_model_group_ids = self.thresholded_model_group_ids
        if len(thresholded_model_group_ids) == 0:
            logging.warning('Zero model group ids found that passed configured thresholds. '
                            'Nothing to plot')
            return
        self.best_distance_plotter.plot_all_best_dist(
            self.metrics,
            thresholded_model_group_ids,
            self.train_end_times
        )
        logging.info('Showing model group performance plots for all metrics')
        self.model_group_performance_plotter.plot_all(
            metric_filters=self.metric_filters,
            model_group_ids=thresholded_model_group_ids,
            train_end_times=self.train_end_times
        )

    def set_one_metric_filter(
            self,
            metric='precision@',
            parameter='100_abs',
            max_from_best=0.05,
            threshold_value=0.1):
        """Set one thresholding metric filter
        If one wnats to update multiple filters, one should use `update_metric_filters()` instead.

        Args:
            metric (string) model evaluation metric such as 'precision@'
            parameter (string) model evaluation parameter such as '100_abs'
            max_from_best (string) The maximum value that the given metric can be below the best
                for a given train end time
            threshold_value (string) The thresold value that the given metric can be
            plot (boolean, default True) Whether or not to also plot model group performance
                and thresholding details at this time.
        """
        new_filters = [{'metric': metric,
                        'parameter': parameter,
                        'max_from_best': max_from_best,
                        'threshold_value': threshold_value
                        }]
        self.update_metric_filters(new_filters)

    def update_metric_filters(
            self,
            new_filters=None,
            plot=True):
        """Update the thresholding metric filters

        Args:
            new_filters (list) A list of metrics to filter model
                groups on, and how to filter them. This is an identical format to
                the list given to 'initial_metric_filters' in the constructor.
                Each entry should be a dict with the keys:

                    metric (string) -- model evaluation metric, such as 'precision@'
                    parameter (string) -- model evaluation metric parameter,
                        such as '300_abs'
                    max_below_best (float) The maximum value that the given metric
                        can be below the best for a given train end time
                    threshold_value (float) The threshold value that the given metric can be
            plot (boolean, default True) Whether or not to also plot model group performance
                and thresholding details at this time.
        """
        logging.info('Updating metric filters with new config %s', new_filters)
        self.model_group_thresholder.update_filters(new_filters)
        if plot:
            logging.info('After config update, plotting model groups')
            self.plot_model_groups()

    def plot_selection_rules(self):
        """Plot data about the configured selection rules. The three plots outlined below
        are plotted for each metric.

        We base a lot of this on the concept of the 'regret'.
        The regret refers to the difference in performance between a model group
        and the best model group for the next testing window if a selection rule is followed.

        1. A distance-next-time plot, showing the fraction of models worse then a succession of
            regret thresholds for each selection rule
        2. A regret-over-time plot for each selection rule
        3. A metric-over-time plot for each selection rule
        """
        for metric_definition in self.metrics:
            common_kwargs = dict(
                bound_selection_rules=self.selection_rules,
                regret_metric=metric_definition['metric'],
                regret_parameter=metric_definition['parameter'],
                model_group_ids=self.thresholded_model_group_ids,
                train_end_times=self.train_end_times[:-1],
                # We can't calculate regrets for the most recent train end time,
                # so don't send that in. Assumes that the train_end_times
                # are sorted in the constructor
            )
            self.selection_rule_plotter.plot_all_selection_rules(**common_kwargs)

            df = self.selection_rule_performance_plotter.generate_plot_data(**common_kwargs)
            self.selection_rule_performance_plotter.regret_plot_from_dataframe(
                metric=metric_definition['metric'],
                parameter=metric_definition['parameter'],
                df=df
            )
            self.selection_rule_performance_plotter.raw_next_time_plot_from_dataframe(
                metric=metric_definition['metric'],
                parameter=metric_definition['parameter'],
                df=df
            )

            key = metric_definition['metric'] + metric_definition['parameter']
            self.results_for_rule[key] = df

    def register_selection_rule_grid(self, rule_grid, plot=True):
        """Register a grid of selection rules

        Args:
            rule_grid (list) Groups of selection rules that share parameters

            Each entry in the list is considered a group, and is expected to be a dict
                with two keys: 'shared_parameters', and 'selection_rules'.

                'shared_parameters': A list of dicts, each with a set of parameters that are taken
                    by all selection rules in this group.

                    For each of these shared parameter sets, the grid will create selection rules
                    combining the set with all possible selection rule/parameter combinations.

                    This can be used to quickly combine, say, a variety of rules that
                        all are concerned with precision at top 100 entities.

                'selection_rules': A list of dicts, each with:
                    A 'name' attribute that matches a selection rule in audition.selection_rules
                    Parameters and values taken by that selection rule. Values in list form are
                    all added to the grid.
                    If the selection rule has no parameters, or the parameters are all covered
                    by the shared parameters in this group, none are needed here.

                Each selection rule in the group must have all of its required parameters
                covered by the shared parameters in its group and the parameters given to it.

                Refer to audition.selection_rules.SELECTION_RULES for available selection rules
                and their needed parameters.
                The exceptions are the first two arguments to each selection rule,
                'df' and 'train_end_time'.
                These are contextual and thus provided internally by Audition.

            Example: [{
                    'shared_parameters': [
                            {'metric': 'precision@', 'parameter': '100_abs'},
                            {'metric': 'recall@', 'parameter': '100_abs'},
                        ],
                        'selection_rules': [
                            {'name': 'most_frequent_best_dist',
                             'dist_from_best_case': [0.1, 0.2, 0.3]},
                            {'name': 'best_current_value'}
                        ]
                }]
            plot (boolean, defaults to True) Whether or not to plot the selection
                rules at this time.
        """
        self.selection_rules = make_selection_rule_grid(rule_grid)
        if plot:
            self.plot_selection_rules()

    def write_tyra_config(self, write_path):
        """Write the final selection rules and model groups to a YAML file, for later use
        by the 'Tyra' webapp.

        Args:
            write_path (string) The smart_open-ready path to a file where
                the resulting YAML file should go.
        """
        logging.info('Writing final model group ids to export to Tyra')
        with smart_open(write_path, 'w') as f:
            yaml.dump({'selection_rule_model_groups': self.selection_rule_model_group_ids}, f)
