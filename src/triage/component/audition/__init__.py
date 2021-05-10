import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import json
import os

from .distance_from_best import DistanceFromBestTable, BestDistancePlotter
from .thresholding import model_groups_filter, ModelGroupThresholder
from .regrets import SelectionRulePicker, SelectionRulePlotter
from .selection_rule_performance import SelectionRulePerformancePlotter
from .model_group_performance import ModelGroupPerformancePlotter
from .selection_rule_grid import make_selection_rule_grid
from .pre_audition import PreAudition


class Auditioner:
    def __init__(
        self,
        db_engine,
        model_group_ids,
        train_end_times,
        initial_metric_filters,
        models_table=None,
        distance_table=None,
        directory=None,
        agg_type='worst',
        baseline_model_group_ids=None,
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
            db_engine (sqlalchemy.engine): A database engine with access to a
                results schema of a completed modeling run
            model_group_ids (list): A large list of model groups to audition. No effort should
                be needed to pick 'good' model groups, but they should all be groups that could
                be used if they are found to perform well. They should also each have evaluations
                for any train end times you wish to include in analysis
            train_end_times (list): A list of train end times that all of the given model groups
                contain evaluations for and that you want to be deemed important in the analysis
            initial_metric_filters (list): A list of metrics to filter model
                groups on, and how to filter them. Each entry should be a dict
                of the format:

                    {
                        'metric': 'string',
                        'parameter': 'string',
                        'max_below_best': .5,
                        'threshold_value': .5
                     }

                    metric (string): model evaluation metric, such as 'precision@'
                    parameter (string): model evaluation metric parameter,
                        such as '300_abs'
                    max_below_best (float): The maximum value that the given metric
                        can be below the best for a given train end time
                    threshold_value (float): The minimum value that the given metric can be
            models_table (string, optional): The name of the results schema
                models table that you want to use. Will default to 'models',
                which is also the default in triage.
            distance_table (string, optional): The name of the 'best distance' table to use.
                Will default to 'best_distance', but this can be sent if you want to avoid
                clobbering the results from a prior analysis.
            agg_type (string) Method for aggregating metric values (for instance, if there
                are multiple models at a given train_end_time with different random seeds).
                Can be: 'mean', 'best', or 'worst' (the default)
            baseline_model_group_ids (list): An optional list of model groups for baseline 
                models which will be included on all plots without being subject to filtering 
                or included as candidate models from the selection process.
        """
        self.metric_filters = initial_metric_filters
        # sort the train end times so we can reliably pick off the last time later
        self.train_end_times = sorted(train_end_times)
        self.directory = directory
        models_table = models_table or "models"
        distance_table = distance_table or "best_distance"
        self.distance_from_best_table = DistanceFromBestTable(
            db_engine=db_engine,
            models_table=models_table,
            distance_table=distance_table,
            agg_type=agg_type
        )
        self.best_distance_plotter = BestDistancePlotter(
            self.distance_from_best_table, self.directory
        )

        if baseline_model_group_ids:
            self.baseline_model_groups = model_groups_filter(
                train_end_times=train_end_times,
                initial_model_group_ids=baseline_model_group_ids,
                models_table=models_table,
                db_engine=db_engine,
            )
        else:
            self.baseline_model_groups = set([])

        self.first_pass_model_groups = model_groups_filter(
            train_end_times=train_end_times,
            initial_model_group_ids=model_group_ids,
            models_table=models_table,
            db_engine=db_engine,
        )

        self.model_group_thresholder = ModelGroupThresholder(
            distance_from_best_table=self.distance_from_best_table,
            train_end_times=train_end_times,
            initial_model_group_ids=self.first_pass_model_groups,
            initial_metric_filters=initial_metric_filters,
        )
        self.model_group_performance_plotter = ModelGroupPerformancePlotter(
            self.distance_from_best_table, self.directory
        )

        self.selection_rule_picker = SelectionRulePicker(self.distance_from_best_table)
        self.selection_rule_plotter = SelectionRulePlotter(
            self.selection_rule_picker, self.directory
        )
        self.selection_rule_performance_plotter = SelectionRulePerformancePlotter(
            self.selection_rule_picker, directory
        )

        # note we populate the distance from best table using both the
        # baseline and candidate model groups
        self.distance_from_best_table.create_and_populate(
            self.first_pass_model_groups | self.baseline_model_groups, 
            self.train_end_times, 
            self.metrics
        )
        self.results_for_rule = {}

    @property
    def metrics(self):
        return [
            {"metric": f["metric"], "parameter": f["parameter"]}
            for f in self.metric_filters
        ]

    @property
    def thresholded_model_group_ids(self) -> list:
        """The model group thresholder will have a varying list of model group ids
        depending on its current thresholding rules, this is a reference to whatever
        that current list is.

        Returns:
            list of model group ids allowed by the current metric threshold rules
        """
        return self.model_group_thresholder.model_group_ids

    @property
    def average_regret_for_rules(self) -> dict:
        """
        Returns the average regret for each selection rule, over the specified list of train/test periods.

        Returns:
            A dict with a key-value pair for each selection rule and the average regret for that rule. Structure:

                {'descriptive rule_name': .5}
        """
        result = dict()
        for k in self.results_for_rule.keys():
            result[k] = (
                self.results_for_rule[k]
                .groupby("selection_rule")["regret"]
                .mean()
                .to_dict()
            )
        return result

    @property
    def selection_rule_model_group_ids(self) -> dict:
        """
        Calculate the current winners for each selection rule and the most recent date

        Returns:
            A dict with a key-value pair for each selection rule and the list of n
            model_group_ids that it selected. Structure:

                {'descriptive rule_name':[1,2,3]}
        """
        logger.debug("Calculating selection rule picks for all rules")
        model_group_ids = dict()
        thresholded_ids = self.thresholded_model_group_ids
        for selection_rule in self.selection_rules:
            logger.debug("Calculating selection rule picks for %s", selection_rule)
            model_group_ids[
                selection_rule.descriptive_name
            ] = self.selection_rule_picker.model_group_from_rule(
                bound_selection_rule=selection_rule,
                model_group_ids=thresholded_ids,
                # evaluate the selection rules for the most recent
                # time period and use those as candidate model groups
                train_end_time=self.train_end_times[-1],
            )
            logger.debug(
                "For rule %s, model group %s was picked",
                selection_rule,
                model_group_ids[selection_rule.descriptive_name],
            )
        return model_group_ids

    def save_result_model_group_ids(self, fname="results_model_group_ids.json"):
        with open(os.path.join(self.directory, fname), "w") as f:
            f.write(json.dumps(self.selection_rule_model_group_ids))

    def plot_model_groups(self):
        """Display model group plots, one of the below for each configured metric.

        1. A cumulative plot showing the effect of different worse-than-best
        thresholds for the given metric across the thresholded model groups.

        2. A performance-over-time plot showing the value for the given
        metric over time for each thresholded model group
        """
        logger.debug("Showing best distance plots for all metrics")
        thresholded_model_group_ids = self.thresholded_model_group_ids
        if len(thresholded_model_group_ids) == 0:
            logger.warning(
                "Zero model group ids found that passed configured thresholds. "
                "Nothing to plot"
            )
            return
        self.best_distance_plotter.plot_all_best_dist(
            self.metrics, 
            thresholded_model_group_ids | self.baseline_model_groups, 
            self.train_end_times
        )
        logger.debug("Showing model group performance plots for all metrics")
        self.model_group_performance_plotter.plot_all(
            metric_filters=self.metric_filters,
            model_group_ids=thresholded_model_group_ids | self.baseline_model_groups,
            train_end_times=self.train_end_times,
        )

    def set_one_metric_filter(
        self,
        metric="precision@",
        parameter="100_abs",
        max_from_best=0.05,
        threshold_value=0.1,
    ):
        """Set one thresholding metric filter
        If one wnats to update multiple filters, one should use `update_metric_filters()` instead.

        Args:
            metric (string): model evaluation metric such as 'precision@'
            parameter (string): model evaluation parameter such as '100_abs'
            max_from_best (string): The maximum value that the given metric can be below the best
                for a given train end time
            threshold_value (string): The thresold value that the given metric can be
            plot (boolean, default True): Whether or not to also plot model group performance
                and thresholding details at this time.
        """
        new_filters = [
            {
                "metric": metric,
                "parameter": parameter,
                "max_from_best": max_from_best,
                "threshold_value": threshold_value,
            }
        ]
        self.update_metric_filters(new_filters)

    def update_metric_filters(self, new_filters=None, plot=True):
        """Update the thresholding metric filters

        Args:
            new_filters (list): A list of metrics to filter model
                groups on, and how to filter them. This is an identical format to
                the list given to 'initial_metric_filters' in the constructor.
                Each entry should be a dict with the keys:
initial_metric_filters
                    metric (string) -- model evaluation metric, such as 'precision@'
                    parameter (string) -- model evaluation metric parameter,
                        such as '300_abs'
                    max_below_best (float) The maximum value that the given metric
                        can be below the best for a given train end time
                    threshold_value (float) The threshold value that the given metric can be
            plot (boolean, default True): Whether or not to also plot model group performance
                and thresholding details at this time.
        """
        logger.debug("Updating metric filters with new config %s", new_filters)
        self.model_group_thresholder.update_filters(new_filters)
        if plot:
            logger.debug("After config update, plotting model groups")
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
                regret_metric=metric_definition["metric"],
                regret_parameter=metric_definition["parameter"],
                model_group_ids=self.thresholded_model_group_ids,
                train_end_times=self.train_end_times[:-1],
                # We can't calculate regrets for the most recent train end time,
                # so don't send that in. Assumes that the train_end_times
                # are sorted in the constructor
            )
            self.selection_rule_plotter.plot_all_selection_rules(**common_kwargs)

            df = self.selection_rule_performance_plotter.generate_plot_data(
                **common_kwargs
            )
            self.selection_rule_performance_plotter.regret_plot_from_dataframe(
                metric=metric_definition["metric"],
                parameter=metric_definition["parameter"],
                df=df,
            )
            self.selection_rule_performance_plotter.raw_next_time_plot_from_dataframe(
                metric=metric_definition["metric"],
                parameter=metric_definition["parameter"],
                df=df,
            )

            key = metric_definition["metric"] + metric_definition["parameter"]
            self.results_for_rule[key] = df

    def register_selection_rule_grid(self, rule_grid, plot=True):
        """Register a grid of selection rules

        Args:
            rule_grid (list): Groups of selection rules that share parameters. See documentation below for schema.
            plot: (boolean, defaults to True) Whether or not to plot the selection
                rules at this time.

        `rules_grid` is a list of dicts. Each dict, which defines a group, has two required keys:
        `shared_parameters` and `selection_rules`.

        `shared_parameters`: A list of dicts, each with a set of parameters that are taken
        by all selection rules in this group.

        For each of these shared parameter sets, the grid will create selection rules
        combining the set with all possible selection rule/parameter combinations.

        This can be used to quickly combine, say, a variety of rules that
        all are concerned with precision at top 100 entities.

        `selection_rules`: A list of dicts, each with:

        - A 'name' attribute that matches a selection rule in audition.selection_rules
        - Parameters and values taken by that selection rule. Values in list form are
        all added to the grid. If the selection rule has no parameters, or the parameters are all covered
        by the shared parameters in this group, none are needed here.

        Each selection rule in the group must have all of its required parameters
        covered by the shared parameters in its group and the parameters given to it.

        Refer to [Selection Rules](../selection_rules/#selection-rules) for available selection rules
        and their parameters.
        The exceptions are the first two arguments to each selection rule,
        'df' and 'train_end_time'.
        These are contextual and thus provided internally by Audition.

        Example:
        ```
        [{
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
        ```
        """
        self.selection_rules = make_selection_rule_grid(rule_grid)
        if plot:
            self.plot_selection_rules()


class AuditionRunner:
    def __init__(self, config_dict, db_engine, directory=None):
        self.dir = directory
        self.config = config_dict
        self.db_engine = db_engine

    def run(self):
        pre_aud = PreAudition(self.db_engine)
        model_group_ids = pre_aud.get_model_groups(self.config["model_groups"]["query"])
        query_end_times = self.config["time_stamps"]["query"].format(
            ", ".join(map(str, model_group_ids))
        )
        end_times = pre_aud.get_train_end_times(query=query_end_times)

        aud = Auditioner(
            db_engine=self.db_engine,
            model_group_ids=model_group_ids,
            train_end_times=end_times,
            initial_metric_filters=[
                {
                    "metric": self.config["filter"]["metric"],
                    "parameter": self.config["filter"]["parameter"],
                    "max_from_best": self.config["filter"]["max_from_best"],
                    "threshold_value": self.config["filter"]["threshold_value"],
                }
            ],
            models_table=self.config["filter"]["models_table"],
            distance_table=self.config["filter"]["distance_table"],
            directory=self.dir,
            agg_type=self.config["filter"].get("agg_type") or 'worst',
        )

        aud.plot_model_groups()
        aud.register_selection_rule_grid(rule_grid=self.config["rules"], plot=True)
        aud.save_result_model_group_ids()

        logger.debug(f"Audition ran! Results are stored in {self.dir}.")

    def validate(self):
        try:
            logger.debug("Validate!")
        except Exception as err:
            raise err
