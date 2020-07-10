class BaseRules:
    def __init__(self):
        self._metric = None
        self._parameter = None
        self.shared_parameters = []
        self.selection_rules = []

    def _does_parameters_exist(self, params_dict):
        return params_dict in self.shared_parameters

    def _does_selection_rule_exisit(self, rule_dict):
        return rule_dict in self.selection_rules

    def _append(self, params_dict, rule_dict):
        if not self._does_parameters_exist(params_dict):
            self.shared_parameters.append(params_dict)
        if not self._does_selection_rule_exisit(rule_dict):
            self.selection_rules.append(rule_dict)

    def create(self):
        return [
            {
                "shared_parameters": self.shared_parameters,
                "selection_rules": self.selection_rules,
            }
        ]


class SimpleRuleMaker(BaseRules):

    """
    Holds methods that generate parameter grids for selection rules that
    evaluate the performance of a model group in terms of a single metric.
    These include:

    - [best_current_value][triage.component.audition.selection_rules.best_current_value]
    - [best_average_value][triage.component.audition.selection_rules.best_average_value]
    - [lowest_metric_variance][triage.component.audition.selection_rules.lowest_metric_variance]
    - [most_frequent_best_dist][triage.component.audition.selection_rules.most_frequent_best_dist]
    - [best_avg_var_penalized][triage.component.audition.selection_rules.best_avg_var_penalized]
    - [best_avg_recency_weight][triage.component.audition.selection_rules.best_avg_recency_weight]
    """

    def add_rule_best_current_value(self, metric=None, parameter=None, n=1):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {"metric": self._metric, "parameter": self._parameter}
        rule_dict = {"name": "best_current_value", "n": n}
        self._append(params_dict, rule_dict)
        return self.create()

    def add_rule_best_average_value(self, metric=None, parameter=None, n=1):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {"metric": self._metric, "parameter": self._parameter}
        rule_dict = {"name": "best_average_value", "n": n}
        self._append(params_dict, rule_dict)
        return self.create()

    def add_rule_lowest_metric_variance(self, metric=None, parameter=None, n=1):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {"metric": self._metric, "parameter": self._parameter}
        rule_dict = {"name": "lowest_metric_variance", "n": n}
        self._append(params_dict, rule_dict)
        return self.create()

    def add_rule_most_frequent_best_dist(
        self,
        metric=None,
        parameter=None,
        n=1,
        dist_from_best_case=[0.01, 0.05, 0.1, 0.15],
    ):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {"metric": self._metric, "parameter": self._parameter}
        rule_dict = {
            "name": "most_frequent_best_dist",
            "dist_from_best_case": dist_from_best_case,
            "n": n,
        }
        self._append(params_dict, rule_dict)
        return self.create()

    def add_rule_best_avg_recency_weight(
        self,
        metric=None,
        parameter=None,
        n=1,
        curr_weight=[1.5, 2.0, 5.0],
        decay_type=["linear"],
    ):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {"metric": metric, "parameter": parameter}
        rule_dict = {
            "name": "best_avg_recency_weight",
            "curr_weight": curr_weight,
            "decay_type": decay_type,
            "n": n,
        }
        self._append(params_dict, rule_dict)
        return self.create()

    def add_rule_best_avg_var_penalized(
        self, metric=None, parameter=None, stdev_penalty=0.5, n=1
    ):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {"metric": metric, "parameter": parameter}
        rule_dict = {
            "name": "best_avg_var_penalized",
            "stdev_penalty": stdev_penalty,
            "n": n,
        }
        self._append(params_dict, rule_dict)
        return self.create()


class RandomGroupRuleMaker(BaseRules):
    
    """
    The `RandomGroupRuleMaker` class generates a rule that randomly selects `n`
    model groups for each train set.

    Unlike the other two RuleMaker classes, it generates its selection rule spec
    on `__init__`
    """
    
    def __init__(self, n=1):
        self.shared_parameters = [{}]
        self.selection_rules = [{"name": "random_model_group", "n": n}]


class TwoMetricsRuleMaker(BaseRules):

    """
    The `TwoMetricsRuleMaker` class allows for the specification of rules that 
    evaluate a model group's performance in terms of two metrics. It currently
     supports one rule:

     - [best_average_two_metrics][triage.component.audition.selection_rules.best_average_two_metrics]

    """

    def add_rule_best_average_two_metrics(
        self,
        metric1="precision@",
        parameter1="100_abs",
        metric2="recall@",
        parameter2="300_abs",
        metric1_weight=[0.5],
        n=1,
    ):
        params_dict = {"metric1": metric1, "parameter1": parameter1}
        rule_dict = {
            "name": "best_average_two_metrics",
            "metric1_weight": metric1_weight,
            "metric2": [metric2],
            "parameter2": [parameter2],
            "n": n,
        }
        self._append(params_dict, rule_dict)


def create_selection_grid(*args):
    return list(map(lambda r: r.create()[0], args))
