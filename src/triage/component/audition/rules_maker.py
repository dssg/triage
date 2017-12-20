from itertools import chain

class BaseRules(object):
    def __init__(self):
        self._metric = None
        self._parameter = None
        self.shared_parameters = []
        self.selection_rules = []

    def _is_parameters_existed(self, params_dict):
        return params_dict in self.shared_parameters

    def _is_selection_rule_exisited(self, rule_dict):
        return rule_dict in self.selection_rules

    def _append(self, params_dict, rule_dict):
        if not self._is_parameters_existed(params_dict):
            self.shared_parameters.append(params_dict)
        if not self._is_selection_rule_exisited(rule_dict):
            self.selection_rules.append(rule_dict)

    def create(self):
        return [{
            'shared_parameters': self.shared_parameters,
            'selection_rules': self.selection_rules
        }]


class SimpleRuleMaker(BaseRules):
    def add_rule_best_current_value(self, metric=None, parameter=None):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {'metric': self._metric, 'parameter': self._parameter}
        rule_dict = {'name': 'best_current_value'}
        self._append(params_dict, rule_dict)

    def add_rule_best_average_value(self, metric=None, parameter=None):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {'metric': self._metric, 'parameter': self._parameter}
        rule_dict = {'name': 'best_average_value'}
        self._append(params_dict, rule_dict)

    def add_rule_lowest_metric_variance(self, metric=None, parameter=None):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
        params_dict = {'metric': metric, 'parameter': parameter}
        rule_dict = {'name': 'lowest_metric_variance'}
        self._append(params_dict, rule_dict)

    def add_rule_most_frequent_best_dist(self, metric=None, parameter=None,
                                         dist_from_best_case=[0.01, 0.05, 0.1, 0.15]):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {'metric': metric, 'parameter': parameter}
        rule_dict = {
            'name': 'most_frequent_best_dist',
            'dist_from_best_case': distance_from_best
        }
        self.append(params_dict, rule_dict)

    def add_rule_best_avg_recency_weight(self, metric=None, parameter=None,
                                         curr_weight=[1.5, 2.0, 5.0], decay_type=['linear']):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {'metric': metric, 'parameter': parameter}
        rule_dict = {
            'name': 'best_avg_recency_weight',
            'curr_weight': curr_weight,
            'decay_type': decay_type
        }
        self._append(params_dict, rule_dict)

    def add_rule_best_avg_var_penalized(self, metric=None, parameter=None, stdev_penalty=0.5):
        if metric is not None:
            self._metric = metric
        if parameter is not None:
            self._parameter = parameter
        params_dict = {'metric': metric, 'parameter': parameter}
        rule_dict = {
            'name': 'best_avg_var_penalized',
            'stdev_penalty': stdev_penalty
        }
        self._append(params_dict, rule_dict)


class RandomGroupRuleMaker(BaseRules):
    def __init__(self):
        self.shared_parameters = [{}]
        self.selection_rules = [{'name': 'random_model_group'}]


class TwoMetricsRuleMaker(BaseRules):
    def add_rule_best_average_two_metrics(self, metric1='precision@', parameter1='100_abs',
                                          metric2='recall@', parameter2='300_abs', metric1_weight=0.5):
        params_dict = {
            'metric1': metric1,
            'parameter1': parameter1,
            'metric2': metric2,
            'parameter2': parameter2}
        rule_dict = {'name': 'best_average_two_metrics', 'metric1_weight': metric1_weight}
        self._append(params_dict, rule_dict)


def create_selection_grid(*args):
    return list(chain(*args))
