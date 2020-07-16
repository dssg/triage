import numpy as np
import pandas as pd
from six import string_types

from triage.component.catwalk.exceptions import BaselineFeatureNotInMatrix

OPERATOR_METHODS = {">": "gt", ">=": "ge", "<": "lt", "<=": "le", "==": "eq"}
REQUIRED_KEYS = frozenset(["feature_name", "operator", "threshold"])


def get_operator_method(operator_string):
    """ Convert the user-passed operator into the the name of the apprpriate
    pandas method.
    """
    try:
        operator_method = OPERATOR_METHODS[operator_string]
    except KeyError:
        raise ValueError(
            (
                "Operator '{operator}' extracted from rule is not a "
                "supported operator ({supported_operators}).".format(
                    operator=operator_string,
                    supported_operators=OPERATOR_METHODS.keys(),
                )
            )
        )

    return operator_method


class SimpleThresholder:
    """ The simple thresholder applies a set of predetermined logical rules to a
    test matrix to classify entities. By default, it will classify entities as 1
    if they satisfy any of the rules. When 'and' is set as the logical_operator,
    it will classify entities as 1 only if they pass *all* of the rules.

    Rules are passed as either strings in the format 'x1 > 5' or dictionaries in
    the format {feature_name: 'x1', operator: '>', threshold: 5}. The
    feature_name, operator, and threshold keys are required. Eventually, this
    class may be abstracted into a BaseThreshold class and more complicated
    thresholders could be built around new keys in the dictionaries (e.g., by
    specifying scores that could be applied (and possibly summed) to entities
    satisfying rules) or by an alternative dictionary format that specifies
    more complicated structures for applying rules (for example:
        {
            or: [
                {or: [{}, {}]},
                {and: [{}, {}]}
            ]
        }
    where rules and operators that combine them can be nested).
    """

    def __init__(self, rules, logical_operator="or"):
        self.rules = rules
        self.logical_operator = logical_operator
        self.feature_importances_ = None
        self.rule_combination_method = self.lookup_rule_combination_method(
            logical_operator
        )

    @property
    def rules(self):
        return vars(self)["rules"]

    @rules.setter
    def rules(self, rules):
        """ Validates the rules passed by the user and converts them to the
        internal representation. Can be used to validate rules before running an
        experiment.

        1. If rules are not a list, make them a list.
        2. If rules are strings, convert them to dictionaries.
        3. If dictionaries or strings are not in a supported format, raise
           helpful exceptions.
        """
        if not isinstance(rules, list):
            rules = [rules]

        converted_rules = []
        for rule in rules:
            if isinstance(rule, string_types):
                converted_rules.append(self._convert_string_rule_to_dict(rule))
            else:
                if not isinstance(rule, dict):
                    raise ValueError(
                        (
                            'Rule "{rule}" is not of a supported type (string or '
                            "dict).".format(rule=rule)
                        )
                    )
                if not rule.keys() >= REQUIRED_KEYS:
                    raise ValueError(
                        (
                            'Rule "{rule}" missing one or more required keys '
                            "({required_keys}).".format(
                                rule=rule, required_keys=REQUIRED_KEYS
                            )
                        )
                    )
                rule["operator"] = get_operator_method(rule["operator"])
                converted_rules.append(rule)

        vars(self)["rules"] = converted_rules

    @property
    def all_feature_names(self):
        return [rule["feature_name"] for rule in self.rules]

    def lookup_rule_combination_method(self, logical_operator):
        """ Convert 'and' to 'all' and 'or' to 'any' for interacting with
        pandas DataFrames.
        """
        rule_combination_method_lookup = {"or": "any", "and": "all"}
        return rule_combination_method_lookup[logical_operator]

    def _convert_string_rule_to_dict(self, rule):
        """ Converts a string rule into a dict, raising helpful exceptions if it
        cannot be parsed.
        """
        components = rule.rsplit(" ", 2)

        if len(components) < 3:
            raise ValueError(
                (
                    '{required_keys} could not be parsed from rule "{rule}". Are '
                    "they all present and separated by spaces?".format(
                        required_keys=REQUIRED_KEYS, rule=rule
                    )
                )
            )

        try:
            threshold = int(components[2])
        except ValueError:
            raise ValueError(
                (
                    'Threshold "{threshold}" parsed from rule "{rule}" is not an '
                    "int.".format(threshold=components[2], rule=rule)
                )
            )

        operator = get_operator_method(components[1])

        return {
            "feature_name": components[0],
            "operator": operator,
            "threshold": threshold,
        }

    def _set_feature_importances_(self, x):
        """ Assigns feature importances following the rule: 1 for the features
        we are thresholding on, 0 for all other features.
        """
        feature_importances = [0] * len(x.columns)
        for feature_name in self.all_feature_names:
            try:
                position = x.columns.get_loc(feature_name)
            except KeyError:
                raise BaselineFeatureNotInMatrix(
                    (
                        "Rules refer to a feature ({feature_name}) not included in "
                        "the training matrix!".format(feature_name=feature_name)
                    )
                )
            feature_importances[position] = 1
        self.feature_importances_ = np.array(feature_importances)

    def fit(self, x, y):
        """ Set feature importances and return self.
        """
        self._set_feature_importances_(x)
        return self

    def predict_proba(self, x):
        """ Assign 1 for entities that meet the rules and 0 for those that do not.
        """
        rule_evaluations_list = []
        for rule in self.rules:
            rule_evaluations_list.append(
                getattr(x[rule["feature_name"]], rule["operator"])(rule["threshold"])
            )
        rule_evaluations_dataframe = pd.concat(rule_evaluations_list, axis=1)
        scores = getattr(rule_evaluations_dataframe, self.rule_combination_method)(
            axis=1
        )
        scores = list(scores.astype(int))

        # format it like sklearn output and return
        return np.array([scores, scores]).transpose()
