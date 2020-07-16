from itertools import product
import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from .selection_rules import BoundSelectionRule
from .utils import make_list


def _expand_param_sets(rule_instances, values):
    expanded_param_sets = []
    for v in product(*values):
        params = dict(zip(rule_instances, v))
        expanded_param_sets.append(params)
    return expanded_param_sets


def _build_rule_arguments(expanded_param_set, shared_param_set):
    rule_args = {}
    for key, value in expanded_param_set.items():
        rule_args[key] = value
    for key, value in shared_param_set.items():
        rule_args[key] = value
    return rule_args


def _bound_rules_from(shared_param_set, selection_rule):
    rules = []
    rule_instances_and_values = [
        (key, make_list(value)) for key, value in selection_rule.items()
    ]
    rule_instances, values = zip(*rule_instances_and_values)

    for expanded_param_set in _expand_param_sets(rule_instances, values):
        function_name = expanded_param_set["name"]
        del expanded_param_set["name"]
        rules.append(
            BoundSelectionRule(
                function_name=function_name,
                args=_build_rule_arguments(expanded_param_set, shared_param_set),
            )
        )
    return rules


def make_selection_rule_grid(rule_groups):
    """Convert a compact selection rule group representation to a
        list of bound selection rules.

    Arguments:
        rule_groups (list): List of dicts used to specify selection rule grid. 
        
    Most users will want to use [rulemaker objects](#rulemakers)
    to generate their `rule_group` specifications.
    
    An example rule_groups specification:

    ```
    [{
            'shared_parameters': [
                    {'metric': 'precision@', 'parameter': '100_abs'},
                    {'metric': 'recall@', 'parameter': '100_abs'},
                ],
                'selection_rules': [
                    {'name': 'most_frequent_best_dist', 'dist_from_best_case': [0.1, 0.2, 0.3]},
                    {'name': 'best_current_value'}
                ]
        }, {
            'shared_parameters': [
                {'metric1': 'precision@', 'parameter1': '100_abs'},
            ],
            'selection_rules': [
                {
                    'name': 'best_average_two_metrics',
                    'metric2': ['recall@'],
                    'parameter2': ['100_abs'],
                    'metric1_weight': [0.4, 0.5, 0.6]
                },
            ]
        }]
    ```
    Returns:
        list: list of audition.selection_rules.BoundSelectionRule objects"""


    rules = []
    logger.debug("Expanding selection rule groups into full grid")
    for rule_group in rule_groups:
        logger.debug("Expanding rule group %s", rule_group)
        for shared_param_set, selection_rule in product(
            rule_group["shared_parameters"], rule_group["selection_rules"]
        ):
            logger.debug(
                "Expanding shared param set %s and selection rules %s",
                shared_param_set,
                selection_rule,
            )
            new_rules = _bound_rules_from(shared_param_set, selection_rule)
            logger.debug("Found %s new rules", len(new_rules))
            rules += new_rules
    logger.debug(
        "Found %s total selection rules. Full list: %s",
        len(rules),
        [rule.descriptive_name for rule in rules],
    )
    return rules
