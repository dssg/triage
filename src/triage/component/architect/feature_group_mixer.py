import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import itertools
from triage.component.architect.feature_group_creator import FeatureGroup


def leave_one_in(feature_groups):
    """For each group, return a copy of just that group

    Args:
        feature_groups (list) The feature groups to apply the strategy to

    Returns: A list of feature dicts
    """
    return feature_groups


def leave_one_out(feature_groups):
    """For each group, return a copy of all groups excluding that group

    Args:
        feature_groups (list) The feature groups to apply the strategy to

    Returns: A list of feature dicts
    """
    results = []
    for index_to_exclude in range(0, len(feature_groups)):
        group_copy = feature_groups.copy()
        del group_copy[index_to_exclude]
        feature_dict = FeatureGroup()
        for group in group_copy:
            feature_dict.update(group)
        results.append(feature_dict)
    return results


def all_combinations(feature_groups):
    """Return all combinations of groups, excluding repeated groups

    Args:
        feature_groups (list) The feature groups to apply the strategy to

    Returns: A list of feature dicts
    """
    results = []
    for number_feature_groups in range(len(feature_groups) + 1):
        for combo in itertools.combinations(feature_groups,
                                            number_feature_groups):
            feature_dict = FeatureGroup()
            for group_element in combo:
                feature_dict.update(group_element)
            if feature_dict:
                results.append(feature_dict)
    return results


def all_features(feature_groups):
    """Return a combination of all feature groups

    Args:
        feature_groups (list) The feature groups to apply the strategy to

    Returns: A list of feature dicts
    """
    feature_dict = FeatureGroup()
    for group in feature_groups:
        feature_dict.update(group)
    return [feature_dict]


class FeatureGroupMixer:
    """Generates different combinations of feature groups
    based on a list of strategies"""

    strategy_lookup = {
        "leave-one-out": leave_one_out,
        "leave-one-in": leave_one_in,
        "all-combinations":  all_combinations,
        "all": all_features,
    }

    def __init__(self, strategies):
        for strategy in strategies:
            if strategy not in self.strategy_lookup:
                raise ValueError('Unknown strategy "{}"'.format(strategy))
        self.strategies = strategies

    def generate(self, feature_groups):
        """Apply all strategies to the list of feature groups

        Args:
            feature_groups (list) A list of feature dictionarys,
                each representing a group
        Returns: (list) of feature dictionaries
        """
        final_results = []
        for strategy in self.strategies:
            logger.debug(f"Mixing feature groups using strategy {strategy}")
            results = self.strategy_lookup[strategy](feature_groups)
            logger.spam("Mixing found new feature groups combinations [{results}]")
            final_results += results

        return final_results
