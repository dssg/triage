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
        feature_dict = {}
        for group in group_copy:
            feature_dict.update(group)
        results.append(feature_dict)
    return results


def all_features(feature_groups):
    """Return a combination of all feature groups

    Args:
        feature_groups (list) The feature groups to apply the strategy to

    Returns: A list of feature dicts
    """
    feature_dict = {}
    for group in feature_groups:
        feature_dict.update(group)
    return [feature_dict]


class FeatureGroupMixer(object):
    """Generates different combinations of feature groups
    based on a list of strategies"""
    strategy_lookup = {
        'leave-one-out': leave_one_out,
        'leave-one-in': leave_one_in,
        'all': all_features,
    }

    def __init__(self, strategies):
        for strategy in strategies:
            if strategy not in self.strategy_lookup:
                raise ValueError('Unknown strategy "{}"'.format(strategy))
        self.strategies = strategies

    def generate(self, feature_groups):
        """Apply all strategies to the list of feature groups

        Args:
            feature_groups (list) A list of feature dicts,
                each representing a group
        Returns: (list) of feature dicts
        """
        results = []
        for strategy in self.strategies:
            results += self.strategy_lookup[strategy](feature_groups)
        return results
