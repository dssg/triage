def table_subsetter(config_item, table, features):
    "Return features matching a given table"
    if table == config_item:
        return features
    else:
        return []


def prefix_subsetter(config_item, table, features):
    "Return features matching a given prefix"
    return [
        feature
        for feature in features
        if feature.split('_')[0] == config_item
    ]


def all_subsetter(config_item, table, features):
    return features


class FeatureGroupCreator(object):
    """Divides a feature dictionary into groups based on given criteria"""
    subsetters = {
        'tables': table_subsetter,
        'prefix': prefix_subsetter,
        'all': all_subsetter,
    }

    def __init__(self, definition):
        """
        Args:
            definition (dict) rules for generating feature groups
                Each key must correspond to a key in self.subsetters
                Each value (a list) must be understood by the subsetter
        """
        self.definition = definition

        for subsetter in self.definition.keys():
            if subsetter not in self.subsetters:
                raise KeyError('Unknown subsetter %s received', subsetter)

    def subsets(self, feature_dictionary):
        """Generate subsets of a feature dict

        Args:
            feature_dictionary (dict) tables and the features contained in each

            The feature dictionary is meant to be keyed on source table. Example:

            {
                'feature_table_one': ['feature_one', feature_two'],
                'feature_table_two': ['feature_three', 'feature_four'],
            }

        Returns: (list) subsets of the feature dictionary, in the same
            table-based structure
        """
        subsets = []
        for name, config in sorted(self.definition.items()):
            for config_item in config:
                subset = {}
                for table, features in feature_dictionary.items():
                    matching_features =\
                        self.subsetters[name](config_item, table, features)
                    if len(matching_features) > 0:
                        subset[table] = matching_features
                subsets.append(subset)
        if not any(subset for subset in subsets if any(subset)):
            raise Exception('No matching feature subsets found!')
        return subsets
