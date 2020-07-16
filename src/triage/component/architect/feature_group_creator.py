import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from triage.util.structs import FeatureNameList


class FeatureGroup(dict):
    def __init__(self, *args, **kwargs):
        try:
            self._names = [kwargs.pop("name")]
        except KeyError:
            self._names = []
        try:
            features = kwargs.pop("features_by_table")
        except KeyError:
            features = {}
        super(FeatureGroup, self).__init__(*args, **features)

    @property
    def names(self):
        return self._names

    def update(self, other_group):
        super(FeatureGroup, self).update(other_group)
        self._names += other_group.names


def table_subsetter(config_item, table, features):
    "Return features matching a given table"
    if table == config_item:
        return features
    else:
        return []


def prefix_subsetter(config_item, table, features):
    "Return features matching a given prefix"
    return [feature for feature in features if feature.startswith(config_item)]

def all_subsetter(config_item, table, features):
    return features


class FeatureGroupCreator:
    """Divides a feature dictionary into groups based on given criteria"""

    subsetters = {
        "tables": table_subsetter,
        "prefix": prefix_subsetter,
        "all": all_subsetter
    }

    def __init__(self, definition):
        """
        Args:
            definition (dict) rules for generating feature groups
                Each key must correspond to a key in self.subsetters
                Each value (a list) must be understood by the subsetter
        """
        self.definition = definition

    def validate(self):
        """Validate the object's configuration

        """
        if not isinstance(self.definition, dict):
            raise ValueError("Feature Group Definition must be a dictionary")

        for subsetter_name, value in self.definition.items():
            if subsetter_name not in self.subsetters:
                raise ValueError(f"Unknown subsetter {subsetter_name} received")
            if not hasattr(value, "__iter__") or isinstance(value, (str, bytes)):
                raise ValueError(
                    "Each value in FeatureGroupCreator must be "
                    "iterable and not a string"
                )

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
        logger.spam(
            f"Creating feature groups, using: {self.definition}, Master feature dictionary: {feature_dictionary}",
        )
        subsets = []
        for name, config in sorted(self.definition.items()):
            logger.spam(f"Parsing config grouping method {name}, items {config}")
            for config_item in config:
                subset = FeatureGroup(name=f"{name}: {config_item}")
                logger.spam(f"Finding columns that might belong in [{name}: {config_item}]")
                for table, features in feature_dictionary.items():
                    logger.spam(
                        f"Searching features in table {table} that match group {subset}"
                    )
                    matching_features = self.subsetters[name](
                        config_item, table, features
                    )
                    logger.debug(
                        f"Found {len(matching_features)} matching features in table {table} that match group [{name}: {config_item}]",
                    )
                    if len(matching_features) > 0:
                        subset[table] = FeatureNameList(matching_features)

                subsets.append(subset)

        if not any(subset for subset in subsets if any(subset)):
            raise ValueError(
                f"Problem! The feature group definition {self.definition} did not find any matches",
                f"in feature dictionary {feature_dictionary}"
            )
        logger.verbose(f"Found {len(subsets)} total feature subsets")
        return subsets
