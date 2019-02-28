from triage.component.architect.utils import remove_schema_from_table_name
from triage.util.structs import FeatureNameList
from collections.abc import Iterable


class FeatureDictionary(dict):
    """A feature dictionary, consisting of table names as keys and column names as values

    If a list of feature_blocks is passed, will initialize the feature dictionary with their data.
    """
    def __init__(self, feature_blocks=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for feature_block in feature_blocks:
            cleaned_table = remove_schema_from_table_name(
                feature_block.final_feature_table_name
            )
            self[cleaned_table] = feature_block.feature_columns

    def __setitem__(self, table, feature_names):
        if not isinstance(table, str):
            raise ValueError("key of FeatureDictionary objects represents a table "
                             "name and must be a string")
        if not isinstance(feature_names, Iterable):
            raise ValueError("value of FeatureDictionary objects represents a list of "
                             "feature names, and therefore must be iterable")
        for feature_name in feature_names:
            if not isinstance(feature_name, str):
                raise ValueError(f"invalid value: {feature_name}. "
                                 f"invalid type: {type(feature_name)} "
                                 "The value of FeatureDictionary objects represents a list of "
                                 "feature names, and therefore each item must be a string")
        if isinstance(feature_names, FeatureNameList):
            super().__setitem__(table, feature_names)
        else:
            super().__setitem__(table, FeatureNameList(feature_names))
