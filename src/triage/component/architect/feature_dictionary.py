from triage.component.architect.utils import remove_schema_from_table_name
from triage.util.structs import FeatureNameList
from collections.abc import Iterable
from collections import MutableMapping


class FeatureDictionary(MutableMapping):
    """A feature dictionary, consisting of table names as keys and column names as values

    If a list of feature_blocks is passed, will initialize the feature dictionary with their data.
    """
    def __init__(self, feature_blocks=None, *args, **kwargs):
        self.tables = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        for feature_block in feature_blocks:
            cleaned_table = remove_schema_from_table_name(
                feature_block.final_feature_table_name
            )
            self[cleaned_table] = feature_block.feature_columns

    def __getitem__(self, key):
        return FeatureNameList(self.tables[key])

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
        self.tables[table] = feature_names

    def __delitem__(self, key):
        del self.tables[key]

    def __iter__(self):
        return iter(self.tables)

    def __len__(self):
        return len(self.tables)

    def __sub__(self, other):
        not_in_other = FeatureDictionary()
        for table_name, feature_list in self.items():
            if table_name not in other:
                not_in_other[table_name] = feature_list
                continue
            missing_feature_names = [
                feature_name
                for feature_name in feature_list
                if feature_name not in other[table_name]
            ]
            if missing_feature_names:
                not_in_other[table_name] = missing_feature_names
        return not_in_other

    def __repr__(self):
        return str(self.tables)
