import functools
import operator

import sqlalchemy

from triage.util.structs import FeatureNameList


def str_in_sql(values):
    return ",".join(map(lambda x: "'{}'".format(x), values))


def feature_list(feature_dictionary):
    """Convert a feature dictionary to a sorted list

    Args: feature_dictionary (dict)

    Returns: sorted list of feature names
    """
    if not feature_dictionary:
        return FeatureNameList()
    return FeatureNameList(sorted(
        functools.reduce(
            operator.concat,
            (feature_dictionary[key] for key in feature_dictionary.keys()),
        )
    ))


def retry_if_db_error(exception):
    return isinstance(exception, sqlalchemy.exc.OperationalError)
