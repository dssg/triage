import functools
import operator


def feature_list(feature_dictionary):
    """Convert a feature dictionary to a sorted list.

    Args: feature_dictionary (dict)

    Returns: sorted list of feature names

    """
    return sorted(
        functools.reduce(
            operator.concat,
            (feature_dictionary[key] for key in feature_dictionary.keys())
        )
    )


def str_in_sql(values):
    return ','.join("'{}'".format(value) for value in values)
