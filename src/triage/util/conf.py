import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import re

from dateutil.relativedelta import relativedelta
from datetime import datetime



def parse_from_obj(config, alias):
    """
    Parses a from_obj configuration key. If it's a from_obj_table just returns it.
    If it's a from_obj_query creates the sub_query with alias
    Args:
        config: the yaml dict
        alias: the name of the alias if there's a from_obj_query

    Returns:

    """
    from_obj = config.get("from_obj_table", None)
    if not from_obj:
        from_obj = config.get("from_obj_query", None)
        return " ({}) {} ".format(from_obj, alias) if from_obj else None
    return from_obj

def dt_from_str(dt_str):
    if isinstance(dt_str, datetime):
        return dt_str
    return datetime.strptime(dt_str, "%Y-%m-%d")


def parse_delta_string(delta_string):
    """Given a string in a postgres interval format (e.g., '1 month'),
    parse the units and value from it.

    Assumptions:
    - The string is in the format 'value unit', where
      value is an int and unit is one of year(s), month(s), day(s),
      week(s), hour(s), minute(s), second(s), microsecond(s), or an
      abbreviation matching y, d, w, h, m, s, or ms (case-insensitive).
      For example: 1 year, 1year, 2 years, 1 y, 2y, 1Y.

    :param delta_string: the time interval to convert
    :type delta_string: str

    :return: time units, number of units (value)
    :rtype: tuple

    :raises: ValueError if the delta_string is not in the expected format

    """
    match = parse_delta_string.pattern.search(delta_string)
    if match:
        (pre_value, units) = match.groups()
        return (units, int(pre_value))

    raise ValueError(
        "Could not parse value from time delta string: {!r}".format(delta_string)
    )


parse_delta_string.pattern = re.compile(r"^(\d+) *([^ ]+)$")


def convert_str_to_relativedelta(delta_string):
    """Given a string in a postgres interval format (e.g., '1 month'),
    convert it to a dateutil.relativedelta.relativedelta.

    Assumptions:
    - The string is in the format 'value unit', where
      value is an int and unit is one of year(s), month(s), day(s),
      week(s), hour(s), minute(s), second(s), microsecond(s), or an
      abbreviation matching y, d, w, h, m, s, or ms (case-insensitive).
      For example: 1 year, 1year, 2 years, 1 y, 2y, 1Y.

    :param delta_string: the time interval to convert
    :type delta_string: str

    :return: the time interval as a relativedelta
    :rtype: dateutil.relativedelta.relativedelta

    :raises: ValueError if the delta_string is not in the expected format

    """
    (units, value) = parse_delta_string(delta_string)

    verbose_match = convert_str_to_relativedelta.pattern_verbose.search(units)
    if verbose_match:
        unit_type = verbose_match.group(1) + "s"
        return relativedelta(**{unit_type: value})

    try:
        unit_type = convert_str_to_relativedelta.brief_units[units.lower()]
    except KeyError:
        pass
    else:
        if unit_type == "minutes":
            logger.warning(
                f'Time delta units "{units}" converted to minutes.'
            )
        return relativedelta(**{unit_type: value})

    raise ValueError("Could not handle units. Units: {} Value: {}".format(units, value))


convert_str_to_relativedelta.pattern_verbose = re.compile(
    r"^(year|month|day|week|hour|minute|second|microsecond)s?$"
)

convert_str_to_relativedelta.brief_units = {
    "y": "years",
    "d": "days",
    "w": "weeks",
    "h": "hours",
    "m": "minutes",
    "s": "seconds",
    "ms": "microseconds",
}
