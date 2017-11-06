from dateutil.relativedelta import relativedelta
import warnings
import re
from six import string_types


def convert_to_list(x):
    """Given an object, if it is not a list, convert it to a list.

    :param x: an object to be converted to a list
    :type x: object

    :return: x as a list
    :rtype: list
    """
    if isinstance(x, string_types): x = [x]
    else:
        try: iter(x)
        except TypeError: x = [x]
        else: x = list(x)
    return(x)


def convert_str_to_relativedelta(delta_string):
    """ Given a string in a postgres interval format (e.g., '1 month'),
    convert it to a dateutil.relativedelta.relativedelta.

    Assumptions:
    - the string is in the format 'value unit', where
      value is an int and unit is one of year(s), month(s), day(s),
      week(s), hour(s), minute(s), second(s), microsecond(s).

    :param delta_string: the time interval to convert
    :type delta_string: str

    :return: the time interval as a relativedelta
    :rtype: dateutil.relativedelta.relativedelta

    :raises: ValueError if the delta_string is not in the expected format
    """
    units, value = parse_delta_string(delta_string)

    if units in ['year', 'years']:
        delta = relativedelta(years=value)
    elif units in ['month', 'months']:
        delta = relativedelta(months=value)
    elif units in ['day', 'days']:
        delta = relativedelta(days=value)
    elif units in ['week', 'weeks']:
        delta = relativedelta(weeks=value)
    elif units in ['hour', 'hours']:
        delta = relativedelta(hours=value)
    elif units in ['minute', 'minutes']:
        delta = relativedelta(minutes=value)
    elif units in ['second', 'seconds']:
        delta = relativedelta(seconds=value)
    elif units == 'microsecond' or units == 'microseconds':
        delta = relativedelta(microseconds=value)
    else:
        raise ValueError(
            'Could not handle units. Units: {} Value: {}'.format(units, value)
        )

    return(delta)


def parse_delta_string(delta_string):
    """ Given a string in a postgres interval format (e.g., '1 month'),
    parse the units and value from it.

    Assumptions:
    - the string is in the format 'value unit', where
      value is an int and unit is one of year(s), month(s), day(s),
      week(s), hour(s), minute(s), second(s), microsecond(s).

    :param delta_string: the time interval to convert
    :type delta_string: str

    :return: time units, number of units (value)
    :rtype: tuple

    :raises: ValueError if the delta_string is not in the expected format
    """
    if len(delta_string.split(' ')) == 2:
        units = delta_string.split(' ')[1]
        try:
            value = int(delta_string.split(' ')[0])
        except:
            raise ValueError(
                'Could not parse value from time delta string: {}'.format(
                    delta_string
                )
            )
    else:
        delta_parts = re.split('([a-zA-Z]*)', delta_string)
        try:
            units = delta_parts[1]
        except:
            raise ValueError(
                'No units in time delta string {}'.format(delta_string)
            )
        try:
            value = int(delta_parts[0])
        except:
            raise ValueError(
                'Could not parse value from time delta string: {}'.format(
                    delta_string
                )
            )
    return(units, value)

