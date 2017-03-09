from datetime import datetime
from dateutil.relativedelta import relativedelta

def convert_str_to_relativedelta(delta_string):
    """ Given a string in a postgres interval format (e.g., '1 month'),
    convert it to a dateutil.relativedelta.relativedelta.

    Assumptions:
    - the prediction_window string is in the format 'value unit', where
      value is an int and unit is one of year(s), month(s), day(s),
      week(s), hour(s), minute(s), second(s), microsecond(s).

    :param delta_string: the time interval to convert
    :type delta_string: str

    :return: the time interval as a relativedelta
    :rtype: dateutil.relativedelta.relativedelta

    :raises: ValueError if the delta_string is not in the expected format
    """
    # split the prediction window string into its component parts
    try:
        units = delta_string.split(' ')[1]
    except:
        raise ValueError('Could not parse units from prediction_window string')

    try:
        value = int(delta_string.split(' ')[0])
    except:
        raise ValueError('Could not parse value from prediction_window string')

    if units == 'year' or units == 'years':
        delta = relativedelta(years = value)
    elif units == 'month' or units == 'months':
        delta = relativedelta(months = value)
    elif units == 'day' or units == 'days':
        delta = relativedelta(days = value)
    elif units == 'week' or units == 'weeks':
        delta = relativedelta(weeks = value)
    elif units == 'hour' or units == 'hours':
        delta = relativedelta(hours = value)
    elif units == 'minute' or units == 'minutes':
        delta = relativedelta(minutes = value)
    elif units == 'second' or units == 'seconds':
        delta = relativedelta(seconds = value)
    elif units == 'microsecond' or units == 'microseconds':
        delta = relativedelta(microseconds = value)
    else:
        raise ValueError(
            'Could not handle units. Units: {} Value: {}'.format(units, value)
        )

    return(delta)
