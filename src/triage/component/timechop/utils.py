from six import string_types


def convert_to_list(x):
    """Given an object, if it is not a list, convert it to a list.

    :param x: an object to be converted to a list
    :type x: object

    :return: x as a list
    :rtype: list

    """
    if isinstance(x, string_types):
        return [x]

    try:
        iter(x)
    except TypeError:
        return [x]
    else:
        return list(x)
