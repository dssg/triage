from six import string_types


def convert_to_list(x):
    """
    Given an object, if it is not a list, convert it to a list.

    Arguments:
        x (object): an object to be converted to a list

    return:
        list: x as a list
    """
    if isinstance(x, string_types):
        return [x]

    try:
        iter(x)
    except TypeError:
        return [x]
    else:
        return list(x)
