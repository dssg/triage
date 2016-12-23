# -*- coding: utf-8 -*-

def categorical(col, op, choices, include_null=True, maxlen=32):
    """
    ``categorical(col, op, choices, include_null=True, maxlen=32)``

    Args:
        col: the column name (or equivalent SQL expression)
        op: the SQL operation (e.g., '=' or '~' or 'LIKE')
        choices: A list or dictionary of values. When a dictionary is passed,
            the keys are a short name for the value.
        include_null: Should an extra `{col} is NULL` be added? (default True)
        maxlen: The maximum length of aggregate quantity names (default 32).
            Names longer than this will be truncated.

    Returns: a dictionary of aggregate quantities to be passed to Aggregate()

    A simple helper method to easily create many categorical columns from one
    source column by comparing it against many values. It effectively creates
    many quantities of the form "({col} {op} '{elt}')::INT" for elt in choices.
    The type of the comparison is converted to an integer so it can easily be
    used with 'sum' (for total count) and 'avg' (for relative fraction)
    aggregate functions.

    By default, the aggregates are simply named "{col}_{op}_{choice}", but
    that can easily get long and exceed the maximum column name length. If any
    name ends up longer than ``maxlen`` characters (32 by default), then each
    aggregate name gets truncated with a sequential number appended to ensure
    that they remain identifiable and unique (but note that ordering is not
    preserved).
    """

    if type(choices) is not dict:
        choices = {k: k for k in choices}
    d = {'{}_{}_{}'.format(col, op, nickname): "({} {} '{}')::INT".format(col, op, choice)
         for nickname, choice in choices.items()}
    if include_null:
        d['{}__NULL'.format(col)] = '({} is NULL)::INT'.format(col)
    if any(len(k) > maxlen for k in d.keys()):
        for i, k in enumerate(d.keys()):
            d['%s_%02d' % (k[:maxlen-3], i)] = d.pop(k)
    return d
