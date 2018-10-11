def make_list(a):
    return [a] if not isinstance(a, list) else a


def str_in_sql(values):
    return ",".join(map(lambda x: "'{}'".format(x), values))
