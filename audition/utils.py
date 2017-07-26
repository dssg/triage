def str_in_sql(values):
    return ','.join(map(lambda x: "'{}'".format(x), values))
