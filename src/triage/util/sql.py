def str_in_sql(values):
    """Create SQL suitable for the content of an IN clause from a list"""
    return ",".join(map(lambda x: "'{}'".format(x), values))
