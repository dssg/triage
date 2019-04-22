"""Functions to retrieve basic information about tables in a Postgres database"""
from sqlalchemy import MetaData, Table


def split_table(table_name):
    """Split a fully-qualified table name into schema and table

    Args:
        table_name (string) A table name, either with or without a schema prefix

    Returns: (tuple) of schema and table name
    """
    table_parts = table_name.split(".")
    if len(table_parts) == 2:
        return tuple(table_parts)
    elif len(table_parts) == 1:
        return (None, table_parts[0])
    else:
        raise ValueError("Table name in unknown format")


def table_object(table_name, db_engine):
    """Produce a table object for the given table name

    This does not load data about the table from the engine yet,
    so it is safe to call for a table that doesn't exist.

    Args:
        table_name (string) A table name (with schema)
        db_engine (sqlalchemy.engine)

    Returns: (sqlalchemy.Table)
    """
    schema, table = split_table(table_name)
    meta = MetaData(schema=schema, bind=db_engine)
    return Table(table, meta)


def reflected_table(table_name, db_engine):
    """Produce a loaded table object for the given table name

    Will attempt to load the metadata about the table from the database
    So this will fail if the table doesn't exist.

    Args:
        table_name (string) A table name (with schema)
        db_engine (sqlalchemy.engine)

    Returns: (sqlalchemy.Table) A loaded table object
    """
    schema, table = split_table(table_name)
    meta = MetaData(schema=schema, bind=db_engine)
    return Table(table, meta, autoload=True, autoload_from=db_engine)


def table_exists(table_name, db_engine):
    """Checks whether the table exists

    Args:
        table_name (string) A table name (with schema)
        db_engine (sqlalchemy.engine)

    Returns: (boolean) Whether or not the table exists in the database
    """
    return table_object(table_name, db_engine).exists()


def table_has_data(table_name, db_engine):
    """Check whether the table contains any data

    Args:
        table_name (string) A table name (with schema)
        db_engine (sqlalchemy.engine)

    Returns: (boolean) Whether or not the table has any data
    """
    if not table_exists(table_name, db_engine):
        return False
    results = [
        row for row in db_engine.execute("select * from {} limit 1".format(table_name))
    ]

    return len(results) > 0


def table_has_column(table_name, column, db_engine):
    """Check whether the table contains a column of the given name

    The table is expected to exist.

    Args:
        table_name (string) A table name (with schema)
        column (string) A column name
        db_engine (sqlalchemy.engine)

    Returns: (boolean) Whether or not the table contains the column
    """
    return column in reflected_table(table_name, db_engine).columns


def column_type(table_name, column, db_engine):
    """Find the database type of the given column in the given table

    The table is expected to exist, and contain a column of the given name

    Args:
        table_name (string) A table name (with schema)
        column (string) A column name
        db_engine (sqlalchemy.engine)

    Returns: (sqlalchemy.types) The DDL type of the column; For instance,
        sqlalchemy.types.BOOLEAN instead of
        sqlalchemy.types.Boolean
    """
    return type(reflected_table(table_name, db_engine).columns[column].type)
