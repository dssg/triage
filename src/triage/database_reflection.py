"""Functions to retrieve basic information about tables in a Postgres database"""
from sqlalchemy import MetaData, Table, text, quoted_name


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


def table_object(table_name):
    """Produce a table object for the given table name

    This does not load data about the table from the engine yet,
    so it is safe to call for a table that doesn't exist.

    Args:
        table_name (string) A table name (with schema)
        db_engine (sqlalchemy.engine)

    Returns: (sqlalchemy.Table)
    """
    schema, table = split_table(table_name)
    meta = MetaData(schema=schema)
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
    meta = MetaData(schema=schema)
    return Table(table, meta, autoload_with=db_engine)


def table_exists(table_name, db_engine):
    """Checks whether the table exists

    Args:
        table_name (string) A table name (with schema)
        db_engine (sqlalchemy.engine)

    Returns: (boolean) Whether or not the table exists in the database
    """
    schema, table = split_table(table_name)
    inspector = db_engine.get_inspector() # get the inspector from SerializableDbEngine
    return inspector.has_table(table, schema=schema)


def table_has_data(table_name, db_engine):
    """Check whether the table contains any data

    Args:
        table_name (string) A table name (with schema)
        db_engine (sqlalchemy.engine)

    Returns: (boolean) Whether or not the table has any data
    """
    if not table_exists(table_name, db_engine):
        return False
    
    sql = text(f"select * from {quoted_name(table_name, quote=True)} limit 1")
    with db_engine.connect() as conn:
        return conn.execute(sql).first() is not None
   

def table_row_count(table_name, db_engine):
    """Return the length of the table.

    The table is expected to exist.

    Args:
        table_name (string) A table name (with schema)
        db_engine (sqlalchemy.engine)

    Returns: (int) The number of rows in the table
    """
    sql = text(f"select count(*) from {quoted_name(table_name, quote=True)}")
    with db_engine.connect() as conn:
        return conn.execute(sql).scalar_one()
   

def table_has_duplicates(table_name, column_list, db_engine):
    """Check whether the table has duplicate rows on the set of columns.

    The table is expected to exist and contain the columns in column_list.

    Args:
        table_name (string) A table name (with schema)
        column_list (list) A list of column names
        db_engine (sqlalchemy.engine)

    Returns: (boolean) Whether or not duplicates are found
    """
    if not table_has_data(table_name, db_engine):
        return False

    cols = ", ".join(str(quoted_name(c, quote=True)) for c in column_list)
    sql = text(f"""
        WITH counts AS (
            SELECT 
               {cols}, 
               COUNT(*) AS num_records
            FROM {quoted_name(table_name, quote=True)}
            GROUP BY {cols}
        )
        SELECT MAX(num_records) FROM counts
    """)

    with db_engine.connect() as conn: 
        return conn.execute(sql).scalar_one() > 1
    

def table_has_column(table_name, column, db_engine):
    """Check whether the table contains a column of the given name

    The table is expected to exist.

    Args:
        table_name (string) A table name (with schema)
        column (string) A column name
        db_engine (sqlalchemy.engine)

    Returns: (boolean) Whether or not the table contains the column
    """
    schema, table = split_table(table_name)
    inspector = db_engine.get_inspector() # get inspector from SerializableDbEngine
    columns = inspector.get_columns(table, schema=schema)
    # inspect returns column metadata dictionaries 
    return any(col['name'] == column for col in columns)


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
    schema, table = split_table(table_name)
    inspector = db_engine.get_inspector() # get inspector from SerializableDbEngine
    columns = inspector.get_columns(table, schema=schema)
    for col in columns: 
        if col['name'] == column:
            return type(col['type'])
    raise KeyError(f"Column {column} not found")
    

def schema_tables(schema_name, db_engine):
    inspector = db_engine.get_inspector()
    return inspector.get_table_names(schema=schema_name)
