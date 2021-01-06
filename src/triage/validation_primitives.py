"""Functions for validating input, mostly around database schema and state"""
from triage.database_reflection import (
    table_exists,
    table_has_column,
    column_type,
    table_has_data,
)
from sqlalchemy.types import (
    BIGINT,
    BOOLEAN,
    DATE,
    DATETIME,
    INTEGER,
    SMALLINT,
    TEXT,
    TIMESTAMP,
    VARCHAR,
)
from sqlalchemy.dialects.postgresql.base import TIMESTAMP as POSTGRES_TIMESTAMP


def table_should_exist(table_name, db_engine):
    """Ensures that the table exists in the given database

    Args:
        table_name (string) A table name (with schema)
        db_engine (sqlalchemy.engine)

    Raises: ValueError if the table does not exist
    """
    if not table_exists(table_name, db_engine):
        raise ValueError("{} table does not exist".format(table_name))


def table_should_have_column(table_name, column, db_engine):
    """Ensures that the table has the given column

    Args:
        table_name (string) A table name (with schema)
        column (string) The name of a column
        db_engine (sqlalchemy.engine)

    Raises: ValueError if the table does not contain the column
    """
    table_should_exist(table_name, db_engine)
    if not table_has_column(table_name, column, db_engine):
        raise ValueError("{} table does not have {} column".format(table_name, column))


def table_should_have_data(table_name, db_engine):
    """Ensures that the table has at least one row

    Args:
        table_name (string) A table name (with schema)
        db_engine (sqlalchemy.engine)

    Raises: ValueError if the table does not have at least one row
    """
    table_should_exist(table_name, db_engine)
    if not table_has_data(table_name, db_engine):
        raise ValueError("{} table does not have any data".format(table_name))


def column_should_be_in_types(table_name, column, valid_types, db_engine):
    """Ensures that the given column is one of the given types

    Args:
        table_name (string) A table name (with schema)
        column (string) The name of a column
        valid_types (list) A list of SQLAlchemy DDL types, like sqlalchemy.types.BOOLEAN
        db_engine (sqlalchemy.engine)

    Raises: ValueError if the column is not one of the given types
    """
    table_should_have_column(table_name, column, db_engine)
    reflected_type = column_type(table_name, column, db_engine)
    if reflected_type not in valid_types:
        raise ValueError(
            "{}.{} should be in types {} but was {}".format(
                table_name, column, valid_types, reflected_type
            )
        )


def column_should_be_booleanlike(table_name, column, db_engine):
    """Ensures that the given column can be casted to a boolean

    Allows BOOLEAN, SMALLINT, and INTEGER, as these are commonly used.
    It does not check that the data in a SMALLINT column all conforms to 0/1

    Args:
        table_name (string) A table name (with schema)
        column (string) The name of a column
        db_engine (sqlalchemy.engine)

    Raises: ValueError if the column is not a recognized boolean-compatible type
    """
    table_should_have_column(table_name, column, db_engine)
    column_should_be_in_types(
        table_name, column, [BOOLEAN, SMALLINT, INTEGER], db_engine
    )


def column_should_be_timelike(table_name, column, db_engine):
    """Ensures that the given column can be used for temporal data

    Many date/time operations are fairly compatible with each other,
    so this routine is fairly permissive. If you want to be more strict,
    call column_should_be_in_types directly

    Args:
        table_name (string) A table name (with schema)
        column (string) The name of a column
        db_engine (sqlalchemy.engine)

    Raises: ValueError if the column is not a recognized temporal type
    """
    table_should_have_column(table_name, column, db_engine)
    column_should_be_in_types(
        table_name, column, [DATE, DATETIME, TIMESTAMP, POSTGRES_TIMESTAMP], db_engine
    )


def column_should_be_intlike(table_name, column, db_engine):
    """Ensures that the given column can act as an integer

    Args:
        table_name (string) A table name (with schema)
        column (string) The name of a column
        db_engine (sqlalchemy.engine)

    Raises: ValueError if the column is not a recognized integer type
    """
    table_should_have_column(table_name, column, db_engine)
    column_should_be_in_types(
        table_name, column, [BIGINT, SMALLINT, INTEGER], db_engine
    )


def column_should_be_stringlike(table_name, column, db_engine):
    """Ensures that the given column can act as an string

    Args:
        table_name (string) A table name (with schema)
        column (string) The name of a column
        db_engine (sqlalchemy.engine)

    Raises: ValueError if the column is not a recognized string type
    """
    table_should_have_column(table_name, column, db_engine)
    column_should_be_in_types(table_name, column, [VARCHAR, TEXT], db_engine)


def string_is_tablesafe(string):
    if not string:
        return False
    return all((c.isalpha() and c.islower()) or c.isdigit() or c == '_' for c in string)
