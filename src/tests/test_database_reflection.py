import pytest
import sqlalchemy 

from sqlalchemy import Table, text, quoted_name
from tests.conftest import shared_db_engine
from triage.database_reflection import (
    split_table,
    table_object,
    reflected_table,
    table_exists,
    table_has_data,    
    table_has_duplicates,
    table_has_column,
    column_type,
    table_row_count,
    schema_tables,
)

def test_split_table():
    assert split_table("staging.incidents") == ("staging", "incidents")
    assert split_table("incidents") == (None, "incidents")
    with pytest.raises(ValueError):
        split_table("blah.staging.incidents")


def test_table_object():
    assert isinstance(table_object("incidents"), Table)


def test_reflected_table(db_engine):
    with db_engine.connect() as conn:
        conn.execute(text("create table incidents (col1 varchar)"))
        conn.commit()
        # if table was successfully reflected, it should have metadata.
    table_ = reflected_table("incidents", db_engine)
    print(table_)
    assert table_.metadata is not None


def test_table_exists(db_engine):
    with db_engine.begin() as conn:
        conn.execute(text("create table incidents (col1 varchar)"))
        
    assert table_exists("incidents", db_engine)
    assert not table_exists("compliments", db_engine)


def test_table_has_data(db_engine):
    with db_engine.begin() as conn:
        conn.execute(text("create table incidents (col1 varchar)"))
        conn.execute(text("create table compliments (col1 varchar)"))
        conn.execute(text("insert into compliments values ('good job')"))
    
    assert table_has_data("compliments", db_engine)
    assert not table_has_data("incidents", db_engine)


def test_table_has_duplicates(db_engine):
    with db_engine.connect() as conn:
        conn.execute(text("create table events (col1 int, col2 int)"))
        conn.commit()
        assert not table_has_duplicates("events", ['col1'], db_engine)
    
        conn.execute(text("insert into events values (1,2)"))   
        conn.commit()     
        assert not table_has_duplicates("events", ['col1', 'col2'], db_engine)
        conn.execute(text("insert into events values (1,3)"))
        conn.commit()
        assert not table_has_duplicates("events", ['col1', 'col2'], db_engine)

        assert table_has_duplicates('events', ['col1'], db_engine)
        assert not table_has_duplicates("events", ['col1', 'col2'], db_engine)

        conn.execute(text("insert into events values (1,2)"))
        conn.commit()
        assert table_has_duplicates("events", ['col1', 'col2'], db_engine)


def test_table_row_count(db_engine):
    with db_engine.begin() as conn:
        conn.execute(text("create table incidents (col1 varchar)"))
        conn.execute(text("insert into incidents values ('a'), ('b'), ('c')"))
    
    assert table_row_count("incidents", db_engine) == 3


def test_table_has_column(db_engine):
    with db_engine.begin() as conn:
        conn.execute(text("create table incidents (col1 varchar)"))
    
    assert table_has_column("incidents", "col1", db_engine)
    assert not table_has_column("incidents", "col2", db_engine)


def test_column_type(db_engine):
    with db_engine.begin() as conn:
        conn.execute(text("create table incidents (col1 varchar, col2 int)"))
    
    assert 'VARCHAR' in str(column_type("incidents", "col1", db_engine))
    assert 'INTEGER' in str(column_type("incidents", "col2", db_engine))


def test_schema_tables(db_engine):
    with db_engine.begin() as conn:
        conn.execute(text(f"create schema if not exists test"))
        conn.execute(text(f"create table test.incidents (col1 varchar)"))
        conn.execute(text(f"create table test.compliments (col1 varchar)"))
    
    tables = schema_tables("test", db_engine)
    assert f"incidents" in tables
    assert f"compliments" in tables