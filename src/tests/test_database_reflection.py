import pytest
from sqlalchemy import Table, text
from sqlalchemy.types import VARCHAR

import triage.database_reflection as dbreflect


def test_split_table():
    assert dbreflect.split_table("staging.incidents") == ("staging", "incidents")
    assert dbreflect.split_table("incidents") == (None, "incidents")
    with pytest.raises(ValueError):
        dbreflect.split_table("blah.staging.incidents")


def test_table_object(db_engine):
    assert isinstance(dbreflect.table_object("incidents", db_engine), Table)


def test_reflected_table(db_engine):
    with db_engine.connect() as conn:
        conn.execute(text("create table incidents (col1 varchar)"))
        conn.commit()
    # Table.exists() was removed in SQLAlchemy 2.0, just verify we get a Table back
    table = dbreflect.reflected_table("incidents", db_engine)
    assert table is not None
    assert "col1" in table.columns


def test_table_exists(db_engine):
    with db_engine.connect() as conn:
        conn.execute(text("create table incidents (col1 varchar)"))
        conn.commit()
    assert dbreflect.table_exists("incidents", db_engine)
    assert not dbreflect.table_exists("compliments", db_engine)


def test_table_has_data(db_engine):
    with db_engine.connect() as conn:
        conn.execute(text("create table incidents (col1 varchar)"))
        conn.execute(text("create table compliments (col1 varchar)"))
        conn.execute(text("insert into compliments values ('good job')"))
        conn.commit()
    assert dbreflect.table_has_data("compliments", db_engine)
    assert not dbreflect.table_has_data("incidents", db_engine)


def test_table_has_duplicates(db_engine):
    with db_engine.connect() as conn:
        conn.execute(text("create table events (col1 int, col2 int)"))
        conn.commit()
    assert not dbreflect.table_has_duplicates("events", ['col1', 'col2'], db_engine)
    with db_engine.connect() as conn:
        conn.execute(text("insert into events values (1,2)"))
        conn.execute(text("insert into events values (1,3)"))
        conn.commit()
    assert dbreflect.table_has_duplicates("events", ['col1'], db_engine)
    assert not dbreflect.table_has_duplicates("events", ['col1', 'col2'], db_engine)
    with db_engine.connect() as conn:
        conn.execute(text("insert into events values (1,2)"))
        conn.commit()
    assert dbreflect.table_has_duplicates("events", ['col1', 'col2'], db_engine)


def test_table_has_column(db_engine):
    with db_engine.connect() as conn:
        conn.execute(text("create table incidents (col1 varchar)"))
        conn.commit()
    assert dbreflect.table_has_column("incidents", "col1", db_engine)
    assert not dbreflect.table_has_column("incidents", "col2", db_engine)


def test_column_type(db_engine):
    with db_engine.connect() as conn:
        conn.execute(text("create table incidents (col1 varchar)"))
        conn.commit()
    assert dbreflect.column_type("incidents", "col1", db_engine) == VARCHAR
