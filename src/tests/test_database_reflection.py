from sqlalchemy import Table, text
from triage import create_engine
from sqlalchemy.types import VARCHAR
from testing.postgresql import Postgresql
from unittest import TestCase

import triage.database_reflection as dbreflect


class TestDatabaseReflection(TestCase):
    def setUp(self):
        self.postgresql = Postgresql()
        self.engine = create_engine(self.postgresql.url())

    def tearDown(self):
        self.postgresql.stop()

    def test_split_table(self):
        assert dbreflect.split_table("staging.incidents") == ("staging", "incidents")
        assert dbreflect.split_table("incidents") == (None, "incidents")
        with self.assertRaises(ValueError):
            dbreflect.split_table("blah.staging.incidents")

    def test_table_object(self):
        assert isinstance(dbreflect.table_object("incidents"), Table)

    def test_reflected_table(self):
        with self.engine.begin() as conn:
            conn.execute(text("create table incidents (col1 varchar)"))
            # if table was successfully reflected, it should have metadata.
            table_ = dbreflect.reflected_table("incidents", self.engine)
            assert table_.metadata is not None

    def test_table_exists(self):
        with self.engine.begin() as conn:
            conn.execute(text("create table incidents (col1 varchar)"))
        
        assert dbreflect.table_exists("incidents", self.engine)
        assert not dbreflect.table_exists("compliments", self.engine)

    def test_table_has_data(self):
        with self.engine.begin() as conn:
            conn.execute(text("create table incidents (col1 varchar)"))
            conn.execute(text("create table compliments (col1 varchar)"))
            conn.execute(text("insert into compliments values ('good job')"))
        
        assert dbreflect.table_has_data("compliments", self.engine)
        assert not dbreflect.table_has_data("incidents", self.engine)

    def test_table_has_duplicates(self):
        with self.engine.begin() as conn:
            conn.execute(text("create table events (col1 int, col2 int)"))
        
        assert not dbreflect.table_has_duplicates("events", ['col1', 'col2'], self.engine)
        
        with self.engine.begin() as conn:
            conn.execute(text("insert into events values (1,2)"))
            conn.execute(text("insert into events values (1,3)"))

        assert dbreflect.table_has_duplicates("events", ['col1'], self.engine)
        assert not dbreflect.table_has_duplicates("events", ['col1', 'col2'], self.engine)

        with self.engine.begin() as conn:    
            conn.execute(text("insert into events values (1,2)"))
        
        assert dbreflect.table_has_duplicates("events", ['col1', 'col2'], self.engine)

    def test_table_has_column(self):
        with self.engine.begin() as conn:
            conn.execute(text("create table incidents (col1 varchar)"))
        
        assert dbreflect.table_has_column("incidents", "col1", self.engine)
        assert not dbreflect.table_has_column("incidents", "col2", self.engine)

    def test_column_type(self):
        with self.engine.begin() as conn:
            conn.execute(text("create table incidents (col1 varchar)"))
        
        assert dbreflect.column_type("incidents", "col1", self.engine) == VARCHAR
