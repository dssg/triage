from sqlalchemy import Table
from sqlalchemy import create_engine
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
        assert isinstance(dbreflect.table_object("incidents", self.engine), Table)

    def test_reflected_table(self):
        self.engine.execute("create table incidents (col1 varchar)")
        assert dbreflect.reflected_table("incidents", self.engine).exists()

    def test_table_exists(self):
        self.engine.execute("create table incidents (col1 varchar)")
        assert dbreflect.table_exists("incidents", self.engine)
        assert not dbreflect.table_exists("compliments", self.engine)

    def test_table_has_data(self):
        self.engine.execute("create table incidents (col1 varchar)")
        self.engine.execute("create table compliments (col1 varchar)")
        self.engine.execute("insert into compliments values ('good job')")
        assert dbreflect.table_has_data("compliments", self.engine)
        assert not dbreflect.table_has_data("incidents", self.engine)

    def test_table_has_duplicates(self):
        self.engine.execute("create table events (col1 int, col2 int)")
        assert not dbreflect.table_has_duplicates("events", ['col1', 'col2'], self.engine)
        self.engine.execute("insert into events values (1,2)")
        self.engine.execute("insert into events values (1,3)")
        assert dbreflect.table_has_duplicates("events", ['col1'], self.engine)
        assert not dbreflect.table_has_duplicates("events", ['col1', 'col2'], self.engine)
        self.engine.execute("insert into events values (1,2)")
        assert dbreflect.table_has_duplicates("events", ['col1', 'col2'], self.engine)

    def test_table_has_column(self):
        self.engine.execute("create table incidents (col1 varchar)")
        assert dbreflect.table_has_column("incidents", "col1", self.engine)
        assert not dbreflect.table_has_column("incidents", "col2", self.engine)

    def test_column_type(self):
        self.engine.execute("create table incidents (col1 varchar)")
        assert dbreflect.column_type("incidents", "col1", self.engine) == VARCHAR
