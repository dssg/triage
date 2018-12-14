import logging
from triage.validation_primitives import (
    table_should_exist,
    table_should_have_column,
    column_should_be_timelike
)
import sqlparse


class FromObj(object):
    def __init__(self, from_obj, name, knowledge_date_column):
        self.from_obj = from_obj
        self.name = name
        self.knowledge_date_column = knowledge_date_column

    @property
    def table(self):
        if self.should_materialize():
            return self.materialized_table
        else:
            return self.from_obj

    @property
    def materialized_table(self):
        return f"{self.name}_from_obj"

    @property
    def create_materialized_table_sql(self):
        return f"create table {self.materialized_table} as (select * from {self.from_obj})"

    @property
    def index_materialized_table_sql(self):
        return f"create index on {self.materialized_table} ({self.knowledge_date_column})"

    @property
    def drop(self):
        return f"drop table if exists {self.materialized_table}"

    def should_materialize(self):
        statements = sqlparse.parse(self.from_obj)
        # sqlparse.parse returns a tuple of Statement objects. a from_obj should have just one
        if len(statements) > 1:
            raise ValueError(f"Expecting only one statement in from_obj {self.from_obj}")
        statement = statements[0]
        from_obj = statement.token_first(skip_ws=True, skip_cm=True)
        # token_first returns the first 'token' at the top level. This includes any aliases
        # In other words, it's something that you can "select *" from

        # We only want to materialize subqueries. Subqueries need aliases, many other
        # from_objects don't.
        # The first check, 'has_alias', covers this.

        # The real exception is if just a table is specified but has an alias,
        # for easy reference elsewhere.
        # The second check covers this. The 'real name' in these cases is the name
        # of the original table, whereas for a subquery there is no 'real name' besides the alias
        return from_obj.has_alias() and from_obj.get_alias() == from_obj.get_real_name()

    def maybe_materialize(self, db_engine):
        if self.should_materialize():
            logging.info("from_obj %s looks like a subquery, so creating table", self.name)
            db_engine.execute(self.drop)
            db_engine.execute(self.create)
            logging.info("Created table to hold from_obj. New table: %s", self.materialized_table)
            self.validate(db_engine)
            db_engine.execute(self.index)
            logging.info("Indexed from_obj table: %s", self.materialized_table)
        else:
            logging.info("from_obj did not look like a subquery, so did not materialize")

    def validate(self, db_engine):
        logging.info('Validating from_obj %s', self.materialized_table)
        table_should_exist(self.materialized_table, db_engine)
        logging.info('Table %s successfully found', self.materialized_table)
        table_should_have_column(self.materialized_table, 'entity_id', db_engine)
        logging.info('Successfully found entity_id column in %s', self.materialized_table)
        table_should_have_column(self.materialized_table, self.knowledge_date_column, db_engine)
        column_should_be_timelike(self.materialized_table, self.knowledge_date_column, db_engine)
        logging.info(
            'Successfully found configured knowledge date column in %s',
            self.materialized_table
        )
