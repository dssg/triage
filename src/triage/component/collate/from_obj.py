import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from triage.validation_primitives import (
    table_should_exist,
    table_should_have_column,
    column_should_be_timelike
)
import sqlparse


class FromObj:
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
    def drop_materialized_table_sql(self):
        return f"drop table if exists {self.materialized_table}"

    def should_materialize(self):
        try:
            (statement,) = sqlparse.parse(self.from_obj)
        except ValueError as exc:
            raise ValueError("Expected exactly one statment to be parsed by sqlparse "
                             f"from from_obj {self.from_obj}.") from exc
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
        if not isinstance(from_obj, sqlparse.sql.Identifier):
            logger.warning(
                f"Expected {from_obj} to parse as an Identifier. It did not. "
                f"As a result, falling back to *not* materializing raw from object {self.from_obj}"
            )
            return False
        return from_obj.has_alias() and from_obj.get_alias() == from_obj.get_real_name()

    def maybe_materialize(self, db_engine):
        if self.should_materialize():
            logger.spam(f"from_obj in {self.name} looks like a subquery, so creating table")
            db_engine.execute(self.drop_materialized_table_sql)
            db_engine.execute(self.create_materialized_table_sql)
            logger.spam(f"Created table to hold from_obj. New table: {self.materialized_table}")
            self.validate(db_engine)
            db_engine.execute(self.index_materialized_table_sql)
            logger.spam(f"Indexed from_obj table: {self.materialized_table}")
            logger.debug(f"Materialized table {self.materialized_table}")
        else:
            logger.debug(f"from_obj in {self.name} did not look like a subquery, so did not materialize")

    def validate(self, db_engine):
        logger.spam(f"Validating from_obj {self.materialized_table}")
        table_should_exist(self.materialized_table, db_engine)
        logger.spam(f"Table {self.materialized_table} successfully found")
        table_should_have_column(self.materialized_table, 'entity_id', db_engine)
        logger.spam(f"Successfully found entity_id column in {self.materialized_table}")
        table_should_have_column(self.materialized_table, self.knowledge_date_column, db_engine)
        column_should_be_timelike(self.materialized_table, self.knowledge_date_column, db_engine)
        logger.spam(
            f"Successfully found configured knowledge date column in {self.materialized_table}"
        )
