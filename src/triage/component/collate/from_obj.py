import logging
from triage.validation_primitives import (
    table_should_exist,
    table_should_have_column,
    column_should_be_timelike
)


class MaterializedFromObj(object):
    def __init__(self, from_obj, name, knowledge_date_column):
        self.from_obj = from_obj
        self.name = name
        self.knowledge_date_column = knowledge_date_column

    @property
    def table(self):
        return f"{self.name}_from_obj"

    @property
    def create(self):
        return f"create table {self.table} as (select * from {self.from_obj})"

    @property
    def index(self):
        return f"create index on {self.table} ({self.knowledge_date_column})"

    @property
    def drop(self):
        return f"drop table if exists {self.table}"

    def validate(self, db_engine):
        logging.info('Validating materialized from_obj %s', self.table)
        table_should_exist(self.table, db_engine)
        logging.info('Table %s successfully found', self.table)
        table_should_have_column(self.table, 'entity_id', db_engine)
        logging.info('Successfully found entity_id column in %s', self.table)
        table_should_have_column(self.table, self.knowledge_date_column, db_engine)
        column_should_be_timelike(self.table, self.knowledge_date_column, db_engine)
        logging.info('Successfully found configured knowledge date column in %s', self.table)

    def execute(self, db_engine):
        logging.info("Materializing from_obj for %s", self.name)
        db_engine.execute(self.drop)
        db_engine.execute(self.create)
        logging.info("Created materialized from_obj at table: %s", self.table)
        self.validate(db_engine)
        db_engine.execute(self.index)
        logging.info("Indexed materialized from_obj at table: %s", self.table)
