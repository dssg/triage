import logging
import textwrap
from triage.database_reflection import table_exists


DEFAULT_PROTECTED_GROUPS_NAME = "protected_groups"


class ProtectedGroupsGeneratorNoOp(object):
    def generate_protected_groups_table(self, protected_groups_table_name, protected_groups_query):
        logging.warning(
            "No protected_groups configuration is available, so no labels will be created"
        )

    def generate_protected_groups_table(self, protected_groups_table_name, protected_groups_query):
        logging.warning(
            "No protected_groups configuration is available, so no labels will be created"
        )

    def clean_up(self, protected_groups_table_name):
        pass


class ProtectedGroupsGenerator(object):
    def __init__(self, db_engine, query, replace=True):
        self.db_engine = db_engine
        self.replace = replace
        # query is expected to select a number of entity ids, as_of_dates
        # and an outcome for each given an as-of-date
        self.query = query
        self.protected_groups_table_name = DEFAULT_PROTECTED_GROUPS_NAME

    def generate_protected_groups_table(self, protected_groups_table_name, protected_groups_query):
        self.db_engine.execute("drop table if exists {}".format(protected_groups_table_name))
        self.db_engine.execute(
            """
            create table {} as ({} )
        )""".format(
                protected_groups_table_name,
                protected_groups_query
            )
        )
        logging.info("Creating protected_groups table.")

    def clean_up(self, protected_groups_table_name):
        self.db_engine.execute("drop table if exists {}".format(protected_groups_table_name))
