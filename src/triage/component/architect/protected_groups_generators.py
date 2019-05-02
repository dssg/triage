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
    def __init__(self, db_engine, query, protected_groups_table_name=None, replace=True):
        self.db_engine = db_engine
        self.replace = replace
        # query is expected to select a number of entity ids, as_of_dates
        # and an outcome for each given an as-of-date
        self.query = query
        self.protected_groups_table_name = protected_groups_table_name if protected_groups_table_name else DEFAULT_PROTECTED_GROUPS_NAME

    def _create_protected_groups_table(self, protected_groups_table_name, attribute_columns):
        if self.replace or not table_exists(protected_groups_table_name, self.db_engine):
            self.db_engine.execute("drop table if exists {}".format(protected_groups_table_name))
            self.db_engine.execute(
                """
                create table {} (
                entity_id int,
                as_of_date date,
                {}

            )""".format(
                    protected_groups_table_name,
                    ", ".join([str(col) + " varchar(30)" for col in attribute_columns])
                )
            )
        else:
            logging.info("Not dropping and recreating table because "
                         "replace flag was set to False and table was found to exist")

    def generate_all_dates(self, protected_groups_table_name, as_of_dates, attribute_columns, cohort_table_name,
                           source_table_name, knowledge_date_column):
        self._create_protected_groups_table(protected_groups_table_name)
        logging.info(
            "Creating labels for %s as of dates and %s label timespans",
            len(as_of_dates)
        )
        for as_of_date in as_of_dates:
            if not self.replace:
                logging.info(
                    "Looking for existing protected_groups for as of date %s and label timespan %s",
                    as_of_date
                )
                any_existing_rows = list(self.db_engine.execute(
                    """select 1 from {protected_groups_table}
                    where as_of_date = '{as_of_date}'
                    limit 1
                    """.format(
                        protected_groups_table=protected_groups_table_name,
                        as_of_date=as_of_date
                    )
                ))
                if len(any_existing_rows) == 1:
                    logging.info("Since nonzero existing protected_groups found, skipping")
                    continue

            logging.info(
                "Generating protected groups for as of date %s ",
                as_of_date
            )
            self.generate(
                start_date=as_of_date,
            )
        nrows = [
            row[0]
            for row in self.db_engine.execute(
                "select count(*) from {}".format(protected_groups_table_name)
            )
        ][0]
        if nrows == 0:
            logging.warning("Done creating protected_groups, but no rows in protected_groups table!")
        else:
            logging.info("Done creating protected_groups table %s: rows: %s", protected_groups_table_name, nrows)

    def generate(self, start_date, protected_groups_table_name, attribute_columns, cohort_table_name, source_table_name,
                 knowledge_date_column):

        full_insert_query = textwrap.dedent(
            """
            insert into {protected_groups_table}
            select distinct on (cohort.entity_id, cohort.as_of_date)
                entity_id,
                '{as_of_date}'::date as as_of_date,
                {attribute_columns} 
            from {cohort_table} cohort 
            left join {source_table_name} source_table  on 
                cohort.entity_id = source_table.{entity_id_column} and
                cohort.as_of_date > source_table.{knowledge_date_column}
            where cohort.as_of_date = '{as_of_date}'::date
            order by entity_id, as_of_date, {knowledge_date_column} desc
        """
        ).format(
            protected_groups_table=protected_groups_table_name,
            as_of_date=start_date,
            attribute_columns=", ".join([str(col) for col in attribute_columns]),
            cohort_table=cohort_table_name,
            source_table_name=source_table_name,
            knowledge_date_column=knowledge_date_column
        )
        logging.debug("Running protected_groups creation query")
        logging.debug(full_insert_query)
        self.db_engine.execute(full_insert_query)

    def clean_up(self, protected_groups_table_name):
        self.db_engine.execute("drop table if exists {}".format(protected_groups_table_name))


