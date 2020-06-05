import logging
import textwrap

import pandas
from sqlalchemy import text

from triage.database_reflection import table_exists
from triage.component.catwalk.storage import MatrixStore


class ProtectedGroupsGeneratorNoOp(object):
    def generate_all_dates(self, *args, **kwargs):
        logging.warning(
            "No bias audit configuration is available, so protected groups will not be created"
        )

    def generate(self, *args, **kwargs):
        logging.warning(
            "No bias audit configuration is available, so protected groups will not be created"
        )

    def as_dataframe(self, *args, **kwargs):
        return pandas.DataFrame()


class ProtectedGroupsGenerator(object):
    def __init__(self, db_engine, from_obj, attribute_columns, entity_id_column, knowledge_date_column, protected_groups_table_name, replace=True):
        self.db_engine = db_engine
        self.replace = replace
        self.protected_groups_table_name = protected_groups_table_name
        self.from_obj = from_obj
        self.attribute_columns = attribute_columns
        self.entity_id_column = entity_id_column
        self.knowledge_date_column = knowledge_date_column

    def generate_all_dates(self, as_of_dates, cohort_table_name, cohort_hash):
        table_is_new = False
        if not table_exists(self.protected_groups_table_name, self.db_engine):
            self.db_engine.execute(
                """
                create table if not exists {} (
                entity_id int,
                as_of_date date,
                {},
                cohort_hash text
            )""".format(
                    self.protected_groups_table_name,
                    ", ".join([str(col) + " varchar(60)" for col in self.attribute_columns])
                )
            )
            table_is_new = True
        else:
            logging.info("Not dropping and recreating protected groups table because "
                         "replace flag was set to False and table was found to exist")
        if self.replace:
            self.db_engine.execute(
                f'delete from {self.protected_groups_table_name} where cohort_hash = %s',
                cohort_hash
            )

        logging.info(
            "Creating protected_groups for %s as of dates",
            len(as_of_dates)
        )
        for as_of_date in as_of_dates:
            if not self.replace:
                logging.info(
                    "Looking for existing protected_groups for as of date %s",
                    as_of_date
                )
                any_existing_rows = list(self.db_engine.execute(
                    f"""select 1 from {self.protected_groups_table_name}
                    where as_of_date = '{as_of_date}'
                    and cohort_hash = '{cohort_hash}'
                    limit 1
                    """
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
                cohort_table_name=cohort_table_name,
                cohort_hash=cohort_hash
            )
        if table_is_new:
            self.db_engine.execute(f"create index on {self.protected_groups_table_name} (cohort_hash, as_of_date)")
        nrows = self.db_engine.execute(
            "select count(*) from {}".format(self.protected_groups_table_name)
        ).scalar()
        if nrows == 0:
            logging.warning("Done creating protected_groups, but no rows in protected_groups table!")
        else:
            logging.info("Done creating protected_groups table %s: rows: %s", self.protected_groups_table_name, nrows)


    def generate(self, start_date, cohort_table_name, cohort_hash):
        full_insert_query = text(textwrap.dedent(
            '''
            insert into {protected_groups_table}
            select distinct on (cohort.entity_id, cohort.as_of_date)
                cohort.entity_id,
                '{as_of_date}'::date as as_of_date,
                {attribute_columns},
                \'{cohort_hash}\' as cohort_hash
            from {cohort_table_name} cohort 
            left join (select * from {from_obj}) from_obj  on 
                cohort.entity_id = from_obj.{entity_id_column} and
                cohort.as_of_date > from_obj.{knowledge_date_column}
            where cohort.as_of_date = '{as_of_date}'::date
            order by cohort.entity_id, cohort.as_of_date, {knowledge_date_column} desc
        '''
        ).format(
            protected_groups_table=self.protected_groups_table_name,
            as_of_date=start_date,
            attribute_columns=", ".join([str(col) for col in self.attribute_columns]),
            cohort_hash=cohort_hash,
            cohort_table_name=cohort_table_name,
            from_obj=self.from_obj,
            knowledge_date_column=self.knowledge_date_column,
            entity_id_column=self.entity_id_column
        ))
        logging.debug("Running protected_groups creation query")
        logging.debug(full_insert_query)
        self.db_engine.execute(full_insert_query)

    def as_dataframe(self, as_of_dates, cohort_hash):
        """Queries the protected groups table to retrieve the protected attributes for each date
        Args:
            db_engine (sqlalchemy.engine) a database engine
            as_of_dates (list) the as_of_Dates to query

        Returns: (pandas.DataFrame) a dataframe with protected attributes for the given dates
        """
        as_of_dates_sql = "[{}]".format(
            ", ".join("'{}'".format(date.strftime("%Y-%m-%d %H:%M:%S.%f")) for date in as_of_dates)
        )
        query_string = f"""
            with dates as (
                select unnest(array{as_of_dates_sql}::timestamp[]) as as_of_date
            )
            select *
            from {self.protected_groups_table_name}
            join dates using(as_of_date)
            where cohort_hash = '{cohort_hash}'
        """
        protected_df = pandas.DataFrame.pg_copy_from(
            query_string,
            connectable=self.db_engine,
            parse_dates=["as_of_date"],
            index_col=MatrixStore.indices,
        )
        del protected_df['cohort_hash']
        return protected_df
