import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import textwrap

import pandas as pd
from sqlalchemy import text

from triage.database_reflection import table_exists
from triage.component.catwalk.storage import MatrixStore


class ProtectedGroupsGeneratorNoOp:
    def generate_all_dates(self, *args, **kwargs):
        logger.notice(
            "No bias audit configuration is available, so protected groups will not be created"
        )

    def generate(self, *args, **kwargs):
        logger.notice(
            "No bias audit configuration is available, so protected groups will not be created"
        )

    def as_dataframe(self, *args, **kwargs):
        logger.notice(
            "No bias audit configuration is available, so protected groups were not created: returning an empty data frame"
        )
        return pd.DataFrame()


class ProtectedGroupsGenerator:
    def __init__(self, db_engine, from_obj, attribute_columns, entity_id_column, knowledge_date_column, protected_groups_table_name, replace=True):
        self.db_engine = db_engine
        self.replace = replace
        self.protected_groups_table_name = protected_groups_table_name
        self.from_obj = from_obj
        self.attribute_columns = attribute_columns
        self.entity_id_column = entity_id_column
        self.knowledge_date_column = knowledge_date_column

    def generate_all_dates(self, as_of_dates, cohort_table_name, cohort_hash):
        logger.spam("Creating protected groups table")
        table_is_new = False
        if not table_exists(self.protected_groups_table_name, self.db_engine):
            with self.db_engine.begin() as conn:
                conn.execute(
                    text(
                        f"""
                        create table if not exists {self.protected_groups_table_name} (
                        entity_id int,
                        as_of_date date,
                        {', '.join([str(col) + " varchar" for col in self.attribute_columns])},
                        cohort_hash text
                        )"""
                    )
                )
            logger.debug(f"Protected groups table {self.protected_groups_table_name} created")
            table_is_new = True
        else:
            logger.debug(f"Protected groups table {self.protected_groups_table_name} exist")

        if self.replace:
            with self.db_engine.begin() as conn:
                conn.execute(
                    text(
                        f"delete from {self.protected_groups_table_name} where cohort_hash = '{cohort_hash}'"
                    )
                )
            logger.debug(f"Because replace flag is true, removed from {self.protected_groups_table_name} all rows from cohort {cohort_hash}")

        logger.spam(
            f"Creating protected_groups for {len(as_of_dates)} as of dates",
        )

        for as_of_date in as_of_dates:
            if not self.replace:
                logger.spam(f"Looking for existing protected_groups for as of date {as_of_date}")
                with self.db_engine.connect() as conn:
                    result = conn.execute(
                        text(
                            f"""select 1 from {self.protected_groups_table_name}
                            where as_of_date = :as_of_date
                            and cohort_hash = :cohort_hash
                            limit 1
                            """ 
                        ),
                        {
                            "as_of_date": as_of_date,
                            "cohort_hash": cohort_hash,
                        }
                    )

                    any_existing_rows = result.first() is not None 
                    if any_existing_rows:
                        logger.debug("Since nonzero existing protected_groups found, skipping")
                        continue

            logger.debug(
                f"Generating protected groups for as of date {as_of_date} "
            )
            self.generate(
                start_date=as_of_date,
                cohort_table_name=cohort_table_name,
                cohort_hash=cohort_hash
            )

        with self.db_engine.begin() as conn:
            if table_is_new:
                conn.execute(text(f"create index on {self.protected_groups_table_name} (cohort_hash, as_of_date)"))
            nrows = conn.execute(
                text(
                    f"select count(*) from {self.protected_groups_table_name}"
                )
            ).scalar()
            logger.spam(f"nrows: {nrows}")
            if nrows == 0:
                logger.warning("Done creating protected_groups, but no rows in protected_groups table!")
            else:
                logger.success(f"Protected groups stored in the table "
                            f"{self.protected_groups_table_name} successfully")
                logger.spam(f"Protected groups table has {nrows} rows")

    def generate(self, start_date, cohort_table_name, cohort_hash):
        attribute_columns = ", ".join([str(col) for col in self.attribute_columns])
        full_insert_query = f"""
                insert into {self.protected_groups_table_name}
                select distinct on (cohort.entity_id, cohort.as_of_date)
                    cohort.entity_id,
                    CAST(:start_date as date) as as_of_date,
                    {attribute_columns},
                    :cohort_hash as cohort_hash
                from {cohort_table_name} cohort
                left join (select * from {self.from_obj}) from_obj  on
                    cohort.entity_id = from_obj.{self.entity_id_column} and
                    cohort.as_of_date > from_obj.{self.knowledge_date_column}
                where cohort.as_of_date = CAST(:start_date as date)
                order by cohort.entity_id, cohort.as_of_date, {self.knowledge_date_column} desc
            """
            
        logger.debug("Running protected_groups creation query")
        logger.spam(f"Query to insert protected_df with start_date {start_date} and cohort_hash {cohort_hash}: {full_insert_query}")
        with self.db_engine.begin() as conn:
            conn.execute(
                text(full_insert_query),
                {
                    "start_date": start_date,
                    "cohort_hash": cohort_hash,
                }
            )
            

    def as_dataframe(self, as_of_dates, cohort_hash):
        """Queries the protected groups table to retrieve the protected attributes for each date
        Args:
            db_engine (sqlalchemy.engine) a database engine
            as_of_dates (list) the as_of_Dates to query

        Returns: (pd.DataFrame) a dataframe with protected attributes for the given dates
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
        logger.spam(f"Getting protected_df from query string: {query_string}")
        
        protected_df = pd.read_sql(
            query_string,
            con=self.db_engine,
            parse_dates=["as_of_date"],
            index_col=MatrixStore.indices,
        )
        protected_df[self.attribute_columns] = protected_df[self.attribute_columns].astype(str)
        del protected_df['cohort_hash']
        return protected_df
