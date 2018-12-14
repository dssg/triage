from abc import ABC, abstractmethod

import logging
import sqlparse
from triage.database_reflection import table_exists, table_columns
from triage.util.db import run_statements


class FeatureBlock(ABC):
    def __init__(
        self,
        db_engine,
        cohort_table,
        as_of_dates,
        features_schema_name=None,
        feature_start_time=None,
        features_ignore_cohort=False,
    ):
        self.db_engine = db_engine
        self.cohort_table_name = cohort_table
        self.as_of_dates = as_of_dates
        self.features_schema_name = features_schema_name
        self.feature_start_time = feature_start_time
        self.features_ignore_cohort = features_ignore_cohort

    @property
    @abstractmethod
    def final_feature_table_name(self):
        "The name of the final table with all features filled in (no missing values)"
        pass

    @property
    @abstractmethod
    def feature_columns(self):
        """
        The list of feature columns in the final, postimputation table

        Should exclude any index columns (e.g. entity id, date)
        """
        pass

    @property
    @abstractmethod
    def preinsert_queries(self):
        """
        Return all queries that should be run before inserting any data.

        Returns a list of queries/executable statements
        """
        pass

    @property
    @abstractmethod
    def insert_queries(self):
        """
        Return all inserts to populate this data. Each query in this list should be parallelizable.

        Returns a list of queries/executable statements
        """
        pass

    @property
    @abstractmethod
    def postinsert_queries(self):
        """
        Return all queries that should be run after inserting all data

        Returns a list of queries/executable statements
        """
        pass

    @property
    @abstractmethod
    def imputation_queries(self):
        """
        Return all queries that should be run to fill in missing data with imputed values.

        Returns a list of queries/executable statements
        """
        pass

    def _cohort_table_sub(self):
        """Helper function to ensure we only include state table records
        in our set of input dates and after the feature_start_time.
        """
        datestr = ", ".join(["'%s'::date" % dt for dt in self.as_of_dates])
        mindtstr = (
            " AND as_of_date >= '%s'::date" % (self.feature_start_time,)
            if self.feature_start_time is not None
            else ""
        )
        return """(
        SELECT *
        FROM {st}
        WHERE as_of_date IN ({datestr})
        {mindtstr})""".format(
            st=self.cohort_table_name,
            datestr=datestr,
            mindtstr=mindtstr,
        )

    def verify_no_nulls(self):
        """
        Verify that there are no nulls remaining in the imputed table

        Should raise an error if there are any.
        """

        query_template = """
            SELECT {cols}
            FROM {state_tbl} t1
            LEFT JOIN {aggs_tbl} t2 USING(entity_id, as_of_date)
            """
        cols_sql = ",\n".join(
            [
                """SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS "{col}" """.format(
                    col=column.name
                )
                for column in table_columns(self.final_feature_table_name, self.db_engine)
            ]
        )

        results = self.db_engine.execute(query_template.format(
            cols=cols_sql,
            state_tbl=self._cohort_table_sub(),
            aggs_tbl=self.final_feature_table_name,
        ))
        null_counts = results.first().items()
        nullcols = [col for (col, val) in null_counts if val > 0]

        if len(nullcols) > 0:
            raise ValueError(
                "Imputation failed for {} columns. Null values remain in: {}".format(
                    len(nullcols), nullcols
                )
            )

    def needs_features(self):
        imputed_table = self.final_feature_table_name

        if table_exists(imputed_table, self.db_engine):
            check_query = (
                f"select 1 from {self.cohort_table_name} "
                f"left join {imputed_table} "
                "using (entity_id, as_of_date) "
                f"where {imputed_table}.entity_id is null limit 1"
            )
            if self.db_engine.execute(check_query).scalar():
                logging.warning(
                    "Imputed feature table %s did not contain rows from the "
                    "entire cohort, need to rebuild features", imputed_table)
                return True
        else:
            logging.warning("Imputed feature table %s did not exist, "
                            "need to build features", imputed_table)
            return True
        logging.warning("Imputed feature table %s looks good, "
                        "skipping feature building!", imputed_table)
        return False

    def generate_preimpute_tasks(self, replace):
        if not replace and not self.needs_features():
            return {}
        return {
            "prepare": self.preinsert_queries,
            "inserts": self.insert_queries,
            "finalize": self.postinsert_queries
        }

    def generate_impute_tasks(self, replace):
        if not replace and not self.needs_features():
            return {}
        return {
            "prepare": self.imputation_queries,
            "inserts": [],
            "finalize": []
        }

    def process_table_task(self, task, verbose=False):
        if verbose:
            self.log_verbose_task_info(task)
        run_statements(task.get("prepare", []), self.db_engine)
        run_statements(task.get("inserts", []), self.db_engine)
        run_statements(task.get("finalize", []), self.db_engine)

    def run_preimputation(self, verbose=False):
        self.process_table_task(self.generate_preimpute_tasks(replace=True), verbose=verbose)

    def run_imputation(self, verbose=False):
        self.process_table_task(self.generate_impute_tasks(replace=True), verbose=verbose)
        self.verify_no_nulls()

    def log_verbose_task_info(self, task):
        prepares = task.get("prepare", [])
        inserts = task.get("inserts", [])
        finalize = task.get("finalize", [])
        logging.info("------------------")
        logging.info(
            "%s prepare queries, %s insert queries, %s finalize queries",
            len(prepares),
            len(inserts),
            len(finalize),
        )
        logging.info("------------------")
        logging.info("")
        logging.info("------------------")
        logging.info("PREPARATION QUERIES")
        logging.info("------------------")
        for query_num, query in enumerate(prepares, 1):
            logging.info("")
            logging.info(
                "prepare query %s: %s",
                query_num,
                sqlparse.format(str(query), reindent=True),
            )
        logging.info("------------------")
        logging.info("INSERT QUERIES")
        logging.info("------------------")
        for query_num, query in enumerate(inserts, 1):
            logging.info("")
            logging.info(
                "insert query %s: %s",
                query_num,
                sqlparse.format(str(query), reindent=True),
            )
        logging.info("------------------")
        logging.info("FINALIZE QUERIES")
        logging.info("------------------")
        for query_num, query in enumerate(finalize, 1):
            logging.info("")
            logging.info(
                "finalize query %s: %s",
                query_num,
                sqlparse.format(str(query), reindent=True),
            )
