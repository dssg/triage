import pandas as pd
import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from sqlalchemy import text


class PreAudition:
    def __init__(self, db_engine, baseline_model_types=None):
        """Prepare the model_groups and train_end_times for Auditioner to use

        Args:
            db_engine: (sqlalchemy.engine)
            query: (string): cuztomized SQL query to pull model groups
            baseline_model_types: (list): optional list of model types to use to identify baseline models
        """
        self.db_engine = db_engine
        self.model_groups = None
        self.baseline_model_groups = None

        if baseline_model_types:
            baseline_types_list = ', '.join(["'%s'" % mt for mt in baseline_model_types])
            self.baseline_where = 'AND model_type IN (%s)' % baseline_types_list
            self.nonbaseline_where = 'AND model_type NOT IN (%s)' % baseline_types_list
        else:
            self.baseline_where = 'AND FALSE'
            self.nonbaseline_where = ''


    def get_model_groups_from_label(self, label_def):
        """A funciton to pull model groups based on label definition in order
        to prepare for Auditioner.

        Args:
            labed_def: (string) Label definition in triage schema

        """
        query = """
            SELECT DISTINCT(model_group_id)
            FROM triage_metadata.model_groups
            WHERE model_config->>'label_definition' = :label_definition
            {baseline_clause}
            """

        with self.db_engine.connect() as conn:
            model_groups = pd.read_sql(
                text(query.format(baseline_clause=self.nonbaseline_where)), 
                con=conn, 
                params={"label_definition": label_def}
            )
        self.model_groups = list(model_groups["model_group_id"])
        
        with self.db_engine.connect() as conn:
            baseline_model_groups = pd.read_sql(
                text(query.format(baseline_clause=self.baseline_where)), 
                con=conn, 
                params={"label_definition": label_def}
            )
        self.baseline_model_groups = list(baseline_model_groups["model_group_id"])

        return {'model_groups': self.model_groups, 'baseline_model_groups': self.baseline_model_groups}

    def get_model_groups_from_experiment(self, experiment_hash):
        """A function to pull model groups based on experiment_hash in order
        to prepare for Auditioner.

        Args:
            experiment_hash: (string) Experiment hash
        """
        query = """
            SELECT DISTINCT(model_group_id)
            FROM triage_metadata.models
            JOIN triage_metadata.experiment_models using (model_hash)
            WHERE experiment_hash = :experiment_hash
            {baseline_clause}
            """
        
        with self.db_engine.connect() as conn:
            model_groups = pd.read_sql(
                text(query.format(baseline_clause=self.nonbaseline_where)), 
                con=conn, 
                params={"experiment_hash": experiment_hash}
            )
        self.model_groups = list(model_groups["model_group_id"])

        with self.db_engine.connect() as conn:
            baseline_model_groups = pd.read_sql(
                text(query.format(baseline_clause=self.baseline_where)), 
                con=conn, 
                params={"experiment_hash": experiment_hash}
            )
        self.baseline_model_groups = list(baseline_model_groups["model_group_id"])

        return {'model_groups': self.model_groups, 'baseline_model_groups': self.baseline_model_groups}

    def get_model_groups(self, query):
        """A funciton to pull model groups based on customized query in order
        to preparre for Auditioner.

        Args:
            query: (string) SQL query for model groups
        """
        model_groups = pd.read_sql(query, con=self.db_engine)
        self.model_group = list(model_groups["model_group_id"])
        return self.model_group

    def get_train_end_times(self, after=None, query=None):
        """A function to get a list of train_end_times after certain time

        Args:
            after: (string) YYYY-MM-DD time format
            query: (string) SQL query for train_end_times
        """
        model_groups_stmt = ", ".join(map(str, self.model_groups + self.baseline_model_groups))
        if query is None:
            query = f"""
            SELECT DISTINCT train_end_time
            FROM triage_metadata.models
            WHERE model_group_id IN ({model_groups_stmt})
                AND train_end_time >= :after
            ORDER BY train_end_time
            ;
            """

        end_times = sorted(
            list(
                pd.read_sql(text(query), 
                            con=self.db_engine, 
                            params={"after": after})[
                    "train_end_time"
                ]
            )
        )
        return end_times
