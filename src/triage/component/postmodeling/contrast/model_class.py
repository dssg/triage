
import pandas as pd


from descriptors import cachedproperty


class ModelAnalyzer:

    def __init__(self, model_id, engine):
        self.model_id=model_id
        self.engine=engine

    @cachedproperty
    def metadata(self):
        return next(self.engine.execute(
                    f'''
                    WITH individual_model_ids_metadata AS(
                        SELECT m.model_id,
                           m.model_group_id,
                           m.hyperparameters,
                           m.model_hash,
                           m.train_end_time,
                           m.train_matrix_uuid,
                           m.training_label_timespan,
                           m.model_type,
                           mg.model_config
                        FROM triage_metadata.models m
                        JOIN triage_metadata.model_groups mg
                        USING (model_group_id)
                        AND model_id = {self.model_id}
                    ),
                    individual_model_id_matrices AS(
                        SELECT DISTINCT ON (matrix_uuid)
                           model_id,
                           matrix_uuid,
                           evaluation_start_time as as_of_date
                        FROM test_results.evaluations
                        WHERE model_id = ANY(
                            SELECT model_id
                            FROM individual_model_ids_metadata
                        )
                    )
                    SELECT metadata.*, test.*
                    FROM individual_model_ids_metadata AS metadata
                    LEFT JOIN individual_model_id_matrices AS test
                    USING(model_id);'''
            )
        )
    

    @property
    def model_group_id(self):
        return self.metadata['model_group_id']

    @property
    def model_type(self):
        return self.metadata['model_type']

    @property
    def hyperparameters(self):
        return self.metadata['hyperparameters']

    @property
    def model_hash(self):
        return self.metadata['model_hash']

    @property
    def train_matrix_uuid(self):
        return self.metadata['train_matrix_uuid']

    # TODO: Need to figure out how this would work when there are multiple matrices in the evaluations table
    @property
    def pred_matrix_uuid(self):
        return self.metadata['matrix_uuid']

    @property
    def as_of_date(self):
        return self.metadata['as_of_date']

    @property
    def train_end_time(self):
        return self.metadata['train_end_time']

    @property
    def train_label_timespan(self):
        return self.metadata['training_label_timespan']

    def get_predictions(self, matrix_uuid=None, fetch_null_labels=True):
        """Fetch the predictions from the DB for a given matrix
        
            args:
                matrix_uuid (optional):  
        
        """
        where_clause = f"WHERE model_id = {self.model_id}"

        if matrix_uuid is not None:
            where_clause += f" AND matrix_uuid={matrix_uuid}"

        if not fetch_null_labels:
            where_clause += f" AND label_value IS NOT NULL"

        query = f"""
            SELECT model_id,
                   entity_id,
                   as_of_date,
                   score,
                   label_value,
                   COALESCE(rank_abs_with_ties, RANK() OVER(ORDER BY score DESC)) AS rank_abs,
                   COALESCE(rank_pct_with_ties, percent_rank()
                   OVER(ORDER BY score DESC)) * 100 as rank_pct,
                   test_label_timespan
            FROM test_results.predictions
            {where_clause}        
        """

        preds = pd.read_sql(query, self.engine)

        #TODO: Maybe we should call the script to save predictions here?
        if preds.empty:
            raise RuntimeError(
                "No predictions were found in the database. Please run the add_predictions module to add predictions for the model"
            )

        return preds


    def get_evaluations(parameter, threshold, tiebreaker_ordering):
        pass

    def get_crosstabs(threshold_type, tiebreaker_ordering, matix_uuid=None):
        pass
