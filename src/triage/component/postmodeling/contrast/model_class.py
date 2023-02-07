from descriptors import cachedproperty


class ModelSomething:

    def __init__(self, model_id, engine):
        self.model_id=model_id
        self.engine=engine

    @cachedproperty
    def metadata(self):
        return next(self.engine.execute(
                    f'''WITH
                    individual_model_ids_metadata AS(
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
                        WHERE model_group_id = {self.model_group_id}
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
                    USING(model_id);''')
        )
    

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

    @property
    def pred_matrix_uuid(self):
        return self.metadata['matrix_uuid']

    @property
    def as_of_date(self):
        return self.metadata['as_of_date']


    def get_predictions(matrix_uuid=None):
        """Fetch the predictions from the DB for a given matrix"""
        pass


    def get_evaluations(parameter, threshold, tiebreaker_ordering):
        pass

    def get_crosstabs(threshold_type, tiebreaker_ordering, matix_uuid=None):
        pass
