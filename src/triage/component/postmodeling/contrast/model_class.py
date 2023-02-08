import ohio.ext.pandas
import pandas as pd
import logging

from descriptors import cachedproperty
from sqlalchemy import create_engine

from triage.component.catwalk.storage import ProjectStorage


class ModelAnalyzer:

    id_columns = ['entity_id', 'as_of_date']

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
                        WHERE model_id = {self.model_id}
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
            where_clause += f" AND matrix_uuid='{matrix_uuid}'"

        if not fetch_null_labels:
            where_clause += f" AND label_value IS NOT NULL"

        query = f"""
            SELECT model_id,
                   entity_id,
                   as_of_date,
                   score,
                   label_value,
                   rank_abs_with_ties,
                   rank_pct_with_ties,
                   rank_abs_no_ties,
                   rank_pct_no_ties, 
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

        return preds.set_index(self.id_columns)


    def get_evaluations(self, metrics=None, matrix_uuid=None, subset_hash=None):
        ''' 
        Get evaluations for the model from the DB

        Args:
            metrics Dict[str:List]): Optional. The metrics and parameters for evaluations. 
                                    A dictionary of type {metric:[thresholds]}
                                    If not specified, all the evaluations will be returned

            matrix_uuid (str): Optional. If model was evaluated using multiple matrices
                            one could get evaluations of a specific matrix. Defaults to fetching everything

            subset_hash (str): Optional. For fetching evaluations of a specific subset.    
        '''

        where_clause = f'WHERE model_id={self.model_id}'

        if matrix_uuid is not None:
            where_clause += f" AND matrix_uuid='{matrix_uuid}'"

        if subset_hash is not None:
            where_clause += f" AND subset_hash={subset_hash}"

        if metrics is not None:
            where_clause += " AND ("
            for i, metric in enumerate(metrics):
                parameters = metrics[metric]
                where_clause += f""" metric={metric} AND paramter in ('{"','".join(parameters)}')"""

                if i < len(metrics) - 1:
                    where_clause += "OR"

            where_clause += ") "

        q = f"""
            select
                model_id,
                matix_uuid,
                subset_hash,
                metric, 
                parameter,
                stochastic_value,
                num_labeled_above_threshold,               
                num_positive_labels
            from test_results.evaluations
            {where_clause}
        """

        evaluations = pd.read_csv(q, self.engine)

        return evaluations

    def crosstabs_pos_vs_neg(self, project_path, thresholds, matrix_uuid=None, push_to_db=True, table_name='crosstabs', return_df=True):
        """ Generate crosstabs for the predicted positives (top-k) vs the rest
    
        args:
            project_path (str): Path where the experiment artifacts (models and matrices) are stored
            thresholds (Dict{str: Union[float, int}]): A dictionary that maps threhold type to the threshold
                                                    The threshold type can be one of the rank columns in the test_results.predictions_table
            return_df (bool, optional): Whether to return the constructed df or just to store in the database
                                        Defaults to False (only writing to the db)
            
            matrix_uuid (str, optional): To run crosstabs for a different matrix than the validation matrix from the experiment

            push_to_db (bool, optional): Whether to write the results to the database. Defaults to True
            table_name (str, optional): Table name to use in the db's `test_results` schema. Defaults to crosstabs.
                                        If the table exists, results are appended

            return_df (bool, optional): Whether to return the crosstabs as a dataframe. Defaults to True
        """

        # metrics we calculate for positive and negative predictions
        positive_mean = lambda pos, neg: pos.mean(axis=0)
        negative_mean = lambda pos, neg: neg.mean(axis=0)
        positive_std = lambda pos, neg: pos.std(axis=0)
        negative_std = lambda pos, neg: neg.std(axis=0)
        ratio_positive_negative = lambda pos, neg: pos.mean(axis=0) / neg.mean(axis=0)
        positive_support = lambda pos, neg: (pos > 0).sum(axis=0)
        negative_support = lambda pos, neg: (neg > 0).sum(axis=0)


        crosstab_functions = [
            ("mean_predicted_positive", positive_mean),
            ("mean_predicted_negative", negative_mean),
            ("std_predicted_positive", positive_std),
            ("std_predicted_negative", negative_std),
            ("mean_ratio_predicted_positive_to_predicted_negative", ratio_positive_negative),
            ("support_predicted_positive", positive_support),
            ("support_predicted_negative", negative_support),
        ]


        if matrix_uuid is None:
            matrix_uuid = self.pred_matrix_uuid

        predictions = self.get_predictions(matrix_uuid=matrix_uuid)

    
        if predictions.empty:
            logging.error(f'No predictions found for {self.model_id} and {matrix_uuid}. Exiting!')
            raise ValueError(f'No predictions found {self.model_id} and {matrix_uuid}')


        # initializing the storage engines
        project_storage = ProjectStorage(project_path)
        matrix_storage_engine = project_storage.matrix_storage_engine()

        matrix_store = matrix_storage_engine.get_store(matrix_uuid=matrix_uuid)

        matrix = matrix_store.design_matrix
        
        labels = matrix_store.labels
        features = matrix.columns

        # joining the predictions to the model
        matrix = predictions.join(matrix, how='left')

        all_results = list() 
        
        for threshold_name, threshold in thresholds.items():
            logging.info(f'Crosstabs using threshold: {threshold_name} <= {threshold}')

            msk = matrix[threshold_name] <= threshold
            postive_preds = matrix[msk][features]
            negative_preds = matrix[~msk][features]

            temp_results = list()
            for name, func in crosstab_functions:
                logging.info(name)

                this_result = pd.DataFrame(func(postive_preds, negative_preds))
                this_result['metric'] = name
                temp_results.append(this_result)
            
            temp_results = pd.concat(temp_results)
            temp_results['threshold_type'] = threshold_name
            temp_results['threshold'] = threshold

            all_results.append(temp_results)

        crosstabs_df = pd.concat(all_results).reset_index()

        crosstabs_df.rename(columns={'index': 'feature', 0: 'value'}, inplace=True)
        crosstabs_df['model_id'] = self.model_id
        crosstabs_df['matrix_uuid'] = matrix_uuid

    
        if push_to_db:
            logging.info('Pushing the results to the DB')
            crosstabs_df.set_index(
                ['model_id', 'matrix_uuid', 'feature', 'metric', 'threshold_type', 'threshold'], inplace=True
            )

            # TODO: Figure out to change the owner of the table
            crosstabs_df.pg_copy_to(schema='test_results', name=table_name, con=self.engine, if_exists='append')

        if return_df:
            return crosstabs_df
