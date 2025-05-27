import ohio.ext.pandas
import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.table as tab
import matplotlib.pyplot as plt
#from tabulate import tabulate

from IPython.display import display
import itertools

from descriptors import cachedproperty
from sqlalchemy import create_engine
from sklearn.calibration import calibration_curve
from sklearn import metrics
from scipy.stats import spearmanr, kendalltau

from triage.component.catwalk.storage import ProjectStorage
from triage.component.postmodeling.error_analysis import generate_error_analysis, output_all_analysis
from triage.database_reflection import table_exists
from triage.component.catwalk.utils import sort_predictions_and_labels

from triage.component.postmodeling.add_predictions import add_predictions

from triage.component.postmodeling.utils.plot_functions import (
    plot_score_distribution, plot_score_distribution_by_label, 
    plot_precision_recall_at_k, plot_feature_importance, plot_pairwise_comparison_heatmap
)
id_columns = ['entity_id', 'as_of_date']

class ModelAnalyzer:
    
    def __init__(self, model_id, engine):
        self.model_id=model_id
        self.engine=engine

    @cachedproperty
    def metadata(self):
        return next(self.engine.execute(
                    f'''
                    select 
                    m.model_group_id,
                    m.hyperparameters,
                    m.model_hash,
                    m.train_end_time,
                    m.train_matrix_uuid,
                    m.training_label_timespan,
                    m.model_type,
                    (string_to_array(m.model_type, '.'))[array_length(string_to_array(m.model_type, '.'), 1)] as model_type_short,
                    mg.model_config
                    FROM triage_metadata.models m
                    JOIN triage_metadata.model_groups mg
                    USING (model_group_id)
                    WHERE model_id = {self.model_id}
                    '''
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

    @property
    def test_matrix_uuid(self):
        # return self.metadata['matrix_uuid']
        return next(self.engine.execute(
                    f'''
                    select 
                    matrix_uuid
                    FROM test_results.prediction_metadata
                    WHERE model_id = {self.model_id}
                    '''
            )
        )

    # @property
    # def as_of_date(self):
    #     return self.metadata['as_of_date']

    @property
    def train_end_time(self):
        return self.metadata['train_end_time']

    @property
    def train_label_timespan(self):
        return self.metadata['training_label_timespan']

    def get_model_description(self):
        desc = f'{self.model_type.split(".")[-1]}: {self.hyperparameters}'
        return desc
    
    def get_predictions(self, matrix_uuid=None, fetch_null_labels=True, subset_hash=None, plot_distribution=False, **kwargs):
        """ Fetch the predictions for the model from the DB
            Args:
                matrix_uuid (str): Optional. If model was evaluated using multiple matrices
                            one could get predictions of a specific matrix. Defaults to fetching everything

                fetch_null_labels (bool): Whether to fetch null labels or not. Defaults to True
                subset_hash (str): Optional. For fetching predictions of a specific cohort subset.
        
        """
        where_clause = f"WHERE model_id = {self.model_id}"

        if matrix_uuid is not None:
            where_clause += f" AND matrix_uuid='{matrix_uuid}'"

        if not fetch_null_labels:
            where_clause += f" AND label_value IS NOT NULL"


        predictions_table='test_results.predictions'
         
        if subset_hash:
            # get subset table name
            q = f"select config from triage_metadata.subsets where subset_hash='{subset_hash}'"
            config_df = pd.read_sql(q, self.engine)
            table_name = f"subset_{config_df.iloc[0]['config']['name']}_{subset_hash}"
            predictions_table += f" preds join {table_name} subset on preds.entity_id = subset.entity_id and preds.as_of_date = subset.as_of_date"

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
            FROM {predictions_table}
            {where_clause}        
        """

        predictions = pd.read_sql(query, self.engine)

        #TODO: Maybe we should call the script to save predictions here?
        if predictions.empty:
            logging.warning(f'No predictions were found in {predictions_table} for model_id {self.model_id}. Returning empty dataframe!')
            # raise RuntimeError(
                # "No predictions were found in the database. Please run the add_predictions module to add predictions for the model"
            # )
            return predictions 
        
        if plot_distribution:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
            plot_score_distribution(predictions, nbins=kwargs.get('nbins'), ax=axes[0])
            plot_score_distribution_by_label(predictions, nbins=kwargs.get('nbins'), ax=axes[1])

        return predictions.set_index(id_columns)

    
    def get_top_k(self, threshold_type, threshold, matrix_uuid=None):
        """ Fetch the k intities with highest model score for a given model
        
            Args:
                threshold_type (str): Type of the ranking to use. Has to be one of the four ranking types used in triage
                    - rank_pct_no_ties 
                    - rank_pct_with_ties
                    - rank_abs_no_ties
                    - rank_abs_with_ties
                threshold (Union[float, int]): The threshold rank for creating the list. Int for 'rank_abs_*' and Float for 'rank_pct_*'
                matrix_uuid (optional, str): If a list should be generated out of a matrix that were not used to validate the model during the experiment
        """
        
        where_clause = f'''where model_id={self.model_id}
            AND {threshold_type} <= {threshold}'''
        
        if matrix_uuid is not None:
            where_clause += f" AND matrix_uuid='{matrix_uuid}'"

        q = f"""
            select 
                entity_id, 
                as_of_date,
                score,
                label_value,
                rank_abs_no_ties,
                rank_abs_with_ties,
                rank_pct_no_ties,
                rank_pct_with_ties,
                matrix_uuid
            from test_results.predictions
            {where_clause}
        """

        top_k = pd.read_sql(q, self.engine)

        return top_k

    # TODO: Write a bias function 
    # This could learn from modeling report bias function
    def get_bias_metrics(self, parameter, metric, subset_hash='', attribute_name=None, attribute_value=None, group_size_threshold=0.01):
        """ Fetch the bias metrics for the model
        
            Args:
                parameter(str): The parameter for cut off
        """
        
        q = f'''
            SELECT 
            attribute_name,
            attribute_value,
            group_label_pos,
            group_size,
            group_size_pct,
            prev,
            {metric}, 
            {metric}_disparity,
            tpr_ref_group_value 
            FROM test_results.aequitas a
            where model_id = {self.model_id}
            and subset_hash = '{subset_hash}'
            and "parameter" = '{parameter}'
            and tie_breaker = 'worst'
            and group_size::float / total_entities > {group_size_threshold}
        '''
        
        if attribute_name is not None:
            q += f" and attribute_name='{attribute_name}'"
        if attribute_value is not None:
            q += f" and attribute_value='{attribute_value}'"
                
        df = pd.read_sql(q, self.engine)

        if df.empty:
            logging.error(f'No bias metrcis were found for the attributes provided. Returning empty dataframe!')
            
        return df


    def get_evaluations(self, metrics=None, matrix_uuid=None, subset_hash=None, plot_prk=False, **kwargs):
        ''' 
        Get evaluations for the model from the DB

        Args:
            metrics Dict[str:List]): Optional. The metrics and parameters for evaluations. 
                                    A dictionary of type {metric:[thresholds]}
                                    If not specified, all the evaluations will be returned

            matrix_uuid (str): Optional. If model was evaluated using multiple matrices
                            one could get evaluations of a specific matrix. Defaults to fetching everything

            subset_hash (str): Optional. For fetching evaluations of a specific subset.    
            plot_prk (bool): Whether to plot the precision-recall curve for the evaluations. Defaults to False
        '''

        where_clause = f'WHERE model_id={self.model_id}'

        if matrix_uuid is not None:
            where_clause += f" AND matrix_uuid='{matrix_uuid}'"

        if subset_hash is not None:
            where_clause += f" AND subset_hash='{subset_hash}'"
        else:
            where_clause += f" AND subset_hash=''"

        if metrics is not None:
            where_clause += " AND ("
            for i, metric in enumerate(metrics):
                parameters = metrics[metric]
                where_clause += f""" metric='{metric}' AND parameter in ('{"','".join(parameters)}')"""

                if i < len(metrics) - 1:
                    where_clause += "OR"

            where_clause += ") "

        q = f"""
            select
                model_id,
                matrix_uuid,
                subset_hash,
                metric, 
                parameter,
                stochastic_value as metric_value,
                num_labeled_above_threshold,               
                num_positive_labels
            from test_results.evaluations
            {where_clause}
            order by metric, num_labeled_above_threshold
        """

        evaluations = pd.read_sql(q, self.engine)
        
        if evaluations.empty:
            logging.warning(f'No evaluations were found in test_results.evaluations for model_id {self.model_id}. Returning empty dataframe!')
            
        if len(evaluations.matrix_uuid.unique()) > 1:
            logging.warning(f'Evaluations for {len(evaluations.matrix_uuid.unique())} validation matrices were found for model_id {self.model_id}. Please check the evaluations table!')
        
        if plot_prk:
            ax = plot_precision_recall_at_k(predictions=self.get_predictions(matrix_uuid=matrix_uuid, subset_hash=subset_hash), evaluations=evaluations, ax=kwargs.get('ax'))
            ax.set_title(f'Model: {self.model_id}, Group: {self.model_group_id}')
        
        return evaluations

    def get_feature_importances(self, n_top_features=20, plot=False, **kwargs):
        """ Get the n most important features for the model
            Args:
                n_top_features (int): The number of features to return. Defaults to 20
        
        """
        logging.debug(f'Fetching feature importance from db for model id: {self.model_id}')
        features = pd.read_sql(
           f'''
           select
                feature,
                feature_importance as importance,
                rank_abs
           FROM train_results.feature_importances
           WHERE model_id = {self.model_id}
           and rank_abs <= {n_top_features}
           and abs(feature_importance) > 0 
           order by rank_abs
           ''', con=self.engine)
        
        if features.empty:
            logging.warning(f'No feature importances were found for model_id {self.model_id}. Returning empty dataframe!')
            
        if plot:
            ax = plot_feature_importance(features, n_top_features=n_top_features, **kwargs)
            ax.set_title(f'Model: {self.model_id}, Group: {self.model_group_id}')
        
        return features
    
    def get_feature_group_importances(self, plot=True, **kwargs):
        """
        Get the top most important feature groups as identified by the maximum importance of any feature in the group
        Returns all feature groups, not limited to the top n groups
        """
        
        # Fetching all the feature groups
        q = f'''
            select 
            replace(jsonb_object_keys(mat.feature_dictionary), '_aggregation_imputed', '') as feature_group
            from triage_metadata.models mod left join triage_metadata.matrices mat 
            on mod.train_matrix_uuid = mat.matrix_uuid 
            where mod.model_id = {self.model_id}
        '''
        feature_groups = pd.read_sql(q, self.engine)['feature_group'].tolist()
        
        case_part = ''
        for fg in feature_groups:
            case_part = case_part + "\nWHEN feature like '{fg}%%' THEN '{fg}'".format(fg=fg)

        # get feature group importances
        feature_group_importance = pd.read_sql(f"""
            with raw_importances as (
                select 
                    model_id,
                    feature,
                    feature_importance,
                    CASE {case_part}
                    ELSE 'No feature group'
                    END as feature_group
                FROM train_results.feature_importances
                WHERE model_id = {self.model_id}
            )
            SELECT
            model_id,
            feature_group as feature,
            max(abs(feature_importance)) as importance
            FROM raw_importances
            GROUP BY feature_group, model_id
        """, con=self.engine)
        
        if plot:
            ax = plot_feature_importance(feature_group_importance, n_top_features=len(feature_group_importance), **kwargs)
            ax.set_title(f'Model: {self.model_id}, Group: {self.model_group_id}')
        
        return feature_group_importance

    def calculate_crosstab_pos_vs_neg(self, project_path, thresholds, matrix_uuid=None, push_to_db=True, table_name='crosstabs', return_df=True, replace=True, predictions_table='test_results.predictions'):
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
        # mean_ratio = lambda pos, neg: max(ratio_positive_negative(pos, neg).at[0], (float(1) / ratio_positive_negative(pos, neg)).at[0])
        positive_support = lambda pos, neg: (pos > 0).sum(axis=0)
        negative_support = lambda pos, neg: (neg > 0).sum(axis=0)
        positive_support_pct = lambda pos, neg: round((pos > 0).sum(axis=0).astype(float) / len(pos), 3)
        negative_support_pct = lambda pos, neg: round((neg > 0).sum(axis=0).astype(float) / len(neg), 3)

        def mean_ratio(pos, neg):
            pos_over_neg = ratio_positive_negative(pos, neg)
            neg_over_pos = float(1) / pos_over_neg

            df = pd.DataFrame([pos_over_neg, neg_over_pos])

            return df.max()

        crosstab_functions = [
            ("mean_predicted_positive", positive_mean),
            ("mean_predicted_negative", negative_mean),
            ("std_predicted_positive", positive_std),
            ("std_predicted_negative", negative_std),
            ("mean_ratio", mean_ratio),
            ("support_predicted_positive", positive_support),
            ("support_predicted_negative", negative_support),
            ("support_pct_predicted_positive", positive_support_pct),
            ("support_pct_predicted_negative", negative_support_pct)
        ]


        if matrix_uuid is None:
            matrix_uuids = self.test_matrix_uuid
            if len(matrix_uuids) > 1:
                logging.warning(f'Multiple matrices were found for model_id {self.model_id}. Using the first one: {matrix_uuids[0]}')
            
            matrix_uuid = matrix_uuids[0]
            logging.debug(f'Matrix uuid set to: {matrix_uuid}')

        predictions = self.get_predictions(matrix_uuid=matrix_uuid)

    
        if predictions.empty:
            logging.error(f'No predictions found for {self.model_id} and {matrix_uuid}. Exiting!')
            return None
            # raise ValueError(f'No predictions found {self.model_id} and {matrix_uuid}')

        # Check whether the table exists
        if table_exists(f'test_results.{table_name}', self.engine):
            # checking whether the crosstabs already exist for the model
            logging.debug(f'Checking whether crosstabs already exist for the model {self.model_id} and {matrix_uuid}')
            q = f"select * from test_results.{table_name} where model_id={self.model_id} and matrix_uuid='{matrix_uuid}';"
            df = pd.read_sql(q, self.engine)
            
            if not df.empty:
                logging.warning(f'Crosstabs aleady exist for model {self.model_id} and matrix_uuid={matrix_uuid}')

                if replace:
                    logging.warning('Deleting the existing crosstabs!')
                    with self.engine.connect() as conn:
                        conn.execute(f"delete from test_results.{table_name} where model_id={self.model_id} and matrix_uuid='{matrix_uuid}';")
                else:
                    logging.info(f"Replace set to False. Not calculating crosstabs for model {self.model_id} and matrix_uuid='{matrix_uuid}';")
                    if return_df: return df 
                    else: return 
            
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
            
            #TODO: FIX THIS!
            logging.error('Currently this function does not work!')
            # with self.engine.begin() as conn:
            # TODO: Figure out to change the owner of the table
            # crosstabs_df.pg_copy_to(schema='test_results', name=table_name, con=conn, if_exists='append')

        if return_df:
            return crosstabs_df
        

    def display_crosstabs_pos_vs_neg(
            self, 
            threshold_type, threshold, table_name='crosstabs',
            display_n_features=40,
            filter_features=None,
            support_threshold=0.1,
            return_df=True,
            show_plot=True,
            matrix_uuid=None,
            ax=None):
        """ Display the crosstabs for the model

        Args:
            threshold_type (str): Type of rank threshold to use in splitting predicted positives from negatives. 
                                Has to be one of the rank columns in the test_results.predictions_table

            threshold (Union[int, float]): The rank threshold of the specified type. If the threshold type is an absolute, integer. 
                                        If percentage, should be a float between 0 and 1
        
            table_name (str, optional): Table name to fetch crosstabs from (`test_results` schema). Defaults to 'crosstabs'.
        """

        q = f"""
            select 
                model_id, 
                feature, 
                metric, 
                abs(value) as value 
            from test_results.{table_name} 
            where threshold_type = '{threshold_type}'
            and threshold = {threshold}
            and metric in (
                'mean_ratio',
                'mean_predicted_positive',
                'mean_predicted_negative',
                'support_pct_predicted_positive',
                'support_pct_predicted_negative'
            )
            and model_id = {self.model_id}            
        """

        if matrix_uuid is not None:
            q = q + f"\n and matrix_uuid='{matrix_uuid}'"
        else:
            matrix_uuids = self.test_matrix_uuid
            if len(matrix_uuids) > 1:
                logging.warning(f'Multiple matrices were found for model_id {self.model_id}. Using the first one: {matrix_uuids[0]}')

        ct = pd.read_sql(q, self.engine)

        if ct.empty:
            logging.error(
                f'''Crosstabs not found for model {self.model_id} and matrix={matrix_uuid} in table test_results.{table_name}. 
                Please use crosstabs_pos_vs_neg function to generate crosstabs'''
            )
            raise ValueError('Crosstabs not found!')

        # Creating the pivot table to convert to wide format indexed by the column
        pivot_table = pd.pivot(
            ct, 
            index='feature', 
            columns='metric', 
            values='value'
        )

        # Shortening the names, and removing the index names to plot more cleanly
        pivot_table.rename(
            columns={
                'mean_ratio': 'ratio',
                'mean_predicted_positive': '(+)mean',
                'mean_predicted_negative': '(-)mean',
                'support_pct_predicted_negative': '(-)supp',
                'support_pct_predicted_positive': '(+)supp'
            }, 
            inplace=True
        )
        pivot_table.index.name='feature_name'
        pivot_table.columns.name=''

        if filter_features is not None:
            return pivot_table.loc[filter_features]

        
        # Filtering by the support threshold
        msk1 = pivot_table['(+)supp'] > support_threshold
        # Features with highest postive : negative ratio
        # df1 = pivot_table[msk].sort_values(
        #     'ratio', 
        #      ascending = False
        # ).head(display_n_features)


        msk2 = pivot_table['(-)supp'] > support_threshold
        # Features with the highest negative : positive ratio
        # df2 = pivot_table[msk].sort_values(
        #     ['ratio', '(-)supp'], 
        #      ascending = [True, False]
        # ).head(display_n_features)

        df = pivot_table[msk1 | msk2].sort_values(
            'ratio', 
             ascending = False
        ).head(display_n_features)

        if show_plot:
            if ax is None: 
                fig, ax = plt.subplots(figsize=(4, 1 + (display_n_features *2) / 5), dpi=100)

            t = tab.table(
                ax, 
                cellText=df.values.round(1), 
                colLabels=df.columns, 
                rowLoc='right',
                rowLabels=df.index, 
                bbox=[0, 0, 1, 1]
            )
        
            ax.set_title(
                f'Model Group: {self.model_group_id} | Train end: {self.train_end_time} | Model: {self.model_id}\n',
                fontdict={
                    'fontsize': 10,
                    'verticalalignment': 'center',
                    'horizontalalignment': 'center'
                },
                x=-0.1
            )
            ax.axis('off')

            # Table formatting
            t.auto_set_column_width([0, 1, 2, 3, 4])
            for key, cell in t.get_celld().items():
                cell.set_fontsize(9)
                cell.set_linewidth(0.1)
                # cell.PAD = 0.01
                
        
        title_str = f'model_group: {self.model_group_id} | train_end_time: {self.train_end_time} | model_id: {self.model_id}'
        # print(title_str)
        # print(tabulate(df.round(1), headers='keys', tablefmt='RST'))


        if return_df:
            df.style.set_caption(title_str)
            return df    
        

    def error_analysis(self, project_path):
        """
        Generates three main error anlaysis with Decision Trees in order to identity 
        what are the features with most importance when the model make mistakes 
        in negative labels (FN), positive labels (FP) and both, negative and 
        positive label errors (FN & FP).

        Args:
            project_path (string): Path for the output of the project 

        Returns: 
            error_analysis_results (list): List of dictionaries with all the error
            analysis made.
        """
        model_id = self.model_id
        connection = self.engine
        error_analysis_results = generate_error_analysis(model_id, connection, project_path=project_path)
        #TODO do we want to 
        return error_analysis_results
    
 
    
    # TODO: make this more general purpose
    def plot_bias_threshold_curve(self, attribute_name, attribute_value, bias_metric, ax):
        bias_df = self.get_aequitas()
        # limit to specific attribute+value
        bias_df = bias_df.loc[(bias_df['attribute_name']==attribute_name) & (bias_df['attribute_value']==attribute_value)]

        bias_df['perc_points'] = [x.split('_')[0] for x in bias_df['parameter'].tolist()]
        bias_df['perc_points'] = pd.to_numeric(bias_df['perc_points'])

        msk_pct = bias_df['parameter'].str.contains('pct')

        # plot precision
        sns.lineplot(
            x='perc_points',
            y=bias_metric, 
            data=bias_df[msk_pct], 
            label=self.model_group_id,
            ax=ax, 
            estimator='mean', ci='sd'
        )
        ax.set_xlabel('List size percentage (k%)')
        #ax.set_ylabel(f'{bias_metric} for {attribute_name}:{attribute_value}')
        return ax

    # TODO: This doesn't seem to work any more (doesn't throw an error)
    def plot_precision_threshold_curve(self, ax, matrix_uuid=None):
        eval_df = self.get_evaluations(matrix_uuid=matrix_uuid)

        eval_df['perc_points'] = [x.split('_')[0] for x in eval_df['parameter'].tolist()]
        eval_df['perc_points'] = pd.to_numeric(eval_df['perc_points'])

        msk_prec = eval_df['metric']=='precision@'
        msk_pct = eval_df['parameter'].str.contains('pct')

        # plot precision
        sns.lineplot(
            x='perc_points',
            y='metric_value', 
            data=eval_df[msk_pct & msk_prec], 
            label=self.model_group_id,
            ax=ax, 
            estimator='mean', ci='sd'
        )
        ax.set_xlabel('List size percentage (k%)')
        ax.set_ylabel('Precision')
        ax.set_title(f'Model: {self.model_id}, Group: {self.model_group_id}')
        return ax

                 
    # TODO: Figure out how to finish this
    def plot_calibration_curve(self, ax, nbins=20, min_score=0, max_score=1, matrix_uuid=None):

        """
        Plot the calibration curve of predicted probability scores across model groups

        Args:
            ax: matplotlib Axes object to plot on
            nbins (int): the number of bins to define the calibration curve
            min_score (optional, float between 0 and 1): zoom in on the plot, set x and y min axis value
            max_score (optional, float between 0 and 1): zoom in on the plot, set x and y max axis value
            matrix_uuid (optional): specify a matrix to get predictions for

        Return: 
            ax: modified matplotlib Axes object
        """
        # TODO can the bins be adjusted when zooming in on the plot?
        scores = self.get_predictions(matrix_uuid) # TODO what about null labels?
        scores = scores.dropna()
        # TODO do we want to restrict to label=1 only?
        cal_x, cal_y = calibration_curve(scores['label_value'], scores['score'], n_bins=nbins)
        sns.lineplot(
            x=cal_x,
            y=cal_y, 
            marker='o', 
            ax=ax
        )
        # plot perfectly calibrated line y = x for comparison
        probabilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # filter out probabilities
        probabilities_plot = []
        for p in probabilities:
            probabilities_plot.append(p)
        sns.lineplot(
            data = pd.DataFrame(list(zip(probabilities, probabilities)), columns=['x', 'y']),
            x = 'x', 
            y = 'y', 
            color = 'gray',
            linestyle='dashed',
            ax=ax
        )
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of data')
        ax.set_title(f'Model: {self.model_id}, Group: {self.model_group_id}')
        ax.set_xlim([min_score, max_score])
        ax.set_ylim([min_score, max_score])
        return ax
    
    
    # NOTE: proceed with caution using this function (reason given inside)
    def save_predictions(self, project_path, replace=False):
        """
        Save the predictions to the project path
        Args:
            project_path (str): Path where the experiment artifacts (models and matrices) are stored
            replace (bool): Whether to replace the existing predictions or not. Defaults to False
        """
        
        # TODO: should we support adding predictions of a single model?
        # In the current form, this function call will save predictions for all model_ids with same model_group_id and train_end_time
        # This is not a common scenario (multiple model_ids for same group and time), but it is possible and a class tied to a specific model_id should not trigger actions related to other model_ids
        add_predictions(
            db_engine=self.engine,
            project_path=project_path,
            model_groups=[self.model_group_id],
            train_end_times_range= {'range_start_date': self.train_end_time, 'range_end_date': self.train_end_time},
            replace=replace,
        )
        

        
"""This class is still WIP. Proceed with caution"""
class ModelGroupAnalyzer: 

    def __init__(self, engine, model_groups, experiment_hashes, project_path=None, train_end_times=None, use_all_model_groups=False) -> None:
        self.model_groups = model_groups
        self.experiment_hashes = experiment_hashes # TODO made experiment hashes into a list to plot models from different experiments for MVESC, there's probably a better way to generalize this
        self.engine = engine
        self.project_path = project_path
        if use_all_model_groups: # shortcut to listing out all model groups for an experiment
            self.use_all_model_groups()
        self.models = self.get_model_ids()

        #TODO: Would be good to let the user restrict the train_end_times included in the report

        
    @property
    def model_ids(self):
        model_ids = list()
        for mg, d in self.models.items():
            for t, mod_obj in d.items():
                model_ids.append(mod_obj.model_id)

    @property
    def model_types(self):
        pass

    def use_all_model_groups(self):
        experiment_hashes = "','".join(self.experiment_hashes)
        q = f"""
                select model_group_id 
                from triage_metadata.models
                    join triage_metadata.experiment_models using (model_hash)
                where experiment_hash in ('{experiment_hashes}')
            """
        model_groups = pd.read_sql(q, self.engine)
        self.model_groups = model_groups['model_group_id'].to_list()

    # TODO: revise this to show the only the model_group_id, list of model ids, model type, and hyperparameters
    def display_model_groups(self):
        data_dict = []
        for mg in self.model_groups:
            for train_end_time in self.models[mg]:
                model_analyzer = self.models[mg][train_end_time]
                data_dict.append([mg, train_end_time, model_analyzer.model_id, model_analyzer.model_type, model_analyzer.hyperparameters])
        all_models = pd.DataFrame(data_dict, columns=['model_group_id', 'train_end_time', 'model_id', 'model_type', 'hyperparameters'])

        # displaying the model_group_ids, model_type, and the hyperparameters
        to_print = all_models.groupby('model_group_id').nth(1)[['model_type', 'hyperparameters']].reset_index().to_dict(orient='records')

        for m in to_print:
            print(m)

        return all_models
    
    def print_model_summary(self):
        ''' This is mostly to be used as a key for modeling report plots (just as a model group number to model group mapping)'''
        data_dict = []
        for mg in self.model_groups:
            for train_end_time in self.models[mg]:
                model_analyzer = self.models[mg][train_end_time]
                data_dict.append([mg, train_end_time, model_analyzer.model_id, model_analyzer.model_type, model_analyzer.hyperparameters])
        
        all_models = pd.DataFrame(data_dict, columns=['model_group_id', 'train_end_time', 'model_id', 'model_type', 'hyperparameters'])
        # to_print = all_models.groupby('model_group_id').nth(1)[['model_type', 'hyperparameters']].reset_index().to_dict(orient='records')

        for i, model in all_models.groupby('model_group_id').nth(1)[['model_type', 'hyperparameters']].reset_index().iterrows():
            print(f"{model['model_group_id']} - {model['model_type']} with ({model['hyperparameters']}) ")

    def cohort_summary(self):
        q = f"""
            select distinct on(train_end_time)
                -- matrix_uuid,
                evaluation_start_time as train_end_time,
                num_labeled_examples as cohort_size,
                num_positive_labels,
                case when num_labeled_examples > 0 then num_positive_labels::float/num_labeled_examples else 0 end as label_base_rate
            from triage_metadata.experiment_matrices join test_results.evaluations using(matrix_uuid)
            where experiment_hash in ('{"','".join(self.experiment_hashes)}') and subset_hash = ''
            order by 1
        """

        matrices = pd.read_sql(q, self.engine)

        print(matrices)
    

    def subset_summary(self, subset_hash):
        q = f"""
            select distinct on(train_end_time)
                -- matrix_uuid,
                evaluation_start_time as train_end_time,
                num_labeled_examples as cohort_size,
                num_positive_labels,
                case when num_labeled_examples > 0 then num_positive_labels::float/num_labeled_examples else 0 end as label_base_rate
            from triage_metadata.experiment_matrices join test_results.evaluations using(matrix_uuid)
            where experiment_hash in ('{"','".join(self.experiment_hashes)}') 
                and subset_hash = '{subset_hash}'
            order by 1
        """

        matrices = pd.read_sql(q, self.engine)

        print(matrices)

    def plot_model_group_performance(self, metric, parameter):
        pass        

        # q = """
        #     select
        
        # """

    def get_model_ids(self):
        """ For the model group ids, fetch the model_ids and initialize the datastructure

            The data structure is a dictionary of dictionaries that maps the individual models of groups to their ModelAnalyzer class
                {model_group_id: {train_end_time: ModelAnalyzer(model_id)}}
                Here, the train_end_time is saved as a string of with format 'YYYY-MM-DD'
        """

        model_groups = "', '".join([str(x) for x in self.model_groups])
        experiment_hashes = "', '".join(self.experiment_hashes)
        q = f"""
            select distinct on (model_group_id, train_end_time)
                model_id, 
                train_end_time::date,
                model_group_id
            from triage_metadata.models 
                join triage_metadata.experiment_models using(model_hash)
            where experiment_hash in ('{experiment_hashes}')
            and model_group_id in ('{model_groups}')        
            """  
        # TODO do we really need experiment_hashes here? can we query with only model_group_ids?

        # TODO: modify to remove pandas
        models = pd.read_sql(q, self.engine).to_dict(orient='records')

        d = dict()
        for experiment_hash in self.experiment_hashes:
            q = f"""
                select distinct on (model_group_id, train_end_time)
                    model_id, 
                    to_char(train_end_time::date, 'YYYY-MM-DD') as train_end_time,
                    model_group_id
                from triage_metadata.models 
                    join triage_metadata.experiment_models using(model_hash)
                where experiment_hash='{experiment_hash}'
                and model_group_id in ('{model_groups}')        
                """  

            # TODO: modify to remove pandas
            models = pd.read_sql(q, self.engine).to_dict(orient='records')

            for m in models:
                if m['model_group_id'] in d:
                    d[m['model_group_id']][m['train_end_time']] = SingleModelAnalyzer(m['model_id'], self.engine)
                else:
                    d[m['model_group_id']] = {m['train_end_time']: SingleModelAnalyzer(m['model_id'], self.engine)}

        return d 
    
    def _get_subplots(self, subplot_width=3, subplot_len=None, n_rows=None, n_cols=None, sharey=False, sharex=False):
        """"""

        if subplot_len is None:
            subplot_len = subplot_width

        # num of rows == number of train_end_times
        if n_rows is None: 
            n_rows = len(self.models[self.model_groups[0]])
        
        # num of cols == number of model groups
        if n_cols is None: 
            n_cols = len(self.model_groups)
        
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize = (subplot_width*n_cols, subplot_len*n_rows),
            dpi=100,
            sharey=sharey, 
            sharex=sharex
        )

        return fig, axes


    def _make_plot_grid(self, plot_type, subplot_width=3, subplot_len=None, sharey=False, sharex=False, **kw):
        """
            Abstracts out generating the plot grid (time x model group) for comparing
            model_scores,  
        """
        fig, axes = self._get_subplots(subplot_width=subplot_width, subplot_len=subplot_len, sharey=sharey, sharex=sharex)
        
        print(len(axes), len(axes[0]))

        for j, mg in enumerate(self.models):
            for i, train_end_time in enumerate(self.models[mg]):
                model_analyzer = self.models[mg][train_end_time]

                plot_func = getattr(model_analyzer, plot_type)

                if (len(self.models) == 1) or (len(self.models[mg]) == 1):
                    if (len(self.models) == 1) and (len(self.models[mg]) == 1):
                        ax = axes
                    else:
                        ax = axes[i]
                else:
                    ax = axes[i, j]
                ax = plot_func(ax=ax, **kw)

                if j==0:
                    ax.set_ylabel(f'{train_end_time}')
                else:
                    ax.set_ylabel('')
                
                ax.set_xlabel('')

        fig.tight_layout()

        return fig

    def plot_score_distributions(self, use_labels=False):
        """for the model group ids plot score grid"""

        if use_labels:
            self._make_plot_grid(plot_type='plot_score_label_distribution')
        else:
            self._make_plot_grid(plot_type='plot_score_distribution')
        
    def plot_calibration_curves(self):
        """calibration curves for all models"""
        self._make_plot_grid(plot_type='plot_calibration_curve')

    def plot_prk_curves(self, **kw):
        self._make_plot_grid(plot_type='plot_precision_recall_curve', **kw)
        
    def plot_recall_curves_overlaid(self, n_splits=None, **kw,):
        # Number of columns    
        if n_splits is None:
            n_cols = len(self.models[self.model_groups[0]])
        else:
            n_cols = n_splits
            
        fig, axes = self._get_subplots(
            subplot_width=3, 
            subplot_len=3, 
            sharey=True, 
            sharex=True, 
            n_cols=n_cols, 
            n_rows=1
        )
        
        as_of_dates = list(sorted(self.models[self.model_groups[0]].keys()))[-n_cols:]
        
        for i, aod in enumerate(as_of_dates):
            for model_group in sorted(self.model_groups):
                mod = self.models[model_group].get(aod)
                if mod is None:
                    continue

                mod.plot_precision_recall_curve(only_recall=True,ax = axes[i], title_string=aod)
                axes[i].set(alpha=0.1)
                axes[i].legend().remove()
                axes[i].set_xlabel('Population pct (k%)')
                axes[i].set_ylabel('recall@k')
                sns.despine()
        plt.tight_layout()
        axes[-1].legend(sorted(self.model_groups))
        
        
        

  
    def plot_bias_threshold(self, attribute_name, attribute_values, bias_metric):
        """
            Plot bias_metric for the specified list of attribute_values for a particular attribute_name across different thresholds (list %)
        """
        fig, axes = self._get_subplots(subplot_width=6, n_rows=len(attribute_values), n_cols=len(self.models[self.model_groups[0]]))
        for _, mg in enumerate(self.models):
            for i, attribute_value in enumerate(attribute_values):
                for j, train_end_time in enumerate(self.models[mg]):
                    mode_analyzer = self.models[mg][train_end_time]

                    mode_analyzer.plot_bias_threshold_curve(
                        attribute_name=attribute_name,
                        attribute_value=attribute_value,
                        bias_metric=bias_metric,
                        ax=axes[i, j]
                    )
                    if j==0:
                        axes[i, j].set_ylabel(f'{attribute_name}:{attribute_value}')
                    else:
                        axes[i, j].set_ylabel('')
                    if i == 0:
                        axes[i, j].set_title(f'{train_end_time}')
                    else:
                        axes[i, j].set_title('')
        fig.suptitle(f"{bias_metric} Threshold Curve for {attribute_name}")
        fig.tight_layout()

    def plot_precision_threshold(self):
        """
            Plot precision against threshold (list %)
        """
        if len(self.models) <= 1:
            print("Not available when there is only one model group (look at plot_prk_curves instead)")
            return
        fig, axes = self._get_subplots(subplot_width=6, n_cols=1, sharey=True)
        for _, mg in enumerate(self.models):
            for j, train_end_time in enumerate(self.models[mg]):
                mode_analyzer = self.models[mg][train_end_time]
                mode_analyzer.plot_precision_threshold_curve(
                    ax=axes[j]
                )
                axes[j].set_title(f'{train_end_time}')
        fig.suptitle("Precision Threshold Curve")
        fig.tight_layout()

    def plot_feature_importance(self, n_top_features=20):
        """ plot all feature importance  """
        self._make_plot_grid(plot_type='plot_feature_importance', subplot_width=7, n_top_features=n_top_features)
        
            
    def plot_feature_group_importance(self, n_top_groups=20):
        """ plot all feature group importance  """
        self._make_plot_grid(plot_type='plot_feature_group_importance', subplot_width=7, n_top_groups=n_top_groups)


    def create_scatter_disparity_performance(self, metric, parameter, aeq_parameter, attr_col, attribute_values, 
                                         performance_col='stochastic_value', bias_metric='tpr', tiebreaker='worst', flip_disparity=False, 
                                         mitigated_tags=[], mitigated_bdfs=[], mitigated_performances=[], ylim=None):
        """
            Create scatterplot of one bias metric (e.g. tpr disparity) vs an evaluation metric (e.g. precision at some threshold) for a particular list of attribute value (must be from the same attribute)
            A simplified version of the scatterplot function here: https://github.com/dssg/fairness_tutorial/blob/master/notebooks/bias_reduction_with_outputs.ipynb 
        """
        # TODO add legend for model identification?
        evals_df_list = {}
        aequitas_df_list = {}
        fig, axes = self._get_subplots(subplot_width=6, n_rows=len(attribute_values))
        
        for k, attribute_value in enumerate(attribute_values):
            for i, mg in enumerate(self.models):
                for j, train_end_time in enumerate(self.models[mg]):
                    mode_analyzer = self.models[mg][train_end_time]
                    if i == 0:
                        evals_df_list[train_end_time] = []
                        aequitas_df_list[train_end_time] = []
                    evals_df_list[train_end_time].append(mode_analyzer.get_evaluations(metrics={metric: [parameter]})) # TODO only allow one eval metric here?
                    aequitas_df_list[train_end_time].append(mode_analyzer.get_aequitas())

            for j, train_end_time in enumerate(self.models[self.model_groups[0]]):

                evals_df = pd.concat(evals_df_list[train_end_time])
                aequitas_df = pd.concat(aequitas_df_list[train_end_time])
                # filter aequitas by eval metric
                aequitas_df = aequitas_df.loc[(aequitas_df['parameter']==aeq_parameter) * (aequitas_df['tie_breaker']==tiebreaker)]
                disparity_df = aequitas_df.loc[(aequitas_df['attribute_name']==attr_col) & (aequitas_df['attribute_value']==attribute_value)].copy()
                disparity_metric = bias_metric + '_disparity'
                scatter_schema = ['model_id', performance_col, 'attribute_name', 'attribute_value', bias_metric, disparity_metric, 'model_tag']
                if flip_disparity:
                    disparity_df[disparity_metric]= disparity_df.apply(lambda x: 1/x[disparity_metric] , axis=1)
                scatter = pd.merge(evals_df, disparity_df, how='left', on=['model_id'], sort=True, copy=True)
                scatter = scatter[['model_id', performance_col, 'attribute_name', 'attribute_value', bias_metric, disparity_metric]].copy()
                scatter['model_tag'] = 'Other Models'
                scatter.sort_values('stochastic_value', ascending = False, inplace=True, ignore_index=True)
                scatter_final = pd.DataFrame()

                ax = axes[k,j]
                ax.scatter(
                    x='stochastic_value', y=disparity_metric,
                    data=scatter

                )
                if not scatter_final.empty:
                    ax.scatter(
                        x='stochastic_value', y=disparity_metric,
                        data=scatter_final
                    )
                
                if j==0:
                    axes[k, j].set_ylabel(f'{attr_col}:{attribute_value}')
                else:
                    axes[k, j].set_ylabel('')
                if k == 0:
                    axes[k, j].set_title(f'{train_end_time}')
                else:
                    axes[k, j].set_title('')
                axes[k, j].set_xlabel(f'{metric}{parameter}')

                if ylim:
                    plt.ylim(0, 10)
        flip_placeholder = 'Flipped' if flip_disparity else ''
        fig.suptitle(f'{flip_placeholder} {disparity_metric} vs. {metric}{parameter} for {attr_col}')


    def calculate_crosstabs_pos_vs_neg(self, project_path, thresholds, table_name='crosstabs', **kwargs):
        """ Generate crosstabs for the predicted positives (top-k) vs the rest

        args:
            project_path (str): Path where the experiment artifacts (models and matrices) are stored
            thresholds (Dict{str: Union[float, int}]): A dictionary that maps threhold type to the threshold
                                                    The threshold type can be one of the rank columns in the test_results.predictions_table
            table_name (str, optional): Table name to use in the db's `test_results` schema. Defaults to crosstabs.
                                        If the table exists, results are appended
            **kwargs: This method can take other arguments sent to ModelAnalyzer.crosstabs_pos_vs_ng function
        """

        for i, mg in enumerate(self.models):
            for j, train_end_time in enumerate(self.models[mg]):
                model_analyzer = self.models[mg][train_end_time]

                df = model_analyzer.crosstabs_pos_vs_neg(
                    project_path=project_path,
                    thresholds=thresholds,
                    table_name=table_name,
                    **kwargs
                )


    def display_crosstab_pos_vs_neg(
        self, 
        threshold_type,
        threshold, 
        table_name='crosstabs', 
        project_path=None,
        display_n_features=40,
        filter_features=None,
        support_threshold=0.1,
        show_plots=True,
        return_dfs=True):

        """ display crosstabs for one threshold for all the models in the model groups
            
        Args:
            threshold_type (str): Type of rank threshold to use in splitting predicted positives from negatives. 
                                Has to be one of the rank columns in the test_results.predictions_table

            threshold (Union[int, float]): The rank threshold of the specified type. If the threshold type is an absolute, integer. If percentage, should be a float between 0 and 1
                        
            table_name (str, optional): Table name to fetch from/write to in the db's `test_results` schema. Defaults to crosstabs.
            
            project_path (str, optional): Path where the experiment artifacts are stored. Optional if the crosstabs are already calculated,
                required if the crosstab need to be calculated
            
            display_n_features (int, optional): Number of features to return. defaults to 40 (sorted by mean ratio desc). This is ignored if `filter_features` is specified

            filter_features (List[str], optional): The list of features that we are interested in. If not specified, `display_n_features` features are returned

            support_threshold (float, optional): The threshold of pct support for the feature (instances with non-zero values) among predicted positives 
        """

        dfs = dict()

        for i, mg in enumerate(self.models):
            for j, train_end_time in enumerate(self.models[mg]):
                model_analyzer = self.models[mg][train_end_time]

                idx = i * len(self.models[self.model_groups[0]]) + j 


                if len(self.models) == 1 and len(self.models[mg]) == 1:
                    axes = [axes] 
                try:
                    df = model_analyzer.display_crosstabs_pos_vs_neg(
                        threshold_type=threshold_type,
                        threshold=threshold,
                        display_n_features=display_n_features,
                        filter_features=filter_features,
                        support_threshold=support_threshold,
                        table_name=table_name,
                        # ax=(axes if show_plots else None),
                        show_plot=False, #TODO -- remove this parameter from 
                        return_df=True
                    )

                    dfs[model_analyzer.model_id] = df
                except ValueError as e:
                    logging.error('Please run calculate_crosstabs_pos_vs_neg function to calculate crosstabs first for all models!')
                    raise e
                
                print(f'\nModel Group: {mg}, Validation date: {train_end_time}'.center(30, ' '))
                display(df)

        if return_dfs:
            return dfs
        
    def _pairwise_feature_importance_comparison_single_split(self, train_end_time, n_top_features, model_group_ids=None,plot=True):
        """ For a given train_end_time, compares the top n features (highest absolute importance) of two models 
            
            Args:
                train_end_time (str): The prediction date we care about in YYYY-MM-DD format
                n_top_features (int): Number of features to consider for the comparispn
                model_group_ids (int, optional): Model group ids to consider, if not provided, all model groups included in the report are used
                plot (bool, optional): Whether to plot the results. Defaults to True.
        """
        
        feature_lists = dict()
        
        if model_group_ids is not None:
            
            for mg in model_group_ids:
                feature_lists[mg] = self.models[mg][train_end_time].get_feature_importances(n_top_features=n_top_features) 
                if feature_lists[mg].empty:
                    logging.warning('No feature importance values were found for model group {mg}. Excluding from comparison')
                    feature_lists.pop(mg)
        # By default all feature importance values are considered
        else:
            model_group_ids = self.model_groups
            for mg, m in self.models.items():
                feature_lists[mg] = m[train_end_time].get_feature_importances(n_top_features=n_top_features) 
                if feature_lists[mg].empty:
                    logging.warning('No feature importance values were found for model group {mg}. Excluding from comparison')
                    feature_lists.pop(mg)
            
        pairs = list(itertools.combinations(feature_lists.keys(), 2))
        
        logging.info(f'Performing {len(pairs)} comparisons')
        
        metrics = ['jaccard', 'overlap', 'rank_corr']
        results = dict()
        
        for m in metrics:
            results[m] = pd.DataFrame(index=sorted(model_group_ids), columns=sorted(model_group_ids))
            # filling the diagonal with 1
            results[m].values[[np.arange(results[m].shape[0])]*2] = 1
            
        for model_group_pair in pairs:
            model_group_pair = sorted(model_group_pair)
            logging.info(f'Comparing {model_group_pair[0]} and {model_group_pair[1]}')

            df1 = feature_lists[model_group_pair[0]]
            df2 = feature_lists[model_group_pair[1]]
            
            f1 = set(df1.feature)
            f2 = set(df2.feature)
            
            if (len(f1) == 0 or len(f2)) == 0:
                logging.error('No feature importance available for the models!') 
                continue

            inter = f1.intersection(f2)
            un = f1.union(f2)    
            results['jaccard'].loc[model_group_pair[1], model_group_pair[0]] = len(inter)/len(un)

            # If the list sizes are not equal, using the smallest list size to calculate simple overlap
            results['overlap'].loc[model_group_pair[1], model_group_pair[0]] = len(inter)/ min(len(f1), len(f2))

            # calculating rank correlation
            df1.sort_values('feature_importance', ascending=False, inplace=True)
            df2.sort_values('feature_importance', ascending=False, inplace=True)

            # only returning the corr coefficient, not the p-value
            results['rank_corr'].loc[model_group_pair[0], model_group_pair[1]] = spearmanr(df1.feature.iloc[:], df2.feature.iloc[:])[0]

    
        if plot:
            fig, axes = plt.subplots(1, len(metrics), figsize=(10, 3))            
            
            for i, m in enumerate(metrics):
                sns.heatmap(
                    data=results[m].fillna(0),
                    cmap='Greens',
                    vmin=0,
                    vmax=1,
                    annot=True,
                    linewidth=0.1,
                    ax=axes[i]
                )

                axes[i].set_title(m)

            fig.suptitle(train_end_time)
            fig.tight_layout()
        
        return results
    

    def _pairwise_list_comparison_single_fold(self, threshold_type, threshold, train_end_time, matrix_uuid=None, plot=True):
        """For a given train_end_time, compares the lists generated by the analyzed model groups

        Args:
            threshold_type (str): Type of the ranking to use. Has to be one of the four ranking types used in triage
                    - rank_pct_no_ties 
                    - rank_pct_with_ties
                    - rank_abs_no_ties
                    - rank_abs_with_ties
            threshold (Union[float, int]): The threshold rank for creating the list. Int for 'rank_abs_*' and Float for 'rank_pct_*'
            train_end_time (str): The prediction date we care about in YYYY-MM-DD format
        """

        lists = dict()

        for mg, m in self.models.items():
            lists[mg] = m[train_end_time].get_top_k(threshold_type, threshold, matrix_uuid)

        
        pairs = list(itertools.combinations(lists.keys(), 2))

        logging.info(f'Performing {len(pairs)} comparisons')

        metrics = ['jaccard', 'overlap', 'rank_corr']
        results = dict()

        # Initializing three data frames to hold pairwise metrics
        for m in metrics:
            results[m] = pd.DataFrame(index=sorted(self.model_groups), columns=sorted(self.model_groups))
            results[m].values[[np.arange(results[m].shape[0])]*2] = 1

        for model_group_pair in pairs:
            logging.info(f'Comparing {model_group_pair[0]} and {model_group_pair[1]}')

            model_group_pair = sorted(model_group_pair)
            
            df1 = lists[model_group_pair[0]]
            df2 = lists[model_group_pair[1]]

            # calculating jaccard similarity and overlap
            entities_1 = set(df1.entity_id)
            entities_2 = set(df2.entity_id)

            if (len(entities_1) == 0 or len(entities_2)) == 0:
                logging.error('No prediction saved for the models!') 

            inter = entities_1.intersection(entities_2)
            un = entities_1.union(entities_2)    
            results['jaccard'].loc[model_group_pair[1], model_group_pair[0]] = len(inter)/len(un)

            # If the list sizes are not equal, using the smallest list size to calculate simple overlap
            results['overlap'].loc[model_group_pair[1], model_group_pair[0]] = len(inter)/ min(len(entities_1), len(entities_2))

            # calculating rank correlation
            df1.sort_values('score', ascending=False, inplace=True)
            df2.sort_values('score', ascending=False, inplace=True)

            # only returning the corr coefficient, not the p-value
            if len(df1) == len(df2):
                results['rank_corr'].loc[model_group_pair[0], model_group_pair[1]] = spearmanr(df1.entity_id.iloc[:], df2.entity_id.iloc[:])[0]
            else:
                logging.warning(f'Not calculating rank correlation. List sizes are not equal ({len(df1)}, {len(df2)})')

        if plot:
            fig, axes = plt.subplots(1, len(metrics), figsize=(10, 3))            
            
            for i, m in enumerate(metrics):
                sns.heatmap(
                    data=results[m].fillna(0),
                    cmap='Greens',
                    vmin=0,
                    vmax=1,
                    annot=True,
                    linewidth=0.1,
                    ax=axes[i]
                )

                axes[i].set_title(m)

            fig.suptitle(train_end_time)
            fig.tight_layout()
        return results

    def _get_individual_model_ids(self):
        """Get individual model ids associated with the given model groups
        """
        model_groups = ", ".join([str(x) for x in self.model_groups])

        q = f"""
            select
                model_id 
            from triage_metadata.models
            where model_group_id in({model_groups})
        """
        df_model_ids = pd.read_sql(q, self.engine)

        return df_model_ids.model_id.to_list()


    def execute_error_analysis(self, model_ids=None):
        """Generates the error analysis of a model
            args:
                model_ids (int, optional): A list of model_ids we are interested in analyzing. If not provided, all models are considered.
        """
        if model_ids is None:
            model_ids = self.model_ids

        for model_id in model_ids:
            logging.info(f"generating error analysis for model id {model_id}")
            results = generate_error_analysis(model_id, self.engine, self.project_path)
            output_all_analysis(results)
    
    
    def pairwise_top_k_list_comparison(self, threshold_type, threshold, train_end_times=None, matrix_uuid=None, plot=True):
        """
            Compare the top-k lists for the given train_end_times for all model groups considered (pairwise)
            
            Args:
                threshold_type (str): Type of the ranking to use. Has to be one of the four ranking types used in triage
                        - rank_pct_no_ties 
                        - rank_pct_with_ties
                        - rank_abs_no_ties
                        - rank_abs_with_ties
                threshold (Union[float, int]): The threshold rank for creating the list. Int for 'rank_abs_*' and Float for 'rank_pct_*'
                train_end_times (Optional, List[str]): The prediction date we care about in YYYY-MM-DD format
        """

        # If no train_end_times are provided, we consider all the train_end_times
        # NOTE -- Assuming that the all model groups have the same train_end_times
        if train_end_times is None:
            train_end_times = self.models[self.model_groups[0]].keys()

        for train_end_time in train_end_times:
            self._pairwise_list_comparison_single_fold(
                threshold=threshold,
                threshold_type=threshold_type,
                train_end_time=train_end_time,
                matrix_uuid=matrix_uuid,
                plot=plot
            )
            
    def pairwise_feature_importance_comparison(self, n_top_features, model_groups=None, train_end_times=None, plot=True):
        """
            Compare the top-k lists for the given train_end_times for all model groups considered (pairwise)
            
            Args:
                threshold_type (str): Type of the ranking to use. Has to be one of the four ranking types used in triage
                        - rank_pct_no_ties 
                        - rank_pct_with_ties
                        - rank_abs_no_ties
                        - rank_abs_with_ties
                threshold (Union[float, int]): The threshold rank for creating the list. Int for 'rank_abs_*' and Float for 'rank_pct_*'
                train_end_times (Optional, List[str]): The prediction date we care about in YYYY-MM-DD format
        """

        # If no train_end_times are provided, we consider all the train_end_times
        # NOTE -- Assuming that the all model groups have the same train_end_times
        if train_end_times is None:
            train_end_times = self.models[self.model_groups[0]].keys()
            

        for train_end_time in train_end_times:
            self._pairwise_feature_importance_comparison_single_split(
                train_end_time=train_end_time,
                n_top_features=n_top_features,
                model_group_ids=model_groups,
                plot=plot
            )



class ModelComparator:
    def __init__(self, model_ids, engine):        
        # Initializing ModelAnalyzer for each model_id
        self.model_ids = sorted(model_ids)
        self.models = dict()
        for model_id in model_ids:
            self.models[model_id] = ModelAnalyzer(model_id, engine)
            
        self.engine = engine
        self.models_summary()
            
    def models_summary(self):
        """ Print the model summary for all the models we are comparing"""
        for model_id, ma in self.models.items():
            print(f"{model_id} -- {ma.get_model_description()} ")
            
    def compare_ranking(self, plot=True, k_values=None):
        """Compare rankings of the two models
        
        We calculate: spearman's corr, kendall tau, and at varying thresholds overlap and jaccard similarity
        
        """
        
        #NOTE: hardcoding k values for now
        if k_values is None:
            k_values = np.arange(100, 501, 100)
        
        ranks = dict()
        for model_id, ma in self.models.items():
            ranks[model_id] = ma.get_predictions()['rank_abs_no_ties']
            
        
        pairs = list(itertools.combinations(self.model_ids, 2))
        
        # rank correlation
        results = dict()
        results['spearman'] = pd.DataFrame(np.full((len(self.model_ids), len(self.model_ids)), np.nan), index=self.model_ids, columns=self.model_ids)
        results['kendall'] = pd.DataFrame(np.full((len(self.model_ids), len(self.model_ids)), np.nan), index=self.model_ids, columns=self.model_ids)
        
        for k in k_values:
            results[f'overlap@{k}'] = pd.DataFrame(np.full((len(self.model_ids), len(self.model_ids)), np.nan), index=self.model_ids, columns=self.model_ids)
            results[f'jaccard@{k}'] = pd.DataFrame(np.full((len(self.model_ids), len(self.model_ids)), np.nan), index=self.model_ids, columns=self.model_ids)
        
        for pair in pairs:
            pair = sorted(pair)
                        
            # only returning the corr coefficient, not the p-value
            results['spearman'].loc[pair[1], pair[0]] = spearmanr(ranks[pair[0]], ranks[pair[1]])[0]
            results['kendall'] .loc[pair[1], pair[0]] = kendalltau(ranks[pair[0]], ranks[pair[1]])[0]
            
            for k in k_values:
                # calculating jaccard similarity and overlap
                entities_1 = set(ranks[pair[0]].reset_index().entity_id.iloc[:k])
                entities_2 = set(ranks[pair[1]].reset_index().entity_id.iloc[:k])

                inter = entities_1.intersection(entities_2)
                un = entities_1.union(entities_2)    
                
                results[f'jaccard@{k}'].loc[pair[1], pair[0]] = len(inter)/len(un)

                # If the list sizes are not equal, using the smallest list size to calculate simple overlap
                results[f'overlap@{k}'].loc[pair[1], pair[0]] = len(inter)/ min(len(entities_1), len(entities_2))
        
        if plot:
            
            fig, axes = plt.subplots(1, 2, figsize=(2*3 + 0.3, 3))
            
            plot_pairwise_comparison_heatmap(df=results['spearman'], metric_name='spearman', ax=axes[0])
            plot_pairwise_comparison_heatmap(df=results['kendall'], metric_name='kendall', ax=axes[1])
            
            # 2 x k grid -- one row for jaccard and one for overlap
            fig, axes = plt.subplots(2, len(k_values), figsize=(len(k_values)*3 + 0.3, 3*2))
            for i, k in enumerate(k_values):
                plot_pairwise_comparison_heatmap(df=results[f'overlap@{k}'], metric_name=f'overlap@{k}', ax=axes[0, i])
                plot_pairwise_comparison_heatmap(df=results[f'jaccard@{k}'], metric_name=f'jaccard@{k}', ax=axes[1, i])
            
            plt.tight_layout()
            plt.show()
            
        return results
                
    
    def compare_metrics(self, metrics=None, matrix_uuid=None, subset_hash=None, display=True, **kwargs):
        """
            Compare the metrics for the given train_end_times for all model groups considered (pairwise)
            
            Args:
                metrics Dict[str:List]): Optional. The metrics and parameters for evaluations. 
                                    A dictionary of type {metric:[thresholds]}, e.g., {'precision@': ['100_abs'], 'recall@': ['100_abs']}
                                    If not specified, all the evaluations will be returned
                threshold (Union[float, int]): The threshold rank for creating the list. Int for 'rank_abs_*' and Float for 'rank_pct_*'
                matrix_uuid (str): The matrix uuid to use in the comparison
        """
        evals = list()
        columns = ['metric', 'parameter', 'metric_value']
        
        evals = None
        for model_id, ma in self.models.items():        
            df = ma.get_evaluations(metrics=metrics, matrix_uuid=matrix_uuid, subset_hash=subset_hash, plot_prk=False)[columns]
            df.rename(columns={'metric_value': model_id}, inplace=True)
            if evals is None:
                evals = df
            else:
                evals = evals.merge(df, how='inner', on=['metric', 'parameter'])
            
        # evals.set_index(['metric', 'threshold', 'threshold_type'], inplace=True)
        # evals['diff'] = evals[self.model_ids[0]] - evals[self.model_ids[1]]
        evals_pct = evals[evals.parameter.str.contains('pct')]
        evals_abs = evals[evals.parameter.str.contains('abs')]
        
        evals_pct.set_index(['metric', 'parameter'], inplace=True)
        evals_abs.set_index(['metric', 'parameter'], inplace=True)
        
        if display:
            return evals_pct.style.highlight_max(axis=1, color='darkgreen', props='color: white; font-weight: bold; background-color: #4CAF50'), \
            evals_abs.style.highlight_max(axis=1, color='darkgreen', props='color: white; font-weight: bold; background-color: #4CAF50')
        
        return evals_pct, evals_abs
        
        
    def compare_topk(self, threshold_type, threshold, matrix_uuid=None, plot=True, **kwargs):
        """
            Compare the top-k lists for the given train_end_times for all model groups considered (pairwise)
            We compare jaccard, overlap, and rank_correlation
            
            Args:
                threshold_type (str): Type of the ranking to use. Has to be one of the four ranking types used in triage
                        - rank_pct_no_ties 
                        - rank_pct_with_ties
                        - rank_abs_no_ties
                        - rank_abs_with_ties
                threshold (Union[float, int]): The threshold rank for creating the list. Int for 'rank_abs_*' and Float for 'rank_pct_*'
                matrix_uuid (str): The matrix uuid to use in the comparison
                metrics (List[str], optional): The list of metrics to use in the comparison. Defaults to ['jaccard', 'overlap', 'rank_corr']
        """
        
        topk = dict()
        for model_id, ma in self.models.items():
            topk[model_id] = ma.get_top_k(threshold_type, threshold, matrix_uuid)
            
            if topk[model_id].empty:
                logging.warning(f'No prediction saved for the model {model_id}. Excluding from comparison')
                topk.pop(model_id)
                continue
            
        if topk == {}:
            logging.error('No prediction saved for the models. Aborting!') 
            return
        
        pairs = list(itertools.combinations(topk.keys(), 2))
        
        logging.info(f'Performing {len(pairs)} comparisons')
        
        # These are hardcoded for now
        # metrics=['jaccard', 'overlap', 'rank_corr']
        metrics=['jaccard', 'overlap']
        
        results = dict()
        # Initializing three data frames to hold pairwise metrics
        
        for m in metrics:
            # Initializing a null dataframe
            results[m] = pd.DataFrame(np.full((len(self.model_ids), len(self.model_ids)), np.nan), index=self.model_ids, columns=self.model_ids)
            
            # Initialize the diagonal with 1 (this is a triagular matrix)
            np.fill_diagonal(results[m].values, 1)
            
        for model_pair in pairs:
            model_pair = sorted(model_pair)
            
            entities_1 = set(topk[model_pair[0]].entity_id)
            entities_2 = set(topk[model_pair[1]].entity_id)
            
            n_intersect = len(entities_1.intersection(entities_2))
            n_union = len(entities_1.union(entities_2))
            
            results['jaccard'].loc[model_pair[1], model_pair[0]] = n_intersect/n_union
            
            # If the list sizes are not equal, using the smallest list size to calculate simple overlap
            results['overlap'].loc[model_pair[1], model_pair[0]] = n_intersect/min(len(entities_1), len(entities_2))
            
            # calculating rank correlation
            # TODO: FIX ME!
            # results['rank_corr'].loc[model_group_pair[0], model_group_pair[1]] = spearmanr(
            #     topk[model_group_pair[0]].score.iloc[:], 
            #     topk[model_group_pair[1]].score.iloc[:]
            # )[0]
        
        if plot:
            fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics)*3 + 0.3, 3))   
            
            for i, m in enumerate(metrics):
                plot_pairwise_comparison_heatmap(df=results[m], metric_name=m, ax=axes[i])
            
            fig.suptitle(f'{threshold_type}, top {threshold}')
            fig.tight_layout()    
        
        return results
        
    
    def compare_feature_importance(self, n_top_features=20, plot=True):
        """ Compare the feature importance of the two models"""
        
        feature_importances = dict()
        
        for model_id, ma in self.models.items():
            feature_importances[model_id] = ma.get_feature_importances(n_top_features=n_top_features)
            
            if feature_importances[model_id].empty:
                logging.warning(f'No feature importance values were found for model {model_id}. Excluding from comparison')
                feature_importances.pop(model_id)
                continue
        if feature_importances == {}:
            logging.error('No feature importance values were found for the models. Aborting!') 
            return
        
        pairs = list(itertools.combinations(feature_importances.keys(), 2))
        
        metrics = ['jaccard', 'overlap', 'rank_corr']
        results = dict()
        
        for m in metrics:
            results[m] = pd.DataFrame(np.full((len(self.model_ids), len(self.model_ids)), np.nan), index=self.model_ids, columns=self.model_ids)
            
            # Initialize the diagonal with 1 (this is a triagular matrix)
            np.fill_diagonal(results[m].values, 1)
            
        for model_pair in pairs:
            model_pair = sorted(model_pair)
            
            f1 = set(feature_importances[model_pair[0]].feature)
            f2 = set(feature_importances[model_pair[1]].feature)
            
            if len(f1) != len(f2):
                logging.warning(f'Feature counts are not equal, skipping rank corr - {model_pair[0]}: {len(f1)}, {model_pair[1]}: {len(f2)}') 
                results['rank_corr'].loc[model_pair[0], model_pair[1]] = np.nan
            else:
                # only returning the corr coefficient, not the p-value 
                results['rank_corr'].loc[model_pair[0], model_pair[1]] = spearmanr(feature_importances[model_pair[0]].feature.iloc[:], feature_importances[model_pair[1]].feature.iloc[:])[0]
                
            
            inter = f1.intersection(f2)
            un = f1.union(f2)    
            results['jaccard'].loc[model_pair[1], model_pair[0]] = len(inter)/len(un)

            # If the list sizes are not equal, using the smallest list size to calculate simple overlap
            results['overlap'].loc[model_pair[1], model_pair[0]] = len(inter)/ min(len(f1), len(f2))

            # calculating rank correlation
            feature_importances[model_pair[0]].sort_values('importance', ascending=False, inplace=True)
            feature_importances[model_pair[1]].sort_values('importance', ascending=False, inplace=True)
            
        if plot:
            fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics)*3 + 0.3, 3))   
            
            for i, m in enumerate(metrics):
                plot_pairwise_comparison_heatmap(df=results[m], metric_name=m, ax=axes[i])
            
            fig.tight_layout()  
            
        return results
    
    
    def compare_score_distribution(self):
        """ Comparing the score distributions of the two models"""
        pass
    
    def compare_equity(self):
        pass 

class ModelGroupComparator:
    def __init__(self, model_group_ids):
        pass