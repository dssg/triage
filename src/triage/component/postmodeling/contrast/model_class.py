import ohio.ext.pandas
import pandas as pd
import logging
import seaborn as sns
import matplotlib.table as tab
import matplotlib.pyplot as plt
from tabulate import tabulate

from descriptors import cachedproperty
from sqlalchemy import create_engine
from sklearn.calibration import calibration_curve

from triage.component.catwalk.storage import ProjectStorage
from triage.component.postmodeling.error_analysis import generate_error_analysis


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
                           m.train_end_time::date,
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
                where_clause += f""" metric={metric} AND parameter in ('{"','".join(parameters)}')"""

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
                stochastic_value,
                num_labeled_above_threshold,               
                num_positive_labels
            from test_results.evaluations
            {where_clause}
        """

        evaluations = pd.read_sql(q, self.engine)

        return evaluations

    def get_eval_labeling_window(self):
        # TODO right now this assumes all evaluations for this model are for the same test_label_timespan (won't be true when we can specify matrix_uuid)
        q = f"""
            select 
                distinct labeling_window
            from triage_metadata.models join test_results.evaluations e using(model_id)
            join triage_metadata.matrices using (matrix_uuid)
            where model_id = {self.model_id}
            """
        labeling_windows = pd.read_sql(q, self.engine)
        return labeling_windows.iloc[0]['labeling_window']


    def get_feature_importances(self):
        features = pd.read_sql(
           f'''
           select
                feature,
                feature_importance,
                rank_abs
           FROM train_results.feature_importances
           WHERE model_id = {self.model_id}
           ''', con=self.engine)
        return features
    
    def get_feature_group_importances(self):
        """
        Get the top most important feature groups as identified by the maximum importance of any feature in the group

        """
        # TODO this assumes any experiment linked to this model has the same feature aggregations, is this valid?
        q = f"""
            select distinct experiment_hash from triage_metadata.models m 
            left join triage_metadata.experiment_models em on m.model_hash=em.model_hash 
            where model_group_id={self.model_group_id}
            """
        experiment_hashes = pd.read_sql(q, self.engine)
        experiment_hash = experiment_hashes['experiment_hash'].iloc[0]

        # get feature group names
        q = f"""
            select 
                config->'feature_aggregations' as feature_groups
            from triage_metadata.experiments where experiment_hash = '{experiment_hash}'
        """

        feature_groups = [i['prefix'] for i in pd.read_sql(q, self.engine)['feature_groups'].iloc[0]]
        feature_groups
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
            feature_group,
                max(abs(feature_importance)) as importance_aggregate
            FROM raw_importances
            GROUP BY feature_group, model_id
        """, con=self.engine)
        return feature_group_importance

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
        positive_support_pct = lambda pos, neg: round((pos > 0).sum(axis=0).astype(float) / len(pos), 3)
        negative_support_pct = lambda pos, neg: round((neg > 0).sum(axis=0).astype(float) / len(neg), 3)

        crosstab_functions = [
            ("mean_predicted_positive", positive_mean),
            ("mean_predicted_negative", negative_mean),
            ("std_predicted_positive", positive_std),
            ("std_predicted_negative", negative_std),
            ("mean_ratio_predicted_positive_to_predicted_negative", ratio_positive_negative),
            ("support_predicted_positive", positive_support),
            ("support_predicted_negative", negative_support),
            ("support_pct_predicted_positive", positive_support_pct),
            ("support_pct_predicted_negative", negative_support_pct)
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
        

    def display_crosstabs_pos_vs_neg(
            self, 
            threshold_type, threshold, table_name='crosstabs',
            display_n_features=40,
            filter_features=None,
            support_threshold=0.1,
            return_df=True,
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
                'mean_ratio_predicted_positive_to_predicted_negative',
                'mean_predicted_positive',
                'mean_predicted_negative',
                'support_pct_predicted_positive',
                'support_pct_predicted_negative'
            )
            and model_id = {self.model_id}            
        """

        ct = pd.read_sql(q, self.engine)

        if ct.empty:
            logging.error(
                f'''Crosstabs not found for model {self.model_id} in table test_results.{table_name}. 
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
                'mean_ratio_predicted_positive_to_predicted_negative': 'ratio',
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
        msk = pivot_table['(+)supp'] > support_threshold
        # Features with highest postive : negative ratio
        df1 = pivot_table[msk].sort_values(
            'ratio', 
             ascending = False
        ).head(display_n_features)


        msk = pivot_table['(-)supp'] > support_threshold
        # Features with the highest negative : positive ratio
        df2 = pivot_table[msk].sort_values(
            ['ratio'], 
             ascending = True
        ).head(display_n_features)

        df = pd.concat([df1, df2])

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
        error_analysis_results = generate_error_analysis(model_id, connection)
        #TODO do we want to 
        return error_analysis_results


    def plot_score_distribution(self, ax, nbins=10, top_k=None, matrix_uuid=None):
        """
        Plot the distribution of predicted scores for all entities across model groups and train_end_time
        Optionally, show only the top k predicted scores.

        Args:
            ax: matplotlib Axes object to plot on
            nbins (optional, int): the number of bins to apply to the score histogram
            top_k (optional, int): if not None, displays only the top_k scores.
            matrix_uuid (optional): get model scores for a particular matrix
        """

        predictions = self.get_predictions(matrix_uuid)
        # keep only top_k if specified
        if top_k:
            predictions = predictions.loc[predictions['rank_abs_no_ties'] <= top_k]
        ax.hist(predictions.score,
                 bins=nbins,
                 alpha=0.5,
                 color='blue')
        ax.axvline(predictions.score.mean(),
                    color='black',
                    linestyle='dashed')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Score')
        ax.set_title('Score Distribution')
        return ax
    
    def plot_score_label_distribution(self, ax,
                                      nbins=10,
                                      top_k=None,
                                      matrix_uuid=None, 
                                      label_names = ('Label = 0', 'Label = 1')):
        """
        Plot the distribution of predicted scores for all entities across model groups and train_end_time
        Optionally, show only the top k predicted scores.
        NOTE: this function only handles 0/1 labels, ignores null labels

        Args:
            ax: matplotlib Axes object to plot on
            nbins (int): the number of bins to define score distribution
            top_k (optional, int): if not None, displays only the top_k scores.
            matrix_uuid (optional): specify a matrix to get predictions for
            label_names (tuple[String]): specify the two label names to display on the plot

        Return:
            ax: the modified Axes object
        """

        df_predictions = self.get_predictions(matrix_uuid)
        # keep only top_k if specified
        if top_k:
            df_predictions = df_predictions.loc[df_predictions['rank_abs_no_ties'] <= top_k]
        # TODO for now assumes all predictions are for the same test_label_timespan
        test_label_timespan = df_predictions.iloc[0]['test_label_timespan']
        df__0 = df_predictions[df_predictions.label_value == 0]
        df__1 = df_predictions[df_predictions.label_value == 1]

        ax.hist(df__0.score,
                 bins=nbins,
                 alpha=0.5,
                 color='skyblue',
                 label=label_names[0])
        ax.hist(list(df__1.score),
                 bins=nbins,
                 alpha=0.5,
                 color='orange',
                 label=label_names[1])
        ax.axvline(df__0.score.mean(),
                    color='skyblue',
                    linestyle='dashed')
        ax.axvline(df__1.score.mean(),
                    color='orange',
                    linestyle='dashed')
        ax.legend(bbox_to_anchor=(0., 1.005, 1., .102),
                   loc=8,
                   ncol=2,
                   borderaxespad=0.)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Score')
        ax.set_title('Score Distribution across Labels')
        return ax, test_label_timespan

    def plot_precision_param_curve(self, ax, matrix_uuid=None):
        eval_df = self.get_evaluations(matrix_uuid=matrix_uuid)

        eval_df['perc_points'] = [x.split('_')[0] for x in eval_df['parameter'].tolist()]
        eval_df['perc_points'] = pd.to_numeric(eval_df['perc_points'])

        msk_prec = eval_df['metric']=='precision@'
        msk_pct = eval_df['parameter'].str.contains('pct')

        # plot precision
        sns.lineplot(
            x='perc_points',
            y='stochastic_value', 
            data=eval_df[msk_pct & msk_prec], 
            label=self.model_group_id,
            ax=ax, 
            estimator='mean', ci='sd'
        )
        return ax

    def plot_precision_recall_curve(self, ax, matrix_uuid=None):
        """
        Plots precision-recall curves at each train_end_time for all model groups
        
        Args:
            ax: matplotlib Axes object to plot on
            matrix_uuid (optional): specify a matrix to get predictions for
            
        Return:
            ax: the modified Axes object
        """

        eval_df = self.get_evaluations(matrix_uuid=matrix_uuid)
        # TODO for now assumes all evaluations are for the same test_label_timespan
        labeling_window = self.get_eval_labeling_window()
        
        eval_df['perc_points'] = [x.split('_')[0] for x in eval_df['parameter'].tolist()]
        eval_df['perc_points'] = pd.to_numeric(eval_df['perc_points'])
        
        msk_prec = eval_df['metric']=='precision@'
        msk_recall = eval_df['metric']=='recall@'
        msk_pct = eval_df['parameter'].str.contains('pct')

        # plot precision
        sns.lineplot(
            x='perc_points',
            y='stochastic_value', 
            data=eval_df[msk_pct & msk_prec], 
            label='precision@k',
            ax=ax, 
            estimator='mean', ci='sd'
        )
        # plot recall
        sns.lineplot(
            x='perc_points', 
            y='stochastic_value', 
            data=eval_df[msk_pct & msk_recall], 
            label='recall@k', 
            ax=ax, 
            estimator='mean', 
            ci='sd'
        )
        ax.set_xlabel('List size percentage (k%)')
        ax.set_ylabel('Metric Value')
        ax.set_title('Precision-Recall Curve')
        return ax, labeling_window

    def plot_feature_importance(self, ax, n_top_features=20):
        """
        Plot the top most important individual features across model groups and train end times

        Args:
            ax: matplotlib Axes object to plot on
            n_top_features (int): the number of features to display
            
        Return:
            ax: the modified Axes object
        """

        feature_importance_scores = self.get_feature_importances()
        # keep only top n_top_features
        feature_importance_scores.sort_values(by=['rank_abs'], ascending=True, inplace=True)
        feature_importance_scores = feature_importance_scores.loc[feature_importance_scores['rank_abs'] <= n_top_features]
        # plot
        sns.barplot(
            data=feature_importance_scores,
            x='feature_importance', 
            y='feature',
            color='royalblue',
            ax=ax
        )
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance')
        return ax

    def plot_feature_group_importance(self, ax):
        """
        Plot the top most important feature groups as identified by the maximum importance of any feature in the group

        Args:
            ax: matplotlib Axes object to plot on

        Return: 
            ax: modified matplotlib Axes object
        """
        feature_group_importance = self.get_feature_group_importances()
        feature_group_importance.sort_values(by=['importance_aggregate'], ascending=False, inplace=True)
        sns.barplot(
            data=feature_group_importance,
            x='importance_aggregate', 
            y='feature_group',
            color='royalblue',
            ax=ax
        )
        ax.set_ylabel('Feature Group')
        ax.set_title('Feature Importance')
        return ax

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
        # TODO for now assumes all predictions are for the same test_label_timespan
        test_label_timespan = scores.iloc[0]['test_label_timespan']
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
        ax.set_title('Calibration curve')
        ax.set_xlim([min_score, max_score])
        ax.set_ylim([min_score, max_score])
        return ax, test_label_timespan

