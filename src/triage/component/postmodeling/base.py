import ohio.ext.pandas
import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.table as tab
import matplotlib.pyplot as plt
from io import StringIO
import altair as alt

from IPython.display import display
import itertools

from descriptors import cachedproperty
from sqlalchemy import create_engine
from sklearn.calibration import calibration_curve
from sklearn import metrics
from scipy.stats import spearmanr

from triage.component.catwalk.storage import ProjectStorage
from triage.component.postmodeling.error_analysis import generate_error_analysis, output_all_analysis
from triage.database_reflection import table_exists
from triage.component.catwalk.utils import sort_predictions_and_labels


ID_COLUMNS = ['entity_id', 'as_of_date']

class Model:
    def __init__(self, model_id, engine):
        self.model_id = model_id
        self.engine = engine

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
        return self.metadata['model_type'].split('.')[-1]

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

    def predictions(self, matrix_uuid=None, fetch_null_labels=True, subset_hash=None):
        where_clause = f"WHERE model_id = {self.model_id}"

        if matrix_uuid is not None:
            where_clause += f" AND matrix_uuid='{matrix_uuid}'"

        if not fetch_null_labels:
            where_clause += f" AND label_value IS NOT NULL"

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
            FROM test_results.predictions
            {where_clause}        
        """

        preds = pd.read_sql(query, self.engine)
        
        if preds.empty:
            logging.warning(f'No predictions were found in {predictions_table} for model_id {self.model_id}. Returning empty dataframe!')
            return preds 
        
        return preds.set_index(ID_COLUMNS)

    def evaluations(self, matrix_uuid=None, subset_hash=None):
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
                stochastic_value,
                num_labeled_above_threshold,               
                num_positive_labels
            from test_results.evaluations
            {where_clause}
        """

        evaluations = pd.read_sql(q, self.engine)

        return evaluations 
    
    def feature_list(self):
        q = f'''
            select 
            feature_list
            from triage_metadata.models inner join triage_metadata.model_groups using(model_group_id)
            where model_id = {self.model_id}
        '''
        
        return pd.read_sql(q, self.engine).at[0, 'feature_list']
    
    def feature_importances(self, n_top_features=100):
        features = pd.read_sql(
           f'''
           select
                feature,
                feature_importance,
                rank_abs
           FROM train_results.feature_importances
           WHERE model_id = {self.model_id}
           and rank_abs <= {n_top_features}
           and abs(feature_importance) > 0 
           order by rank_abs
           ''', con=self.engine)
        
        return features 
    
    def feature_group_importance(self):
        # NOTE: The way we are inferring the feature group is sensitive to the feature naming convention
        # The more stable way would be to fetch the prefixes from the experiment config
        q = f'''
            select 
            split_part(feature, '_entity_id_', 1) as feature_group,
            avg(feature_importance) as avg_importance,
            max(feature_importance) as max_importance,
            min(feature_importance) as min_importance,
            stddev(feature_importance) as stddev_importance 
            from train_results.feature_importances
            where model_id = {self.model_group_id}
            group by 1
            order by 3 desc
        ''' 
        
        return pd.read_sql(q, self.engine)
    
    def bias_metrics(self, tie_breaker='worst',subset_hash=None):
        
        if subset_hash is None:
            subset_hash = ''
            
        q = f'''
            select 
            case 
                when split_part(parameter, '_', 2) = 'pct' then split_part(parameter, '_', 1)::float * 100
                else split_part(parameter, '_', 1)::int  
            end as param_numeric,
            split_part(parameter, '_', 2) as param_type,
            *
            from test_results.aequitas a 
            where model_id = {self.model_id}
            and tie_breaker = '{tie_breaker}'
            and subset_hash ='{subset_hash}'
        '''
        
        return pd.read_sql(q, engine)
    

class ModelGroup:
    def __init__(self, model_group_id, engine):
        self.model_group_id = model_group_id
        self.engine = engine
        self.models = self._initialize_models()

        
    def _initialize_models(self):
        """ Create a list of model ids"""
        return None
        
    
        
class ModelAnalyzer:
    def __init__(self,engine, model_id, performance_metric='precision@', bias_metric='tpr_disparity', threshold='1_pct', include_ties=False):
        self.model = Model(model_id=model_id, engine=engine)
        self.engine=engine
        self.performance_metric=performance_metric
        self.bias_metric=bias_metric
        self.threshold=threshold
        self.include_ties=include_ties 
    
    def predicted_positives(self, matrix_uuid=None):
        """ Fetch the k intities with highest model score for a given model
        
            Args:
                matrix_uuid (optional, str): If a list should be generated out of a matrix that were not used to validate the model during the experiment
        """

        ties_suffix = 'with_ties' if self.include_ties else 'no_ties'
        rank_column = f"rank_{self.threshold.split('_')[1]}_{ties_suffix}"
        
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
            where model_id={self.model_id}
            and {rank_column} <= {self.threshold.split('_')[0]}
        """
        
        if matrix_uuid is not None: 
            q += f" and matrix_uuid='{matrix_uuid}'"

        return pd.read_sql(q, self.engine)
  
    def calculate_crosstabs(self, project_path, matrix_uuid=None, push_to_db=True, table_name='crosstabs', return_df=True, replace=True, predictions_table='test_results.predictions'):
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
            matrix_uuid = self.pred_matrix_uuid
            logging.debug(f'Matrix uuid set to: {matrix_uuid}')

        predictions = self.model.predictions(matrix_uuid=matrix_uuid)
        if predictions.empty:
            logging.error(f'No predictions found for {self.model_id} and {matrix_uuid}. Exiting!')
            raise ValueError(f'No predictions found {self.model_id} and {matrix_uuid}')

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

        ties_suffix = 'with_ties' if self.include_ties else 'no_ties'
        rank_column = f"rank_{self.threshold.split('_')[1]}_{ties_suffix}"
        threshold = self.threshold.split('_')[0]
       
        logging.info(f'Crosstabs using threshold: {rank_column} <= {threshold}')

        msk = matrix[rank_column] <= threshold
        postive_preds = matrix[msk][features]
        negative_preds = matrix[~msk][features]

        results = list()
        for name, func in crosstab_functions:
            logging.info(name)

            this_result = pd.DataFrame(func(postive_preds, negative_preds))
            this_result['metric'] = name
            results.append(this_result)
        
        results = pd.concat(results)
        results['threshold_type'] = rank_column
        results['threshold'] = threshold

        crosstabs_df = pd.concat(results).reset_index()

        crosstabs_df.rename(columns={'index': 'feature', 0: 'value'}, inplace=True)
        crosstabs_df['model_id'] = self.model_id
        crosstabs_df['matrix_uuid'] = matrix_uuid

    
        if push_to_db:
            logging.info(f'Pushing the results to the database, {len(crosstabs_df)} rows')
                    
            crosstabs_df.set_index(
                ['model_id', 'matrix_uuid', 'feature', 'metric', 'threshold_type', 'threshold'],
                inplace=True
            )
            
            crosstabs_df = crosstabs_df.reset_index()
            
            if not table_exists(f'test_results.{table_name}', self.engine):
                q = f'''
                    create schema if not exists test_results;
                    
                    create table test_results.{table_name} (
                    model_id INTEGER,
                    matrix_uuid TEXT,
                    feature TEXT,
                    metric TEXT,
                    threshold_type TEXT,
                    threshold FLOAT,
                    value FLOAT  
                    );
                
                '''
                # q = _generate_create_table_sql_statement_from_df(results, f'{table_schema}.{table_name}')
                self.engine.execute(q)
            
            conn = self.engine.raw_connection()
            cursor = conn.cursor()
            
            buffer = StringIO()
            crosstabs_df.to_csv(buffer, index=False, header=False)
            buffer.seek(0)
            
            columns = ', '.join(crosstabs_df.columns)
            cursor.copy_expert(f"COPY test_results.{table_name} ({columns}) FROM STDIN WITH CSV", buffer)
        
            conn.commit()
            cursor.close()
            conn.close()

        if return_df:
            return crosstabs_df  
        
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
        
        error_analysis_results = generate_error_analysis(self.model_id, self.engine, project_path=project_path)
        #TODO do we want to 
        return error_analysis_results    

    def score_distribution(self, matrix_uuid=None, subset_hash=None, n_bins=50, score_extent=(0, 1), display_chart=True, return_chart=True):
        predictions = self.model.predictions(matrix_uuid=matrix_uuid, subset_hash=subset_hash)
        
        alt.data_transformers.disable_max_rows()
        
        bar_chart = alt.Chart(predictions).mark_bar(
            opacity=0.8,
            binSpacing=0
        ).encode(
            x=alt.X('score').bin(maxbins=n_bins, extent=score_extent),
            y=alt.Y('count()').stack(None)
        )
        
        density_chart = alt.Chart(predictions).transform_density(
            'score',
            as_= ['score', 'density'],
            cumulative=False,
            extent=score_extent
        ).mark_area(opacity=0.5).encode(
            x=alt.X('score:Q'),
            y=alt.Y('density:Q').stack(None),
        )
        
        chart = (density_chart | bar_chart).properties(
            title=f'Score Distribution: {self.model.model_type} ({self.model_id})'
        )
        
        if display_chart:
            chart.display()
        
        if return_chart:
            return [density_chart, bar_chart]
               
        
"""This class is still WIP. Proceed with caution"""
class MultiModelAnalyzer: 

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
