"""
Model Evaluator

This script contain a set of elements to help the postmodeling evaluation
of audited models by triage.Audition. This will be a continuing list of
routines that can be scaled up and grow according to the needs of the
project, or other postmodeling approaches.

To run most of this routines you will need:
    - S3 credentials (if used) or specify the path to both feature and
    prediction matrices.
    - Working database conn.
"""

import pandas as pd
import numpy as np
from descriptors import cachedproperty
from sqlalchemy.sql import text
from matplotlib import pyplot as plt
from collections import namedtuple
from itertools import starmap, combinations
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
import seaborn as sns


from utils.aux_funcs import * 

# Get indivual model information/metadata from Audition output


class ModelGroupEvaluator(object):
    '''
    ModelGroup class calls the model group metadata from the database
    and hold metadata features on each of the class attibutes.
    This class will contain any information about the model_group, and 
    will be used to make comparisons and calculations across models. 

    A model_group_id list is needed to instate the class.
    '''
    def __init__(self, model_group_id):
        self.model_group_id = model_group_id

        # Retrive model_id metadata from the model_metadata schema
        model_metadata = pd.read_sql(
        f'''WITH
        individual_model_ids_metadata AS(
        SELECT m.model_id,
               m.model_group_id,
               m.hyperparameters,
               m.model_hash,
               m.experiment_hash,
               m.train_end_time,
               m.train_matrix_uuid,
               m.training_label_timespan,
               mg.model_type,
               mg.model_config
            FROM model_metadata.models m
            JOIN model_metadata.model_groups mg
            USING (model_group_id)
            WHERE model_group_id IN {self.model_group_id}
        ),
        individual_model_id_matrices AS(
        SELECT DISTINCT ON (matrix_uuid)
               model_id,
               matrix_uuid
            FROM test_results.predictions
            WHERE model_id = ANY(
                SELECT model_id
                FROM individual_model_ids_metadata
            )
        )
        SELECT metadata.model_id,
               metadata.model_group_id,
               metadata.model_hash,
               metadata.hyperparameters,
               metadata.train_end_time,
               metadata.train_matrix_uuid,
               metadata.training_label_timespan,
               metadata.model_type,
               metadata.model_config,
               test.matrix_uuid AS test_matrix_uuid
        FROM individual_model_ids_metadata AS metadata
        LEFT JOIN individual_model_id_matrices AS test
        USING(model_id);''', con=conn).to_dict('list')

        # Add metadata attributes to model
        self.model_id = model_metadata['model_id']
        self.model_type = model_metadata['model_type']
        self.train_end_time = model_metadata['train_end_time']
        self.hyperparameters = model_metadata['hyperparameters']
        self.model_hash = model_metadata['model_hash']
        self.train_matrix_uuid = model_metadata['train_matrix_uuid']
        self.test_matrix_uuid = model_metadata['test_matrix_uuid']

    def __repr__(self):
        return (
        f'Model collection object for model_ids:{self.model_id}\n'
        f'Model Groups: {self.model_group_id}\n'
        f'Model types: {self.model_type}\n'
        f'Model hyperparameters: {self.hyperparameters}\n'
        f'''Matrix hashes (train,test): [{self.train_matrix_uuid}, 
                                      {self.test_matrix_uuid}]'''
        )

    @cachedproperty
    def predictions(self):
        preds = pd.read_sql(
            f'''
            SELECT 
                   g.model_group_id,
                   m.model_id,
                   m.entity_id,
                   m.as_of_date,
                   m.score,
                   m.label_value,
                   COALESCE(rank_abs, RANK() OVER(PARTITION BY m.model_id
                   ORDER BY m.score DESC)) AS rank_abs,
                   COALESCE(m.rank_pct, percent_rank() over(ORDER BY m.score DESC)) *
                   100 AS rank_pct,
                   m.rank_pct,
                   m.test_label_timespan
            FROM test_results.predictions m
            LEFT JOIN model_metadata.models g
            USING (model_id)
            WHERE model_id IN {tuple(self.model_id)}
            AND label_value IS NOT NULL
            ''', con=conn)
        return preds

    @cachedproperty
    def feature_importances(self):
        features = pd.read_sql(
           f'''
           SELECT g.model_group_id,
                  m.model_id,
                  m.feature,
                  m.feature_importance,
                  m.rank_abs
           FROM train_results.feature_importances m
           LEFT JOIN model_metadata.models g
           USING (model_id)
           WHERE m.model_id IN {tuple(self.model_id)}
           ''', con=conn)
        return features

    @cachedproperty 
    def metrics(self):
        model_metrics = pd.read_sql(
            f'''
            SELECT g.model_group_id,
                   m.model_id,
                   EXTRACT('YEAR' FROM m.evaluation_end_time) AS as_of_date_year,
                   m.metric,
                   m.parameter,
                   m.value,
                   m.num_labeled_examples,
                   m.num_labeled_above_threshold,
                   m.num_positive_labels
            FROM test_results.evaluations m
            LEFT JOIN model_metadata.models g
            USING (model_id)
            WHERE m.model_id IN {tuple(self.model_id)}
            ''', con=conn)
        return model_metrics

    @cachedproperty
    def feature_groups(self):
        model_feature_groups = pd.read_sql(
            f'''
            WITH 
            feature_groups_raw AS( 
            SELECT 
            model_group_id,
            model_config->>'feature_groups' as features
            FROM model_metadata.model_groups
            WHERE model_group_id IN {self.model_group_id}
            ), 
            feature_groups_unnest AS(
            SELECT model_group_id,
            unnest(regexp_split_to_array(substring(features, '\[(.*?)\]'), ',')) AS group_array
            FROM feature_groups_raw
            ), feature_groups_array AS(
            SELECT 
            model_group_id,
            array_agg(split_part(substring(group_array, '\"(.*?)\"'), ':', 2)) AS feature_group_array
            FROM feature_groups_unnest
            GROUP BY model_group_id
            ), feature_groups_array_ AS(
            SELECT
            model_group_id,
            feature_group_array,
            array_length(feature_group_array, 1) AS number_feature_groups
            FROM feature_groups_array
            ), feature_groups_class_cases 
            AS( 
            SELECT
            model_group_id,
            feature_group_array,
            number_feature_groups,
            CASE
            WHEN number_feature_groups = 1
            THEN 'LOI'
            WHEN  number_feature_groups = first_value(number_feature_groups) OVER w
            THEN 'All features'
            WHEN  number_feature_groups = (first_value(number_feature_groups) OVER w) - 1
            THEN 'LOO'
            ELSE NULL
            END AS experiment_type
            FROM feature_groups_array_
            WINDOW w AS (ORDER BY number_feature_groups DESC)
            ) SELECT * FROM feature_groups_class_cases
            ''', con=conn)
        return model_feature_groups

    def plot_prec_across_time(self,
                              param_type=None,
                              param=None,
                              metric=None,
                              baseline=False,
                              baseline_query=None,
                              figsize=(12,16),
                              fontsize=20):

        '''
        Plot precision across time for all model_group_ids, and baseline,
        if available.

        This function plots the performance of each model_group_id following an
        user defined performance metric. First, this function check if the
        performance metrics are available to both the models, and the baseline.
        Second, filter the data of interest, and lastly, plot the results as
        timelines (model_id date). 
        '''

        # Load metrics and prepare data for analysis
        model_metrics = self.metrics
        model_metrics[['param', 'param_type']] = \
                model_metrics['parameter'].str.split('_', 1, expand=True)
        model_metrics['param'] =  model_metrics['param'].astype(str).astype(float)
        model_metrics['param_type'] = model_metrics['param_type'].apply(lambda x: 'rank_'+x)

        # Filter model_group_id metrics and create pivot table by each
        # model_group_id.
        model_metrics_filter = model_metrics[(model_metrics['metric'] == metric) & 
                                      (model_metrics['param'] == param) &
                                      (model_metrics['param_type'] == param_type)].\
                filter(['model_group_id', 'model_id', 'as_of_date_year',
                        'value'])
        if baseline == True:

            baseline_metrics = pd.read_sql(baseline_query, con=conn)
            baseline_metrics[['param', 'param_type']] = \
                    baseline_metrics['parameter'].str.split('_', 1, expand=True)
            baseline_metrics['param'] = baseline_metrics['param'].astype(str).astype(float)
            baseline_metrics['param_type'] = baseline_metrics['param_type'].apply(lambda x: 'rank_'+x)

            # Filter baseline metrics and create pivot table to join with
            # selected models metrics
            baseline_metrics_filter =  baseline_metrics[(baseline_metrics['metric'] == metric) &
                                                        (baseline_metrics['param'] == param) &
                                                        (baseline_metrics['param_type'] == param_type)].\
                filter(['model_group_id', 'model_id', 'as_of_date_year', 'value'])

            baseline_metrics_filter['model_group_id'] = \
                    baseline_metrics_filter.model_group_id.apply(lambda x: \
                                                                 'baseline_' + \
                                                                 str(x))
            # Join tables by index(as_of_date_year)
            model_metrics_filter = \
            model_metrics_filter.append(baseline_metrics_filter, sort=True)

        model_metrics_filter['as_of_date_year'] = \
                model_metrics_filter.as_of_date_year.astype('int')

        try:
            sns.set_style('whitegrid')
            fig, ax = plt.subplots(figsize=figsize)
            for model_group, df in model_metrics_filter.groupby(['model_group_id']):
                ax = ax = df.plot(ax=ax, kind='line', 
                                  x='as_of_date_year', 
                                  y='value',
                                  label=model_group)
            plt.title(str(metric).capitalize() +\
                      ' for selected model_groups in time.',
                      fontsize=fontsize)
            ax.tick_params(labelsize=16)
            ax.set_xlabel('Year of prediction (as_of_date)', fontsize=20)
            ax.set_ylabel(f'{str(metric)+str(param_type)+str(param)}',
                          fontsize=20)
            plt.xticks(model_metrics_filter.as_of_date_year.unique())
            plt.yticks(np.arange(0,1,0.1))
            legend=plt.legend(bbox_to_anchor=(1.05, 1),
                       loc=2,
                       borderaxespad=0.,
                       title='Model Group',
                       fontsize=fontsize)
            legend.get_title().set_fontsize('16')

        except TypeError:
                print(f'''
                      Oops! model_metrics_pivot table is empty. Several problems
                      can be creating this error:
                      1. Check that {param_type}@{param} exists in the evaluations
                      table 
                      2. Check that the metric {metric} is available to the
                      specified {param_type}@{param}.
                      3. You basline model can have different specifications.
                      Check those! 
                      4. Check overlap between baseline dates and model dates.
                      The join is using the dates for doing these, and it's
                      possible that your timestamps differ. 
                      ''')

    def feature_loi_loo(self,
                        model_subset=None,
                        param_type=None,
                        param=None,
                        baseline=True,
                        baseline_query=None):
        '''
        Plot metric for each model group across time 
        '''

        if model_subset is None:
            model_subset = self.model_group_id

        if baseline == True:
            baseline_metrics = pd.read_sql(baseline_query, con = conn)

        # Load feature groups and subset
        feature_groups = self.feature_groups
        feature_groups_filter = \
        feature_groups[feature_groups['model_group_id'].isin(model_subset)]

        # Load model metrics and subset 
        model_metrics = self.metrics
        model_metrics_filter = \
        model_metrics.loc[model_metrics['model_group_id'].isin(model_subset)]

        # Merge metrics and features and filter by threshold definition
        metrics_merge = model_metrics_filter.merge(feature_groups_filter,
                                                  how='inner',
                                                  on='model_group_id')


        # LOO and LOI definition
        df_loi = metrics_merge[metrics_merge['experiment_type'] == 'LOI']

    def _rank_corr_df(self,
                      model_pair,
                      corr_type: None,
                      param_type: None,
                      param: None,
                      top_n_features: 10):
        '''
        Calculates ranked correlations for ranked observations and features
        using the stats.spearmanr scipy module.
        Arguments: 
            - model_pair (tuple): tuple with model_ids
              observations or features
            - top_n (int): number of rows to rank (top-k model)
        '''

        if corr_type not in ['predictions', 'features']:
            raise Exception(
                f'''Wrong type! Rank correlation is not available\n
                   for {type}. Try the following options:\n 
                   predictions and features''')

        if corr_type == 'predictions':
            # Split df for each model_id 
            model_1 = self.predictions[self.predictions['model_id'] == model_pair[0]]
            model_2 = self.predictions[self.predictions['model_id'] == model_pair[1]]

            # Slice df to take top-n observations
            top_model_1 = model_1.sort_values(param_type, axis=0)[:param].set_index('entity_id')
            top_model_2 = model_2.sort_values(param_type, axis=0)[:param].set_index('entity_id')

            # Merge df's by entity_id and calculate corr
            df_pair_merge = top_model_1.merge(top_model_2, 
                                              how='inner',
                                              left_index=True,
                                              right_index=True,
                                              suffixes=['_1', '_2'])

            df_pair_filter = df_pair_merge.filter(regex='rank_abs*')
            rank_corr = spearmanr(df_pair_filter.iloc[:, 0], df_pair_filter.iloc[:, 1])

            # Return corr value (not p-value)
            return rank_corr[0]
        elif corr_type == 'features':
            # Split df for each model_id 
            model_1 = self.feature_importances[self.feature_importances['model_id'] == model_pair[0]]
            model_2 = self.feature_importances[self.feature_importances['model_id'] == model_pair[1]]

            # Slice df to take top-n observations
            top_model_1 = model_1.sort_values('rank_abs', \
                                              axis=0)[:top_n_features].set_index('feature')
            top_model_2 = model_2.sort_values('rank_abs', \
                                              axis=0)[:top_n_features].set_index('feature')

            # Merge df's by entity_id and calculate corr
            df_pair_merge = top_model_1.merge(top_model_2, 
                                              how='inner',
                                              left_index=True,
                                              right_index=True,
                                              suffixes=['_1', '_2'])

            df_pair_filter = df_pair_merge.filter(regex='rank_abs*')
            rank_corr = spearmanr(df_pair_filter.iloc[:, 0], df_pair_filter.iloc[:, 1])

            # Return corr value (not p-value)
            return rank_corr[0] 
        else:
            pass 

    def plot_ranked_corrlelation(self, 
                                 figsize=(12, 16),
                                 fontsize=20,
                                 model_subset=None,
                                 corr_type=None,
                                 **kwargs):
        '''
        Plot ranked correlation between model_id's using the _rank_corr_df
        method. The plot will visualize the selected correlation matrix
        including all the models
        Arguments:
            - figzise (tuple): tuple with figure size. Default is (12, 16)
            - fontsize (int): Fontsize for plot labels and titles. Default is
              20
            - model_subset (list): subset to only include a subset of model_ids
            - corr_type (str): correlation type. Two options are available:
                features and predictions. 
            - **kwargs: other parameters passed to the _rank_corr_df method
        '''

        if model_subset is None:
            model_subset = self.model_id

        if corr_type  == 'predictions':
            # Calculate rank correlations for predictions
            corrs = [self._rank_corr_df(pair,
                                        corr_type=corr_type,
                                        param=kwargs['param'],
                                        param_type=kwargs['param_type'],
                                        top_n_features=10
                                       ) for pair in combinations(model_subset, 2)]

            # Store results in dataframe using tuples
            corr_matrix = pd.DataFrame(index=model_subset, columns=model_subset)
            for pair, corr in zip(combinations(model_subset, 2), corrs):
                corr_matrix.loc[pair] = corr
        elif corr_type == 'features':

            # Calculate rank correlations for predictions
            corrs = [self._rank_corr_df(pair,
                                        corr_type=corr_type,
                                        param=kwargs['param'],
                                        param_type=kwargs['param_type'],
                                        top_n_features = kwargs \
                                        ['top_n_features']
                                       ) for pair in combinations(model_subset, 2)]

            # Store results in dataframe using tuples
            corr_matrix = pd.DataFrame(index=model_subset, columns=model_subset)
            for pair, corr in zip(combinations(model_subset, 2), corrs):
                corr_matrix.loc[pair] = corr
        else:
            raise AttributeError('Error!')

       # Process data for plot: mask repeated tuples
        corr_matrix_t = corr_matrix.T
        mask = np.zeros_like(corr_matrix_t)
        mask[np.triu_indices_from(mask, k=1)] = True

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('Model Id', fontsize=fontsize)
        ax.set_ylabel('Model Id', fontsize=fontsize)
        plt.title(f'''{corr_type} Rank Correlation for 
                  {kwargs['param_type']}@{kwargs['param']}
                 ''', fontsize=fontsize)
        sns.heatmap(corr_matrix_t.fillna(1),
                    mask=mask,
                    vmax=1,
                    vmin=0,
                    cmap='YlGnBu',
                    annot=True,
                    square=True)


    def plot_jaccard(self,
                     figsize=(12, 16), 
                     fontsize=20,
                     top_n=100,
                     model_subset=None,
                     temporal_comparison=True):
 
        if model_subset is None:
            model_subset = self.model_id

        if temporal_comparison == True:
            as_of_dates =  self.predictions['as_of_date'].unique()
            dfs_dates = [self.predictions[self.predictions['as_of_date']==date] 
                         for date in as_of_dates]

            for preds_df in dfs_dates:
               # Filter predictions dataframe by individual dates 
                df_preds_date = preds_df.copy() 
                df_preds_date['above_tresh'] = np.where(df_preds_date['rank_abs'] <= top_n, 1, 0) 
                df_sim_piv = df_preds_date.pivot(index='entity_id',
                                                 columns='model_id',
                                                 values='above_tresh')

                # Calculate Jaccard Similarity for the selected models
                res = pdist(df_sim_piv.T, 'jaccard')
                df_jac = pd.DataFrame(1-squareform(res), 
                                      index=preds_df.model_id.unique(),
                                      columns=preds_df.model_id.unique())
                mask = np.zeros_like(df_jac)
                mask[np.triu_indices_from(mask, k=1)] = True

                # Plot matrix heatmap
                fig, ax = plt.subplots(figsize=figsize)
                ax.set_xlabel('Model Id', fontsize=fontsize)
                ax.set_ylabel('Model Id', fontsize=fontsize)
                plt.title(f'''Jaccard Similarity Matrix Plot \
                          (as_of_date:{preds_df.as_of_date.unique()})
                          ''', fontsize=fontsize)
                sns.heatmap(df_jac,
                            mask=mask,
                            cmap='Greens', 
                            vmin=0, 
                            vmax=1, 
                            annot=True, 
                            linewidth=0.1)

        else:
                # Call predicitons
                df_sim = self.predictions
                df_sim['above_tresh_'] = (df_sim.rank_abs <= top_n).astype(int)
                df_sim_piv = df_sim.pivot(index='entity_id',
                                          columns='model_id',
                                          values='above_tresh_')

                # Calculate Jaccard Similarity for the selected models
                res = pdist(df_sim_piv[model_subset].T, 'jaccard')
                df_jac = pd.DataFrame(1-squareform(res),
                                      index=model_subset,
                                      columns=model_subset)
                mask = np.zeros_like(df_jac)
                mask[np.triu_indices_from(mask, k=1)] = True

                # Plot matrix heatmap
                fig, ax = plt.subplots(figsize=figsize)
                ax.set_xlabel('Model Id', fontsize=fontsize)
                ax.set_ylabel('Model Id', fontsize=fontsize)
                plt.title('Jaccard Similarity Matrix Plot', fontsize=fontsize)
                sns.heatmap(df_jac,
                            mask=mask, 
                            cmap='Greens', 
                            vmin=0, 
                            vmax=1, 
                            annot=True, 
                            linewidth=0.1)

   
