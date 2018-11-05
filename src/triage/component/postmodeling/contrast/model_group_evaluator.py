"""
Model Evaluator

This script contain a set of elements to help the postmodeling evaluation
of audited models by triage.Audition. This will be a continuing list of
routines that can be scaled up and grow according to the needs of the
project, or other postmodeling approaches.

"""

import pandas as pd
import numpy as np
import seaborn as sns
from descriptors import cachedproperty
from sqlalchemy.sql import text
from matplotlib import pyplot as plt
from collections import namedtuple
from itertools import starmap, combinations
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr

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

        if len(model_group_id) == 1:
            self.model_group_id = model_group_id + model_group_id
        else:
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
                   EXTRACT('YEAR' from m.as_of_date) AS as_of_date_year,
                   m.score,
                   m.label_value,
                   COALESCE(rank_abs, RANK() OVER(PARTITION BY m.model_id
                   ORDER BY m.score DESC)) AS rank_abs,
                   COALESCE(m.rank_pct, percent_rank() OVER(PARTITION BY
                   m.model_id ORDER BY m.score DESC)) * 100 AS rank_pct,
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

    @cachedproperty
    def same_time_models(self):
        time_models = pd.read_sql(
                      f'''
                      SELECT
                      train_end_time,
                      array_agg(model_id) AS model_id_array
                      FROM model_metadata.models
                      WHERE model_group_id IN {self.model_group_id}
                      GROUP BY train_end_time
                      ''', con = conn)
        return time_models

    def plot_prec_across_time(self,
                              param_type=None,
                              param=None,
                              metric=None,
                              baseline=False,
                              baseline_query=None,
                              df=False,
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

        Arguments:
            - param_type (string): A parameter string with a threshold
            definition: rank_pct, or rank_abs. These are usually defined in the
            postmodeling configuration file.
            - param (int): A threshold value compatible with the param_type.
            This value is also defined in the postmodeling configuration file.
            - metric (string): A string defnining the type of metric use to
            evaluate the model, this can be 'precision@', or 'recall@'.
            - baseline (bool): should we include a baseline for comparison?
            - baseline_query (str): a SQL query that returns the evaluation data from
            the baseline models. This value can also be defined in the
            configuration file.
            - df (bool): If True, no plot is rendered, but a pandas DataFrame
            is returned with the data.
            - figsize (tuple): tuple with figure size parameters
            - fontsize (int): Fontsize for titles
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
            baseline_metrics_filter =  baseline_metrics.filter(['model_group_id', 'model_id', 'as_of_date_year', 'value'])

            baseline_metrics_filter['model_group_id'] = \
                    baseline_metrics_filter.model_group_id.apply(lambda x: \
                                                                 'baseline_' + \
                                                                 str(x))
            # Join tables by index(as_of_date_year)
            model_metrics_filter = \
            model_metrics_filter.append(baseline_metrics_filter, sort=True)

        model_metrics_filter['as_of_date_year'] = \
                model_metrics_filter.as_of_date_year.astype('int')
        model_metrics_filter = model_metrics_filter.sort_values('as_of_date_year')

        if df == True:
            return model_metrics_filter

        else:
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
                        metric=None,
                        baseline=False,
                        baseline_query=None,
                        df=False,
                        figsize=(16,12),
                        fontsize=20):
        '''
        Plot precision across time for the selected model_group_ids, and
        include a leave-one-out, leave-one-in feature analysis to explore the
        leverage/changes of the selected metric across different models.

        This function plots the performance of each model_group_id following an
        user defined performance metric. First, this function check if the
        performance metrics are available to both the models, and the baseline.
        Second, filter the data of interest, and lastly, plot the results as
        timelines (model_id date).

        Arguments:
            -model_subset (list): A list of model_group_ids, in case you want
            to override the selected models.
            - param_type (string): A parameter string with a threshold
            definition: rank_pct, or rank_abs. These are usually defined in the
            postmodeling configuration file.
            - param (int): A threshold value compatible with the param_type.
            This value is also defined in the postmodeling configuration file.
            - metric (string): A string defnining the type of metric use to
            evaluate the model, this can be 'precision@', or 'recall@'.
            - baseline (bool): should we include a baseline for comparison?
            - baseline_query (str): a SQL query that returns the evaluation data from
            the baseline models. This value can also be defined in the
            configuration file.
            - df (bool): If True, no plot is rendered, but a pandas DataFrame
            is returned with the data.
            - figsize (tuple): tuple with figure size parameters
            - fontsize (int): Fontsize for titles

    '''

        if model_subset is None:
            model_subset = self.model_group_id

        # Load feature groups and subset
        feature_groups = self.feature_groups
        feature_groups_filter = \
        feature_groups[feature_groups['model_group_id'].isin(model_subset)]

        # Format metrics columns and filter by metric of interest
        model_metrics = self.metrics
        model_metrics[['param', 'param_type']] = \
                model_metrics['parameter'].str.split('_', 1, expand=True)
        model_metrics['param'] =  model_metrics['param'].astype(str).astype(float)
        model_metrics['param_type'] = model_metrics['param_type'].apply(lambda x: 'rank_'+x)

        model_metrics_filter = model_metrics[(model_metrics['metric'] == metric) &
                                      (model_metrics['param'] == param) &
                                      (model_metrics['param_type'] == param_type)].\
                filter(['model_group_id', 'model_id', 'as_of_date_year',
                        'value'])

        # Merge metrics and features and filter by threshold definition
        metrics_merge = model_metrics_filter.merge(feature_groups_filter,
                                                  how='inner',
                                                  on='model_group_id')

        # LOO and LOI definition
        all_features = set(metrics_merge.loc[metrics_merge['experiment_type'] == 'All features'] \
                           ['feature_group_array'][0])

        metrics_merge_experimental = \
        metrics_merge[metrics_merge['experiment_type'] != 'All features']

        metrics_merge_experimental['feature_experiment'] = \
                metrics_merge_experimental.apply(lambda row: \
                                   list(all_features.difference(row['feature_group_array']))[0] + '_loo' \
                                   if row['experiment_type'] == 'LOO' \
                                   else row['feature_group_array'] + '_loi', axis=1)

        metrics_merge_experimental = \
        metrics_merge_experimental.filter(['feature_experiment',
                                           'as_of_date_year', 'value'])

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

            baseline_metrics_filter['feature_experiment'] = \
                    baseline_metrics_filter.model_group_id.apply(lambda x: \
                                                                 'baseline_' + \
                                                                 str(x))

            metrics_merge_experimental = \
                    metrics_merge_experimental.append(baseline_metrics_filter,
                                                     sort=True)
        if df == True:
            return metrics_merge_experimental

        else:
            try:
                sns.set_style('whitegrid')
                fig, ax = plt.subplots(figsize=figsize)
                for feature, df in metrics_merge_experimental.groupby(['feature_experiment']):
                    ax = df.plot(ax=ax, kind='line',
                                      x='as_of_date_year',
                                      y='value',
                                      label=feature)
                metrics_merge[metrics_merge['experiment_type'] == 'All features']. \
                        groupby(['experiment_type']). \
                        plot(ax=ax,
                             kind='line',
                             x='as_of_date_year',
                             y='value',
                             label='All features')
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
                           title='Experiment Type',
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

    def _rank_corr_df(self,
                      model_pair,
                      top_n_features: 10,
                      corr_type=None,
                      param_type=None,
                      param=None):
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
            top_model_1 = model_1[model_1[param_type] < param].set_index('entity_id')
            top_model_2 = model_2[model_2[param_type] < param].set_index('entity_id')

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
            model_1 = \
                    self.feature_importances[self.feature_importances['model_id'] \
                                             == model_pair[0]]
            model_2 = \
                    self.feature_importances[self.feature_importances['model_id'] \
                                             == model_pair[1]]

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

    def plot_ranked_corrlelation_preds(self,
                                       model_subset=None,
                                       temporal_comparison=False,
                                       figsize=(12, 16),
                                       fontsize=20,
                                       **kwargs):
        '''
        Plot ranked correlation between model_id's using the _rank_corr_df
        method. The plot will visualize the selected correlation matrix
        including all the models
        Arguments:
            - model_subset (list): subset to only include a subset of model_ids
            - corr_type (str): correlation type. Two options are available:
                features and predictions.
            - temporal_comparison (bool): Compare same prediction window models?
              Default is False
            - figzise (tuple): tuple with figure size. Default is (12, 16)
            - fontsize (int): Fontsize for plot labels and titles. Default is
              20
            - **kwargs: other parameters passed to the _rank_corr_df method
        '''
        if model_subset is None:
            model_subset = self.model_id
        models_to_use = []
        model_as_of_date = []
        for key, values in \
            self.same_time_models[['train_end_time', 'model_id_array']].iterrows():
            if values[1][0] in model_subset:
                models_to_use.append(values[1][0])
                model_as_of_date.append(values[0])
        model_subset = models_to_use

        if temporal_comparison == True:
            for key, values in \
                self.same_time_models[['train_end_time', 'model_id_array']].iterrows():

                model_subset = values[1]
                model_as_of_date = values[0]

                # Calculate rank correlations for predictions
                corrs = [self._rank_corr_df(pair,
                         corr_type='predictions',
                         param=kwargs['param'],
                         param_type=kwargs['param_type'],
                         top_n_features=10
                         ) for pair in combinations(model_subset, 2)]

                # Store results in dataframe using tuples
                corr_matrix = pd.DataFrame(index=model_subset, columns=model_subset)
                for pair, corr in zip(combinations(model_subset, 2), corrs):
                    corr_matrix.loc[pair] = corr

                # Process data for plot: mask repeated tuples
                corr_matrix_t = corr_matrix.T
                mask = np.zeros_like(corr_matrix_t)
                mask[np.triu_indices_from(mask, k=1)] = True

                fig, ax = plt.subplots(figsize=figsize)
                ax.set_xlabel('Model Id', fontsize=fontsize)
                ax.set_ylabel('Model Id', fontsize=fontsize)
                plt.title(f'''Predictions Rank Correlation for
                          {kwargs['param_type']}@{kwargs['param']}
                          (date: {model_as_of_date})
                        ''', fontsize=fontsize)
                sns.heatmap(corr_matrix_t.fillna(1),
                            mask=mask,
                            vmax=1,
                            vmin=0,
                            cmap='YlGnBu',
                            annot=True,
                            square=True)

        else:
            corrs = [self._rank_corr_df(pair,
                                        corr_type='predictions',
                                        param=kwargs['param'],
                                        param_type=kwargs['param_type'],
                                        top_n_features=10
                                        ) for pair in combinations(model_subset, 2)]

            # Store results in dataframe using tuples
            corr_matrix = pd.DataFrame(index=model_subset, columns=model_subset)
            for pair, corr in zip(combinations(model_subset, 2), corrs):
                corr_matrix.loc[pair] = corr

            # Process data for plot: mask repeated tuples
            corr_matrix_t = corr_matrix.T
            mask = np.zeros_like(corr_matrix_t)
            mask[np.triu_indices_from(mask, k=1)] = True

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel('Model Id', fontsize=fontsize)
            ax.set_ylabel('Model Id', fontsize=fontsize)
            plt.title(f'''Predictions Rank Correlation for
                      {kwargs['param_type']}@{kwargs['param']}
                     ''', fontsize=fontsize)
            sns.heatmap(corr_matrix_t.fillna(1),
                        mask=mask,
                        vmax=1,
                        vmin=0,
                        cmap='YlGnBu',
                        annot=True,
                        square=True)


    def plot_ranked_correlation_features(self,
                                         model_subset=None,
                                         temporal_comparison=False,
                                         figsize=(12, 16),
                                         fontsize=20,
                                         **kwargs):
        '''
        Plot ranked correlation between model_id's using the _rank_corr_df
        method. The plot will visualize the selected correlation matrix
        including all the models
        Arguments:
            - model_subset (list): subset to only include a subset of model_ids
            - corr_type (str): correlation type. Two options are available:
                features and predictions.
            - temporal_comarison (bool): Compare same prediction window models?
              Default is False
            - figzise (tuple): tuple with figure size. Default is (12, 16)
            - fontsize (int): Fontsize for plot labels and titles. Default is
              20
            - **kwargs: other parameters passed to the _rank_corr_df method
        '''

        if model_subset is None:
            model_subset = self.model_id
        models_to_use = []
        model_as_of_date = []
        for key, values in \
            self.same_time_models[['train_end_time', 'model_id_array']].iterrows():
            if values[1][0] in model_subset:
                models_to_use.append(values[1][0])
                model_as_of_date.append(values[0])
        model_subset = models_to_use

        if  temporal_comparison == True:

            # Calculate rank correlations for features 
            corrs = [self._rank_corr_df(pair,
                                        corr_type='features',
                                        top_n_features = kwargs \
                                        ['top_n_features']
                                       ) for pair in combinations(model_subset, 2)]

            # Store results in dataframe using tuples
            corr_matrix = pd.DataFrame(index=model_subset, columns=model_subset)
            for pair, corr in zip(combinations(model_subset, 2), corrs):
                corr_matrix.loc[pair] = corr

            # Process data for plot: mask repeated tuples
            corr_matrix_t = corr_matrix.T
            mask = np.zeros_like(corr_matrix_t)
            mask[np.triu_indices_from(mask, k=1)] = True

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel('Model Id', fontsize=fontsize)
            ax.set_ylabel('Model Id', fontsize=fontsize)
            plt.title(f'''Feature Rank Correlation for
                      Top-{kwargs['top_n_features']}
                     (date: {model_as_of_date})
                    ''', fontsize=fontsize)
            sns.heatmap(corr_matrix_t.fillna(1),
                        mask=mask,
                        vmax=1,
                        vmin=0,
                        cmap='YlGnBu',
                        annot=True,
                        square=True)

        else:
            corrs = [self._rank_corr_df(pair,
                                        corr_type='features',
                                        top_n_features=10
                                        ) for pair in combinations(model_subset, 2)]
            # Store results in dataframe using tuples
            corr_matrix = pd.DataFrame(index=model_subset, columns=model_subset)
            for pair, corr in zip(combinations(model_subset, 2), corrs):
                corr_matrix.loc[pair] = corr

            # Process data for plot: mask repeated tuples
            corr_matrix_t = corr_matrix.T
            mask = np.zeros_like(corr_matrix_t)
            mask[np.triu_indices_from(mask, k=1)] = True

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel('Model Id', fontsize=fontsize)
            ax.set_ylabel('Model Id', fontsize=fontsize)
            plt.title(f'''Feature Rank Correlation for
                      Top-{kwargs['top_n_features']}
                    ''', fontsize=fontsize)
            sns.heatmap(corr_matrix_t.fillna(1),
                        mask=mask,
                        vmax=1,
                        vmin=0,
                        cmap='YlGnBu',
                        annot=True,
                        square=True)

    def plot_jaccard(self,
                     param_type=None,
                     param=None,
                     model_subset=None,
                     temporal_comparison=False,
                     figsize=(12, 16),
                     fontsize=20):

        if model_subset is None:
            model_subset = self.model_id

        preds = self.predictions
        preds_filter = preds[preds['model_id'].isin(self.model_id)]

        if temporal_comparison == True:
            try:
                for key, values in \
                self.same_time_models[['train_end_time', 'model_id_array']].iterrows():
                    preds_filter_group = \
                    preds_filter[preds_filter['model_id'].isin(values[1])]
                    # Filter predictions dataframe by individual dates
                    if param_type == 'rank_abs':
                        df_preds_date = preds_filter_group.copy()
                        df_preds_date['above_tresh'] = \
                                np.where(df_preds_date['rank_abs'] <= param, 1, 0)
                        df_sim_piv = df_preds_date.pivot(index='entity_id',
                                                         columns='model_id',
                                                         values='above_tresh')
                    elif param_type == 'rank_pct':
                        df_preds_date = preds_filter_group.copy()
                        df_preds_date['above_tresh'] = \
                                np.where(df_preds_date['rank_pct'] <= param, 1, 0)
                        df_sim_piv = df_preds_date.pivot(index=['entity_id',
                                                                'as_of_date'],
                                                         columns='model_id',
                                                         values='above_tresh')
                    else:
                        raise AttributeError('''Error! You have to define a parameter type to
                                             set up a threshold
                                             ''')
                # Calculate Jaccard Similarity for the selected models
                res = pdist(df_sim_piv.T, 'jaccard')
                df_jac = pd.DataFrame(1-squareform(res),
                                      index=preds_filter_group.model_id.unique(),
                                      columns=preds_filter_group.model_id.unique())

                mask = np.zeros_like(df_jac)
                mask[np.triu_indices_from(mask, k=1)] = True

                # Plot matrix heatmap
                fig, ax = plt.subplots(figsize=figsize)
                ax.set_xlabel('Model Id', fontsize=fontsize)
                ax.set_ylabel('Model Id', fontsize=fontsize)
                plt.title(f'''Jaccard Similarity Matrix Plot \n
                          (as_of_date:{values[0]})
                          ''', fontsize=fontsize)
                sns.heatmap(df_jac,
                            mask=mask,
                            cmap='Greens',
                            vmin=0,
                            vmax=1,
                            annot=True,
                            linewidth=0.1)
            except ValueError:
                print(f'''
                      Temporal comparison can be only made for more than one
                      model group.
                     ''')

        else:
                # Call predicitons
                if param_type == 'rank_abs':
                    df_preds_date = preds_filter.copy()
                    df_preds_date['above_tresh'] = \
                            np.where(df_preds_date['rank_abs'] <= param, 1, 0)
                    df_sim_piv = df_preds_date.pivot(index='entity_id',
                                                     columns='model_id',
                                                     values='above_tresh')
                elif param_type == 'rank_pct':
                    df_preds_date = preds_filter.copy()
                    df_preds_date['above_tresh'] = \
                            np.where(df_preds_date['rank_pct'] <= param, 1, 0)
                    df_sim_piv = df_preds_date.pivot(index='entity_id',
                                                     columns='model_id',
                                                     values='above_tresh')
                else:
                    raise AttributeError('''Error! You have to define a parameter type to
                                         set up a threshold
                                         ''')

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
