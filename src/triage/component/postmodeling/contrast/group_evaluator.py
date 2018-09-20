"""
Model Evaluator

This script contain a set of elements to help the postmodeling evaluation
of audited models by triage.Audition. This will be a continuing list of
routines that can be scaled up and grow according to the needs of the
project, or other postmodeling approaches.

To run most of this routines you will need:
    - S3 credentials (if used) or specify the path to both feature and
    prediction matrices.
    - Working database db_engine.
"""

import pandas as pd
import numpy as np
from sqlalchemy.sql import text
from matplotlib import pyplot as plt
from collections import namedtuple
from itertools import starmap, combinations
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
import seaborn as sns

from utils.file_helpers import download_s3
from utils.test_conn import db_engine
from utils.aux_funcs import recombine_categorical

# Get indivual model information/metadata from Audition output


class ModelGroup(object):
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
        model_metadata = pd.read_sql("""
        WITH
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
            WHERE model_group_id IN {}
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
        USING(model_id);
        """.format(self.model_group_id),
                   con=db_engine).to_dict('list')

        # Add metadata attributes to model
        self.model_id = model_metadata['model_id']
        self.model_type = model_metadata['model_type']
        self.train_end_time = model_metadata['train_end_time']
        self.hyperparameters = model_metadata['hyperparameters']
        self.model_hash = model_metadata['model_hash']
        self.train_matrix_uuid = model_metadata['train_matrix_uuid']
        self.test_matrix_uuid = model_metadata['test_matrix_uuid']
        self.train_matrix = None
        self.pred_matrix = None
        self.preds = None
        self.feature_importances = None
        self.model_metrics = None

    def __str__(self):
        s = 'Model collection object for model_ids: {}'.format(self.model_id)
        s += '\n Model Groups: {}'.format(self.model_group_id)
        s += '\n Model types: {}'.format(self.model_type)
        s += '\n Model hyperparameters: {}'.format(self.hyperparameters)
        s += '\Matrix hashes (train,test): {}'.format((self.train_matrix_uuid,
                                                      self.test_matrix_uuid))
        return s

    def _load_predictions(self):
        preds = pd.read_sql("""
                SELECT model_id,
                       entity_id,
                       as_of_date,
                       score,
                       label_value,
                       COALESCE(rank_abs, RANK() 
                       OVER(PARTITION BY model_id ORDER BY score DESC)) AS rank_abs,
                       rank_pct,
                       test_label_timespan
                FROM test_results.predictions
                WHERE model_id IN {}
                AND label_value IS NOT NULL
                """.format(tuple(self.model_id)),
                con=db_engine)
        self.preds = preds

    def _load_feature_importances(self):
        features = pd.read_sql("""
               SELECT model_id,
                      feature,
                      feature_importance,
                      rank_abs
               FROM train_results.feature_importances 
               WHERE model_id IN {}
               """.format(tuple(self.model_id)),
               con=db_engine)
        self.feature_importances = features

    def _load_metrics(self):
        model_metrics = pd.read_sql("""
                SELECT model_id,
                       metric,
                       parameter,
                       value,
                       num_labeled_examples,
                       num_labeled_above_threshold
                       num_positive_labels
                FROM test_results.evaluations
                WHERE model_id IN {}
                """.format(tuple(self.model_id)),
                con=db_engine)
        self.model_metrics = model_metrics

    def _rank_corr_df(self,
                      model_pair,
                      top_n,
                      type):
        '''
        Calculates ranked correlations for ranked observations and features
        using the stats.spearmanr scipy module.
        Arguments: 
            - model_pair (tuple): tuple with model_ids
              observations or features
            - top_n (int): number of rows to rank (top-k model)
        '''

        if type not in ['predictions', 'features']:
            raise Exception('Wrong type! Rank correlation is not available'
                    + '\n for {}. Try the following options:'.format(type) 
                    + '\n predictions and features')

        if type is 'predictions':

            # Split df for each model_id 
            model_1 = self.preds[self.preds['model_id'] == model_pair[0]]
            model_2 = self.preds[self.preds['model_id'] == model_pair[1]]

            # Slice df to take top-n observations
            top_model_1 = model_1.sort_values('rank_abs', axis=0)[:top_n].set_index('entity_id')
            top_model_2 = model_2.sort_values('rank_abs', axis=0)[:top_n].set_index('entity_id')

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

        if type is 'features':

            # Split df for each model_id 
            model_1 = self.feature_importances[self.feature_importances['model_id'] == model_pair[0]]
            model_2 = self.feature_importances[self.feature_importances['model_id'] == model_pair[1]]

            # Slice df to take top-n observations
            top_model_1 = model_1.sort_values('rank_abs', axis=0)[:top_n].set_index('feature')
            top_model_2 = model_2.sort_values('rank_abs', axis=0)[:top_n].set_index('feature')

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
     


    def plot_jaccard(self,
                     figsize=(12, 16), 
                     fontsize=20,
                     top_n=100,
                     model_subset=None,
                     temporal_comparison=True):
 
        if model_subset is None:
            model_subset = self.model_id

        if self.preds is None:
            self._load_predictions()

        if temporal_comparison == True:
            as_of_dates =  self.preds['as_of_date'].unique()
            dfs_dates = [self.preds[self.preds['as_of_date']==date] 
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
                plt.title('Jaccard Similarity Matrix Plot (as_of_date:{})'.format(preds_df.as_of_date.unique()), fontsize=fontsize)
                sns.heatmap(df_jac,
                            mask=mask,
                            cmap='Greens', 
                            vmin=0, 
                            vmax=1, 
                            annot=True, 
                            linewidth=0.1)

        else:

                # Call predicitons
                df_sim = self.preds
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

    def plot_ranked_corrlelation(self, 
                                 figsize=(12, 16),
                                 fontsize=20,
                                 top_n=100,
                                 model_subset=None,
                                 type=None,
                                 ids=None):

        if model_subset is None:
            model_subset = self.model_id

        if type is 'predictions':
            if self.preds is None:
                self._load_predictions()

            # Calculate rank correlations for predictions
            corrs = [self._rank_corr_df(pair,
                                        type=type,
                                        top_n=top_n) for pair in combinations(model_subset, 2)]

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
            plt.title('Top-{} Predictions Rank Correlation'.format(top_n), fontsize=fontsize)
            sns.heatmap(corr_matrix_t.fillna(1),
                        mask=mask,
                        vmax=1,
                        vmin=0,
                        cmap='YlGnBu',
                        annot=True,
                        square=True)
    
        if type is 'features':
            if self.feature_importances is None:
                self._load_feature_importances()

            # Calculate rank correlations for predictions
            corrs = [self._rank_corr_df(pair,
                                        type=type,
                                        top_n=top_n) for pair in combinations(model_subset, 2)]

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
            plt.title('Top-{} Feature Rank Correlation'.format(top_n), fontsize=fontsize)
            sns.heatmap(corr_matrix_t.fillna(1),
                        mask=mask,
                        vmax=1,
                        vmin=0,
                        cmap='YlGnBu',
                        annot=True,
                        square=True)

