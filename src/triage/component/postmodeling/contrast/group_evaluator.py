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
from sklearn import metrics
from collections import namedtuple
from itertools import starmap
from scipy.spatial.distance import squareform, pdist
import seaborn as sns

from utils.file_helpers import download_s3
from utils.test_conn import db_engine
from utils.aux_funcs import recombine_categorical, pd_int_var

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
                      entity_id,
                      as_of_date,
                      feature,
                      method,
                      feature_value,
                      importance_score
               FROM test_results.individual_importances
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

    def _fetch_matrix(self, matrix_hash, path):
        '''
        Load matrices into object (Beware!)
        This can be a memory intensive process. This function will use the
        stored model parameters to load matrices from a .csv file. If the path
        is a s3 path, the method will use s3fs to open it. If the path is a local
        machine path, it will be opened as a pandas dataframe.

        Arguments:
            - PATH <s3 path or local path>
        '''
        if 's3' in path:
            mat = download_s3(str(path + self.train_matrix_uuid))
        else:
            mat = pd.read_csv(str(path + self.train_matrix_uuid + '.csv'))

        return(mat)

    def load_features_preds_matrix(self, *path):
        '''
        Load predicion matrix (from s3 or system file) and merge with
        label values from the test_results.predictions tables. The outcome
        is a pandas dataframe with a matrix for each entity_id, its predicted
        label, scores, and the feature matrix. This last object will be store
        as the pred_matrix attribute of the class.

        Arguments:
            - Arguments inherited from _fetch_matrices: 
                - path: relative path to the triage matrices folder or s3 path
        '''

        if self.train_matrix is None:
            mat = [self._fetch_matrix(hash_, *path) for hash_ in self.pred_matrix_uuid]
            mat_tuple = zip(mat, self.model_id)
            mat_named = list(starmap(pd_int_var(df, m_id, 'model_id'), mat_tuple))

        if self.preds is None:
            self._load_predictions()

        if self.feature_importances is None:
            self._load_feature_importances()

        if self.pred_matrix is None:
            # There is probably a less memory intense way of doing this
            pred_mat_tuple = zip(mat_named, self.preds)

            # Merge feature/prediction matrix with predictions
            merged_df_list = list(starmap(lambda m, p:
                                          m.merge(p,
                                                  on='entity_id',
                                                  how='inner',
                                                  suffixes=('test', 'pred')),
                                            pred_mat_tuple))
            self.pred_matrix = merged_df_list

    def load_train_matrix(self, *path):
        '''
        Load training metrix (from s3 or system file). This object will be store 
        as the train_matrix object of the class.

        Arguments:
            - Arguments inherited from _fecth_matrices:
                - path: relative path to the triage matrices folder or s3 path
        '''
        if self.train_matrix is None:
            self.train_matrix = self._fetch_matrix(self.train_matrix_uuid, *path)
 
    def plot_jaccard(self, 
                     figsize=(12, 16), 
                     fontsize=20,
                     top_n=100,
                     model_subset=None):
 
        if model_subset is None:
            model_subset = self.model_id

        if self.preds is None:
            self._load_predictions()


        # Call predicitons (this has to be filtered to comparable models)
        df_sim = self.preds
        df_sim['above_tresh'] = (df_sim['score'] <= top_n).astype(int)
        df_sim_piv = df_sim.pivot(index='entity_id', columns='model_id', values='above_tresh')

        # Calculate Jaccard Similarity for the selected models
        res = pdist(df_sim_piv[model_subset].T, 'jaccard')
        df_jac = pd.DataFrame(1-squareform(res), 
                              index=model_subset,
                              columns=model_subset)

        # Plot matrix heatmap
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('Model Id', fontsize=fontsize)
        ax.set_ylabel('Model Id', fontsize=fontsize)
        plt.title('Jaccard Similarity Matrix Plot', fontsize=fontsize)
        sns.heatmap(df_jac, cmap='Greens', vmin=0, vmax=1, annot=True, linewidth=0.1)










