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
from sqlalchemy.sql import text
from matplotlib import pyplot as plt
from sklearn import metrics

from collections import namedtuple
from utils.file_helpers import download_s3
from utils.test_conn import db_engine

# Get individual model_ids from Audition outcome


def get_models_ids(audited_model_group_ids):
    query = db_engine.execute(text("""
    SELECT model_group_id,
           model_id
    FROM model_metadata.models
    WHERE model_group_id = ANY(:ids);
    """), ids=audited_model_group_ids)

    ModelTuple = namedtuple('Model', ['model_group_id', 'model_id'])
    l = [ModelTuple(*i) for i in query]

    return(l)

# Get indivual model information/metadata from Audition output

class ModelExtractor(object):
    '''
    ModelExtractor class calls the model metadata from the database
    and hold model_id metadata features on each of the class attibutes.
    This class will contain any information about the model, and will be
    used to make comparisons and calculations across models. 

    A pair of (model_group_id, model_id) is needed to instate the class. These
    can be feeded from the get models_ids. 
    '''
    def __init__(self, model_group_id, model_id):
        self.model_id = model_id
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
            WHERE model_group_id = {}
            AND model_id = {}
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
        SELECT metadata.*, test.*
        FROM individual_model_ids_metadata AS metadata
        LEFT JOIN individual_model_id_matrices AS test
        USING(model_id);
        """.format(self.model_group_id, self.model_id),
                   con=db_engine)

        # Add metadata attributes to model
        self.model_type = model_metadata.loc[0, 'model_type']
        self.hyperparameters = model_metadata.loc[0, 'hyperparameters']
        self.model_hash = model_metadata.loc[0, 'model_hash']
        self.train_matrix_uuid = model_metadata.loc[0, 'train_matrix_uuid']
        self.pred_matrix_uuid = model_metadata.loc[0, 'matrix_uuid']
        self.train_matrix = None
        self.pred_matrix = None
        self.preds = None
        self.feature_importances = None

    def _load_predictions(self):
        preds = pd.read_sql("""
                SELECT model_id,
                       entity_id,
                       as_of_date,
                       score,
                       label_value,
                       rank_abs,
                       rank_pct,
                       test_label_timespan
                FROM test_results.predictions
                WHERE model_id = {}
                AND label_value IS NOT NULL
                """.format(self.model_id),
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
               WHERE model_id = {}
               """.format(self.model_id),
               con=db_engine)
        self.feature_importances = features

    def _load_metrics(self):
        metrics = pd.read_sql("""
                SELECT model_id,
                       metric,
                       metric_parameter,
                       value,
                       num_labeled_examples,
                       num_labeled_above_treshold,
                       num_positive_labels
                FROM test_results.evaluation
                WHERE model_id = {}
                """.format(self.model_id),
                con=db_engine)

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
            mat = self._fetch_matrix(self.pred_matrix_uuid, *path)

        if self.preds is None:
            self._load_predictions()

        if self.feature_importances is None:
            self._load_feature_importances()

        if self.pred_matrix is None:
            # Merge feature/prediction matrix with predictions
            merged_df = mat.merge(self.preds,
                                  on='entity_id',
                                  how='inner',
                                  suffixes=('test', 'pred'))

            self.pred_matrix = merged_df

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

#    def plot_score_label_distributions(self):

    def plot_feature_importances(self, n_features=30, figsize=(16, 12)):
        """Generate a bar chart of the top n feature importances (by absolute value)
        Arguments:
           n_features (int) -- number of top features to plot
           figsize (tuple) -- figure size to pass to matplotlib
        """
        # TODO: allow more of the figure arguments to be passed to the method

        if self.feature_importances is None:
            self._load_feature_importances()

        humanized_featnames = self.feature_importances['feature']
        feature_importances = self.feature_importances['feature_value']

        # TODO: refactor to just make this a slice of self.features
        importances = list(zip(humanized_featnames, list(feature_importances)))
        importances = pd.DataFrame(importances, columns=['Feature', 'Score'])
        importances = importances.set_index('Feature')

        # Sort by the absolute value of the importance of the feature
        importances['sort'] = abs(importances['Score'])
        importances = importances.sort_values(by='sort', ascending=False).drop('sort', axis=1)
        importances = importances[0:n_features]
 
        # Show the most important positive feature at the top of the graph
        importances = importances.sort_values(by='Score', ascending=True)

        fig, ax = plt.subplots(figsize=figsize)
        ax.tick_params(labelsize=16)
        importances.plot(kind="barh", legend=False, ax=ax)
        ax.set_frame_on(False)
        ax.set_xlabel('Score', fontsize=20)
        ax.set_ylabel('Feature', fontsize=20)
        plt.tight_layout()
        plt.title('Top Feature Importances', fontsize=20).set_position([.5, 0.99])

 

