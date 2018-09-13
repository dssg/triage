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

from utils.file_helpers import download_s3
from utils.test_conn import db_engine
from utils.aux_funcs import recombine_categorical

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


class Model(object):
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
        self.model_metrics = None

    def __str__(self):
        s = 'Model object for model_id: {}'.format(self.model_id)
        s += '\n Model Group: {}'.format(self.model_group_id)
        s += '\n Model type: {}'.format(self.model_type)
        s += '\n Model hyperparameters: {}'.format(self.hyperparameters)
        s += '\Matrix hashes (train,test): {}'.format([self.train_matrix_uuid,
                                                      self.pred_matrix_uuid])
        return s

    def _load_predictions(self):
        preds = pd.read_sql("""
                SELECT model_id,
                       entity_id,
                       as_of_date,
                       score,
                       label_value,
                       COALESCE(rank_abs, RANK() OVER(ORDER BY score DESC)) AS rank_abs,
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
                      feature,
                      feature_importance,
                      rank_abs
               FROM train_results.feature_importances
               WHERE model_id = {}
               """.format(self.model_id),
               con=db_engine)
        self.feature_importances = features

    def _load_metrics(self):
        model_metrics = pd.read_sql("""
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

    def plot_score_label_distributions(self, 
                                       save_file=False, 
                                       name_file=None,
                                       label_names = ('Label = 0', 'Label = 1'),
                                       figsize=(16, 12),
                                       fontsize=20):
        '''
        Generate a histogram showing the label distribution for predicted
        entities:
            - Arguments:
                - save_file (bool): save file to disk as png. Default is False.
                - name_file (string): specify name file for saved plot.
                - label_names(tuple): define custom label names for class.
                - figsize (tuple): specify size of plot. 
                - fontsize (int): define custom fontsize. 20 is set by default.
        '''
        if self.preds is None:
            self._load_predictions()

        df_preds = self.preds.filter(items=['score', 'label_value'])
        df_preds_0 = df_preds[df_preds.label_value == 0]
        df_preds_1 = df_preds[df_preds.label_value == 1]

        fig, ax = plt.subplots(1, figsize=figsize)
        plt.hist(df_preds_0.score,
                 bins=20,
                 normed=True,
                 alpha=0.5,
                 color='skyblue',
                 label=label_names[0])
        plt.hist(list(df_preds_1.score),
                 bins=20,
                 normed=True,
                 alpha=0.5,
                 color='orange',
                 label=label_names[1])
        plt.axvline(df_preds_0.score.mean(),
                    color='skyblue',
                    linestyle='dashed')
        plt.axvline(df_preds_1.score.mean(),
                    color='orange',
                    linestyle='dashed')
        plt.legend(bbox_to_anchor=(0., 1.005, 1., .102),
                   loc=8,
                   ncol=2,
                   borderaxespad=0.,
                   fontsize=fontsize-2)
        ax.set_xlabel('Frequency', fontsize=fontsize)
        ax.set_ylabel('Score', fontsize=fontsize)
        plt.title('Score Distribution across Predicted Labels', y =1.2, 
                  fontsize=fontsize)
        plt.show()

        if save_file:
            plt.savefig(str(name_file + '.png'))

    def plot_feature_importances(self,
                                 n_features=30,
                                 figsize=(16, 12),
                                 fontsize=20):
        '''
        Generate a bar chart of the top n feature importances (by absolute value)
        Arguments:
        - n_features (int): number of top features to plot
        - figsize (tuple): figure size to pass to matplotlib
        - fontsize (int): define custom fontsize for labels and legends. 20 is default.
        '''

        if self.feature_importances is None:
            self._load_feature_importances()

        importances = self.feature_importances.filter(items=['feature', 'feature_importance'])
        importances = importances.set_index('feature')

        # Sort by the absolute value of the importance of the feature
        importances['sort'] = abs(importances['feature_importance'])
        importances = importances.sort_values(by='sort', ascending=False).drop('sort', axis=1)
        importances = importances[0:n_features]

        # Show the most important positive feature at the top of the graph
        importances = importances.sort_values(by='feature_importance', ascending=True)

        fig, ax = plt.subplots(figsize=figsize)
        ax.tick_params(labelsize=16)
        importances.plot(kind="barh", legend=False, ax=ax)
        ax.set_frame_on(False)
        ax.set_xlabel('Score', fontsize=20)
        ax.set_ylabel('Feature', fontsize=20)
        plt.tight_layout()
        plt.title('Top {} Feature Importances'.format(n_features), fontsize=fontsize).set_position([.5, 0.99])

    def compute_AUC(self):
        '''
        Utility function to generate ROC and AUC data to plot ROC curve
        Returns (tuple): 
            - (false positive rate, true positive rate, thresholds, AUC)
        '''

        if self.preds is None:
            self._load_predictions()

        label_preds = self.preds.label_value
        score_preds = self.preds.score
        fpr, tpr, thresholds = metrics.roc_curve(
            label_preds, score_preds, pos_label=1)

        return (fpr, tpr, thresholds, metrics.auc(fpr, tpr))

    def plot_ROC(self, 
                 figsize=(16, 12),
                 fontsize=20):
        '''
        Plot an ROC curve for this model and label it with AUC
        using the sklearn.metrics methods. 
        Arguments:
            - figsize (tuple): figure size to pass to matplotlib
            - fontsize (int): fontsize for title. Default is 20.
        '''

        if self.preds is None:
            self._load_predictions()

        fpr, tpr, thresholds, auc = self.compute_AUC()
        auc = "%.2f" % auc

        title = 'ROC Curve, AUC = ' + str(auc)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, "#000099", label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=fontsize)
        plt.ylabel('True Positive Rate', fontsize=fontsize)
        plt.legend(loc='lower right')
        plt.title(title, fontsize=fontsize)
        plt.show()

    def plot_recall_fpr_n(self,
                          figsize=(16, 12),
                          fontsize=20):
        '''
        Plot recall and the false positive rate against depth into the list
        (esentially just a deconstructed ROC curve) along with optimal bounds.
        Arguments:
            - figsize (tuple): define a plot size to pass to matplotlib
            - fontsize (int): define a custom font size to labels and axes 
        '''

        if self.preds is None:
            self._load_predictions()

        preds_labels = self.preds.filter(items=['label_value', 'score'])

        # since tpr and recall are different names for the same metric, we can just
        # grab fpr and tpr from the sklearn function for ROC
        fpr, recall, thresholds, auc = self.compute_AUC()

        # turn the thresholds into percent of the list traversed from top to bottom
        pct_above_per_thresh = []
        num_scored = float(len(preds_labels.score))
        for value in thresholds:
            num_above_thresh = len(preds_labels.loc[preds_labels.score >= value, 'score'])
            pct_above_thresh = num_above_thresh / num_scored
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)

        # plot the false positive rate, along with a dashed line showing the optimal bounds
        # given the proportion of positive labels
        plt.clf()
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot([preds_labels.label_value.mean(), 1], [0, 1], '--', color='gray')
        ax1.plot([0, preds_labels.label_value.mean()], [0, 0], '--', color='gray')
        ax1.plot(pct_above_per_thresh, fpr, "#000099")
        plt.title('Deconstructed ROC Curve\nRecall and FPR vs. Depth',fontsize=fontsize)
        ax1.set_xlabel('Proportion of Population', fontsize=fontsize)
        ax1.set_ylabel('False Positive Rate', color="#000099", fontsize=fontsize)
        plt.ylim([0.0, 1.05])

    def plot_precision_recall_n(self,
                                figsize=(16, 12),
                                fontsize=20):
        """Plot recall and precision curves against depth into the list.
        """

        if self.preds is None:
            self._load_predictions()

        preds_labels = self.preds.filter(items=['label_value', 'score'])

        y_score = preds_labels.score
        precision_curve, recall_curve, pr_thresholds = \
            metrics.precision_recall_curve(preds_labels.label_value, y_score)

        precision_curve = precision_curve[:-1]
        recall_curve = recall_curve[:-1]

        # a little bit of a hack, but ensure we start with a cut-off of 0
        # to extend lines to full [0,1] range of the graph
        pr_thresholds = np.insert(pr_thresholds, 0, 0)
        precision_curve = np.insert(precision_curve, 0, precision_curve[0])
        recall_curve = np.insert(recall_curve, 0, recall_curve[0])

        pct_above_per_thresh = []
        number_scored = len(y_score)
        for value in pr_thresholds:
            num_above_thresh = len(y_score[y_score >= value])
            pct_above_thresh = num_above_thresh / float(number_scored)
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)

        plt.clf()
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(pct_above_per_thresh, precision_curve, "#000099")
        ax1.set_xlabel('Proportion of population', fontsize=fontsize)
        ax1.set_ylabel('Precision', color="#000099", fontsize=fontsize)
        plt.ylim([0.0, 1.05])
        ax2 = ax1.twinx()
        ax2.plot(pct_above_per_thresh, recall_curve, "#CC0000")
        ax2.set_ylabel('Recall', color="#CC0000", fontsize=fontsize)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title("Precision-recall at x-proportion", fontsize=fontsize)
        plt.show()

    def test_feature_diffs(self, feature_type, name_or_prefix, suffix='',
                           score_col='score', 
                           entity_col='entity_id',
                           cut_n=None, cut_frac=None, cut_score=None,
                           nbins=None, 
                           thresh_labels=['Above Threshold', 'Below Threshold'],
                           figsize=(12, 16), 
                           figtitle=None, 
                           xticks_map=None,
                           figfontsize=14, 
                           figtitlesize=16,
                           *path):
        '''
        Look at the differences in a feature across a score cut-off. You can specify one of
        the top n (e.g., cut_n=300), top x% (e.g. cut_frac=0.10 for 10%), or a specific score
        cut-off to use (e.g., cut_score=0.4815). Since our matrices have integer values for all
        features, you'll need to specify the type of feature.

        Arguments:
            feature_type:   one of ['continuous', 'categorical', 'boolean']
                            note that categorical columns are assumed mutually exclusive
            name_or_prefix: the feature (column) name for a continuous or boolean feature, or the
                            prefix for a set of categorical features that span several columns
            suffix:         suffix for categorical features column names (usually a collate aggregate)
            score_col:      name of the column with scores in the dataframe df
            entity_col:     name of the column with the entity identifiers
            cut_n:          top n by score_col to use for the cut-off
            cut_frac:       top fraction (between 0 and 1) by score_col to use for the cut-off
            cut_score:      score cut-off
            nbins:          optional parameter for number of bins in continuous histogram
            tresh_lables:   list of labels for entities above and below the threshold, respectively
            figsize:        figure size tuple to pass through to matplotlib
            figtitle:       optional title for the plot
            xticks_map:     optional mapping for a categorical/boolean variable with more readable names for the values
            figfontsize:    font size to use for the plot
            figtitlesize:   font size to use for the plot title
            *path:     arguments to pass through to fetch_features_and_pred_matrix()
        '''

        if self.pred_matrix is None:
            self.load_features_preds_matrix(*path)

        # sort and select columns we'll need
        if feature_type in ['continuous', 'boolean']:
            df_sorted = self.pred_matrix[[entity_col, score_col, name_or_prefix]].copy()
            df_sorted['sort_random'] = np.random.random(len(df_sorted))
            df_sorted.sort_values([score_col, 'sort_random'], ascending=[False, False], inplace=True)

        # for categoricals, combine columns into one categorical column
        # NOTE: assumes the categoricals are mutually exclusive
        elif feature_type == 'categorical':
            df_sorted = self.pred_matrix[[entity_col, score_col]].copy()
            df_sorted['sort_random'] = np.random.random(len(df_sorted))
            df_sorted.sort_values([score_col, 'sort_random'], ascending=[False, False], inplace=True)

            df_sorted, cat_col = recombine_categorical(df_sorted, self.pred_matrix, name_or_prefix, suffix, entity_col)

        else:
            raise ValueError('feature_type must be one of continuous, boolean, or categorical')


        # calculate the other two cut variables depending on which is specified
        if cut_n:
            cut_score = df_sorted[:cut_n][score_col].min()
            cut_frac = 1.0*cut_n / len(df_sorted)

        elif cut_frac:
            cut_n = int(np.ceil(len(df_sorted)*cut_frac))
            cut_score = df_sorted[:cut_n][score_col].min()

        elif cut_score:
            cut_n = len(df_sorted.loc[df_sorted[score_col] >= cut_score])
            cut_frac = 1.0*cut_n / len(df_sorted)

        else:
            raise ValueError('Must specify one of cut_n, cut_frac, or cut_score')
     # seems like there should be a way to do this without having to touch the index?
        df_sorted.reset_index(inplace=True)
        df_sorted['above_thresh'] = df_sorted.index < cut_n
        df_sorted['above_thresh'] = df_sorted['above_thresh'].map({True: thresh_labels[0], False: thresh_labels[1]})


        # for booleans and categoricals, plot the discrete distributions and calculate chi-squared stats
        if feature_type in ['categorical', 'boolean']:
            if feature_type == 'boolean':
                cat_col = name_or_prefix

            self.categorical_plot_and_stats(df_sorted, cat_col, 'above_thresh',
                                            y_axis_pct=True, figsize=figsize, figtitle=figtitle,
                                            xticks_map=xticks_map,
                                            figfontsize=figfontsize, figtitlesize=figtitlesize)

        # for continuous variables, plot histograms and calculate some stats
        else:
            self.continuous_plot_and_stats(df_sorted, name_or_prefix, 'above_thresh', nbins,
                                           y_axis_pct=True, figsize=figsize, figtitle=figtitle,
                                           figfontsize=figfontsize, figtitlesize=figtitlesize)

