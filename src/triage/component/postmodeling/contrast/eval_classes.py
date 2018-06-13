# Model Evaluations

# This code contains a couple of classes useful for evaluating models built with triage, and generally
# expects the standard triage results schema to be available. Primarily it consists of a collection of
# methods for generating analyses and plots to be used with jupyter notebooks.

# The ExperimentEvaluation class contains methods for looking at metrics accross model groups, including
# Jaccard Similarity of top N lists, plotting evaluation metrics over time by model type or hyperparameters,
# and identifying model configurations that are consistently performing well.

# The ModelGroupEvaluation class contains methods for looking at the performance of a single model group,
# including stack ranking plots an score histograms pooled across the models in the group.

# The ModelEvaluation class contains methods for looking at the performance of an individual model:
# ROC curves, precision/recall curves, recall/fpr curves, feature importances, and feature distributions
# across the top vs rest of a list. It also contains a few utility methods, such as for fetching
# train and test matrices from an S3 bucket.

# In addition to the modules required here, database credentials must be loaded into environment variables
# and fetching matrices from S3 requires AWS credentials in either ~/.boto or environment vars


from drain import util
import boto3
import os, sys
import json

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import scipy.stats as scs
from scipy.spatial.distance import squareform, pdist

import pandas as pd
import numpy as np
from sklearn import metrics

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

# boto3 logging is very noisy by default, so quiet it down
import logging
logging.getLogger('boto3').setLevel(logging.WARN)
logging.getLogger('botocore').setLevel(logging.WARN)
logging.getLogger('nose').setLevel(logging.WARN)
logging.getLogger('s3transfer').setLevel(logging.WARN)



def plot_cats(frame, x_col, y_col, cat_col='model_type', grp_col='model_group_id',
              title='', x_label='', y_label='', cmap_name='Vega10',
              figsize=[12,6], x_ticks = None, y_ticks = None,
              legend_loc=None, legend_fontsize=12,
              label_fontsize=12, title_fontsize=16,
              label_fcn=None):
    """Plot a line plot with each line colored by a category variable.

    Arguments:
        frame (DataFrame) -- a dataframe containing the data to be plotted
        x_col (string) -- name of the x-axis column
        y_col (string) -- name of the y-axis column
        cat_col (string) -- name of the catagory column to color lines
        grp_col (string) -- column that identifies each group of (x_col, y_col) points for each line
        title (string) -- allows specifying a custom title for the graph
        x_label (string) -- allows specifying a custom label for the x-axis
        y_label (string) -- allows specifying a custom label for the y-axis
        cmap_name (string) -- matplotlib color map name to use for plot
        figsize (tuple) -- figure size to pass to matplotlib
        x_ticks (sequence) -- optional ticks to use for x-axis
        y_ticks (sequence) -- optional ticks to use for y-axis
        legend_loc (string) -- allows specifying location of plot legend
        legend_fontsize (int) -- allows specifying font size for legend
        label_fontsize (int) -- allows specifying font size for axis labels
        title_fontzie (int) -- allows specifying font size for plot title
        label_fcn (method) -- function to map category names to more readable names, accepting values of cat_col
    """

    fig, ax = plt.subplots(1,1,figsize=figsize)

    # function for parsing cat_col values into more readable legend lables
    if label_fcn is None and cat_col=='model_type':
        label_fcn = lambda x: x.split('.')[-1]
    elif label_fcn is None:
        label_fcn = lambda x: x

    # seems like you can do everything but color the lines with a simple groupby()... too bad
    # cl = ['b']*5 + ['r']*28
    # df_bestdist.groupby('model_group_id').plot(x='pct_diff', y='pct_of_time', xlim=[0,1], ylim=[0,1.1], ax=ax, color=cl)
    # plt.show()


    categories = np.unique(frame[cat_col])

    # want to step through the discrete color map rather than sampling
    # across the entire range, so create an even spacing from 0 to 1
    # with as many steps as in the color map (cmap.N), then repeat it
    # enough times to ensure we cover all our categories
    cmap = plt.get_cmap(cmap_name)
    ncyc = int(np.ceil(1.0*len(categories) / cmap.N))
    colors = (cmap.colors * ncyc)[:len(categories)]
    colordict = dict(zip(categories, colors))


    # plot the lines, one for each model group, looking up the color by model type from above
    for grp_val in np.unique(frame[grp_col]):
        df = frame.loc[frame[grp_col]==grp_val]
        color = colordict[df.iloc[0][cat_col]]
        df.plot(x_col, y_col, ax=ax, c=color, legend=False)


    # have to set the legend manually since we don't want one legend
    # entry per line on the plot, just one per model type.

    # I had to upgrade matplotlib to get handles working, otherwise
    # had to call like this with plot_labs as a separate list
    # plt.legend(plot_patches, plot_labs, loc=4, fontsize=10)

    plot_lines = []
    # plot_labs = []
    for cat_val in sorted(colordict.keys()):
        # http://matplotlib.org/users/legend_guide.html
        lin = mlines.Line2D([], [], color=colordict[cat_val], label=label_fcn(cat_val))
        plot_lines.append(lin)
        # plot_labs.append(mt)

    plt.legend(handles=plot_lines, loc=legend_loc, fontsize=legend_fontsize)
    ax.set_ylim([0,1.1])
    if x_ticks is not None: ax.set_xticks(x_ticks)
    if y_ticks is not None: ax.set_yticks(y_ticks)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)

    plt.show()



# cut bins in a 0-1 range with nice, sortable names
def cut_bins(df, col, nbins=10):
    """Bin a 0-1 column (such as a score) into uniformly-spaced buckets with nice
       names, such as 'a 0-10'. Bins are inclusive of right edge, exclusive of left.

    Arguments:
        df (DataFrame) -- data frame containing the data to cut
        col (string) -- column name in the data frame
        nbins (int) -- number of bins to cut

    Returns: sequence of named bins
    """
    bins = np.linspace(0,1,nbins+1)
    labs = ['%s %i-%i' % (chr(97+i), bins[i]*100, bins[i+1]*100) for i in range(nbins)]
    bins[0] = -0.01  # ensure we include exact 0's in the bottom bin as pd.cut() is inclusive of the right edge only
    df_bins = pd.cut(df[col], bins, labels=labs)
    df_bins.cat.add_categories(['zz Unknown'], inplace=True)
    df_bins.fillna('zz Unknown', inplace=True)
    return df_bins


# modifies df_dest directly
# assumes categorical var columns are mutually exclusive
def recombine_categorical(df_dest, df_source, prefix, suffix='', entity_col='entity_id'):
    """Combine a categorical variable that has been one-hot encoded into binary columns
       back into a single categorical variable (assumes the naming convention of collate).

       The binary columns here are assumed to be mutually-exclusive and each entity will
       only be given (at most) a single categorical value (in practice, this will be the
       last such value encountered by the for loop).

       Note: modifies the input data frame, df_dest, directly.

    Arguments:
        df_dest (DataFrame) -- data frame into which the recombined categorical variable will be stored
        df_source (DataFrame) -- data frame with the one-hot encoded source columns
        prefix (string) -- prefix shared by the binary columns for the categorical variable, typically
                           something like 'feature_group_entity_id_1y_columnname_'
        suffix (string) -- suffix shared by the binary columns for the categorical variable, typically
                           something like '_min', '_max', '_avg', etc.
        entity_col (string) -- column to identify entities to map the 0/1 binary values in the one-hot
                               columns to catagorical values in the recombined column.

    Returns: tuple of the modified df_dest and the new categorical column name
    """
        cat_col = prefix+suffix+'_CATEGORIES'
        df_dest[cat_col] = np.NaN
        df_dest[cat_col] = df_dest[cat_col].astype('category')

        for col in df_source.columns[df_source.columns.str.startswith(prefix) & df_source.columns.str.endswith(suffix)]:
            cat_val = col.replace(prefix, '').replace(suffix, '')
            df_dest[cat_col].cat.add_categories([cat_val], inplace=True)
            cat_entities = df_source.loc[df_source[col]==1][entity_col]
            df_dest.loc[df_dest[entity_col].isin(cat_entities), cat_col] = cat_val

        return (df_dest, cat_col)



class ModelEvaluation(object):
    """The ModelEvaluation class contains methods for looking at the performance of an individual model:
       ROC curves, precision/recall curves, recall/fpr curves, feature importances, and feature
       distributions across the top vs rest of a list. It also contains a few utility methods, such as
       for fetching train and test matrices from an S3 bucket
    """
    def __init__(self, model_id, models_table='models'):
        self.model_id = model_id

        # Since the full model table may be cluttered with aborted/buggy runs, allow for
        # specifying a different models table to use (e.g. 'models_table_clean') that
        # contains only models associated with runs that should be trusted/considered.
        self.models_table = models_table

        engine = util.create_engine()

        # read the model metadata from the models and model_groups tables
        # note that older models may not have a train matrix uuid available in the db
        sel = """
                SELECT m.model_group_id, m.run_time, m.batch_run_time, m.train_end_time,
                       mg.model_type, mg.hyperparameters, mg.model_config,
                       m.train_matrix_uuid
                FROM results.{} m
                JOIN results.model_groups mg USING(model_group_id)
                WHERE model_id = {}
            """.format(self.models_table, model_id)

        m = pd.read_sql(sel, engine)

        self.model_group_id = m.loc[0, 'model_group_id']

        self.model_type = m.loc[0, 'model_type']
        self.hyperparameters = json.loads(m.loc[0, 'hyperparameters'])
        self.model_config = json.loads(m.loc[0, 'model_config'])

        self.run_time = m.loc[0, 'run_time']
        self.batch_run_time = m.loc[0, 'batch_run_time']
        self.train_end_time = m.loc[0, 'train_end_time']

        self.train_matrix_hash = m.loc[0, 'train_matrix_uuid']

        # grab the prediction matrix uuid from the database
        # note: assumes only one set of predictions for the model and simply
        # grabs the first uuid it finds. some older models may not have their
        # prediction matrix uuid available in the db.
        sel = """
                SELECT matrix_uuid
                FROM results.predictions
                WHERE model_id = {}
                LIMIT 1
            """.format(model_id)

        p = pd.read_sql(sel, engine)

        self.pred_matrix_hash = p.loc[0, 'matrix_uuid']

        engine.dispose()


        # only read these as needed since they may be large
        self.preds = None
        self.labeled_preds = None
        self.features = None

        self.pred_matrix = None
        self.train_matrix = None


    def __str__(self):
        s = "ModelEvaluation Object for model id {}".format(self.model_id)
        s += "\n\nModel group: {}".format(self.model_group_id)
        s += "\nTrain matrix: {}".format(self.train_matrix_hash)
        s += "\nPred. matrix: {}".format(self.pred_matrix_hash)
        return s

    def set_train_matrix_hash(self, train_matrix_hash):
        """Earlier models don't have train and pred hashes available in the database,
           so need to allow for setting them by hand.
        """
        self.train_matrix_hash = train_matrix_hash

    def set_pred_matrix_hash(self, pred_matrix_hash):
        """Earlier models don't have train and pred hashes available in the database,
           so need to allow for setting them by hand.
        """
        self.pred_matrix_hash = pred_matrix_hash

    def _get_preds(self):
        """Loads all of the predictions for this model into the object.

           Note that other methods here will implicitly assume one set of predictions
           per model, so may need modification if that stops being true.
        """
        engine = util.create_engine()
        sel = """select * from results.predictions where model_id = {}""".format(self.model_id)
        self.preds = pd.read_sql(sel, engine)
        self.labeled_preds = self.preds.loc[self.preds['label_value'].notnull()]
        engine.dispose()

    def _get_features(self):
        """Loads the feature importances for this model into the object.
        """
        engine = util.create_engine()
        sel = """select * from results.feature_importances where model_id = {}""".format(self.model_id)
        self.features = pd.read_sql(sel, engine)
        engine.dispose()


    # expects aws cred in ~/.boto or environment vars and database creds in environment vars
    def _fetch_matrix_from_s3(self, matrix_hash,
                             aws_region='us-west-2', bucket_name='dsapp-economic-development',
                             bucket_path='san_jose_housing/matrices/'):
        """Utility function to download a matrix from S3 and load it into a pandas DataFrame.
           The matrix is expected to be in CSV format and contain a header row.

           Note: Requires AWS credentials in ~/.boto or environment variables.

        Arguments:
            matrix_hash (string) -- the hash/uuid for the matrix to load
            aws_region (string) -- the AWS region for the S3 bucket
            bucket_name (string) -- the name of the bucket in S3
            bucket_path (string) -- the location within the bucket to find the matrix

        Returns: DataFrame containing the matrix fetched from S3
        """

        ## Download the matrix from S3 and read into a dataframe

        # url_root = 'http://s3-{}.amazonaws.com/'.format(aws_region)
        file_name = '{}.csv'.format(matrix_hash)
        key_name = bucket_path + file_name

        s3 = boto3.resource('s3', region_name=aws_region)
        s3_obj = s3.Object(bucket_name, key_name)

        # Note: may be memory-intensive for very large matrices
        body = s3_obj.get()['Body'].read()
        body_io = StringIO(body)

        mat = pd.read_csv(body_io)
        del(body, body_io)
        return mat


    # expects aws cred in ~/.boto or environment vars and database creds in environment vars
    def fetch_features_and_pred_matrix(self, entity_col='entity_id', **aws_args):
        """Fetch the prediction matrix from S3, scores and label values from the database,
            and feature importances. The prediction matrix from S3 is merged with the scores
            and labels from the database and the result stored into self.pred_matrix.

        Arguments:
            entity_col (string) -- column name for identifying entities, used to merge matrices
            **aws_args -- arguments passed through to _fetch_matrix_from_s3()
        """

        if self.pred_matrix_hash is None:
            raise ValueError("Prediction matrix hash undefined. Set it with set_pred_matrix_hash()")

        mat = self._fetch_matrix_from_s3(self.pred_matrix_hash, **aws_args)

        # fetch the lables and the feature importances from the database

        if self.features is None:
            self._get_features()

        if self.preds is None:
            self._get_preds()

        # merge the scores onto the feature matrix
        mat = mat.merge(self.preds, on=[entity_col], how='inner', suffixes=('mat', 'pred'))

        self.pred_matrix = mat


    def fetch_train_matrix(self, **aws_args):
        """Fetch the training matrix from S3 and store it in the object.

        Arguments:
            **aws_args -- arguments passed through to _fetch_matrix_from_s3()
        """

        if self.train_matrix_hash is None:
            raise ValueError("Training matrix hash undefined. Set it with set_train_matrix_hash()")

        self.train_matrix = self._fetch_matrix_from_s3(self.train_matrix_hash, **aws_args)


    def plot_feature_importances(self, n_features=30, figsize=(16,12)):
        """Generate a bar chart of the top n feature importances (by absolute value)

        Arguments:
            n_features (int) -- number of top features to plot
            figsize (tuple) -- figure size to pass to matplotlib
        """

        # TODO: allow more of the figure arguments to be passed to the method

        if self.features is None:
            self._get_features()

        humanized_featnames = self.features['feature']
        feature_importances = self.features['feature_importance']

        # TODO: refactor to just make this a slice of self.features
        importances = list(zip(humanized_featnames, list(feature_importances)))
        importances = pd.DataFrame(importances, columns=["Feature", "Importance"])
        importances = importances.set_index("Feature")

        # Sort by the absolute value of the importance of the feature
        importances["sort"] = abs(importances["Importance"])
        importances = importances.sort_values(by="sort", ascending=False).drop("sort", axis=1)
        importances = importances[0:n_features]

        # Show the most important positive feature at the top of the graph
        importances = importances.sort_values(by="Importance", ascending=True)


        fig, ax = plt.subplots(figsize=figsize)
        ax.tick_params(labelsize=16)
        importances.plot(kind="barh", legend=False, ax=ax)
        ax.set_frame_on(False)
        ax.set_xlabel("Importance", fontsize=20)
        ax.set_ylabel("Feature", fontsize=20)
        plt.tight_layout()
        plt.title("Top Feature Importances", fontsize=20).set_position([.5, 0.99])

    def compute_AUC(self):
        """Utility function to generate ROC and AUC data

        Returns: tuple of false positive rate, true positive rate, thresholds, and AUC
        """
        if self.preds is None:
            self._get_preds()

        fpr, tpr, thresholds = metrics.roc_curve(
            self.labeled_preds['label_value'], self.labeled_preds['score'], pos_label=1)

        return (fpr, tpr, thresholds, metrics.auc(fpr, tpr))

    def plot_ROC(self):
        """Plot an ROC curve for this model and label it with AUC
        """

        # TODO: Allow plot formatting arguments to be passed through

        if self.preds is None:
            self._get_preds()

        fpr, tpr, thresholds, auc = self.compute_AUC()
        auc = "%.2f" % auc

        title = 'ROC Curve, AUC = '+str(auc)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, "#000099", label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.show()

    def plot_recall_fpr_n(self):
        """Plot recall and the false positive rate against depth into the list
           (esentially just a deconstructed ROC curve) along with optimal bounds.
        """

        # TODO: Allow plot formatting arguments to be passed through

        if self.preds is None:
            self._get_preds()

        # since tpr and recall are different names for the same metric, we can just
        # grab fpr and tpr from the sklearn function for ROC
        fpr, recall, thresholds, auc = self.compute_AUC()

        # turn the thresholds into percent of the list traversed from top to bottom
        pct_above_per_thresh = []
        num_scored = float(len(self.labeled_preds['score']))
        for value in thresholds:
            num_above_thresh = len(self.labeled_preds.loc[self.labeled_preds['score'] >= value, 'score'])
            pct_above_thresh = num_above_thresh / num_scored
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)

        # plot the false positive rate, along with a dashed line showing the optimal bounds
        # given the proportion of positive labels
        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot([self.labeled_preds['label_value'].mean(), 1], [0, 1], '--', color='gray')
        ax1.plot([0, self.labeled_preds['label_value'].mean()], [0, 0], '--', color='gray')
        ax1.plot(pct_above_per_thresh, fpr, "#000099")
        ax1.set_xlabel('proportion of population')
        ax1.set_ylabel('false positive rate', color="#000099")
        plt.ylim([0.0, 1.05])

        # plot the recall curve, along with a dashed line showing the optimal bounds
        # given the proportion of positive labels
        ax2 = ax1.twinx()
        ax2.plot([0, self.labeled_preds['label_value'].mean()], [0, 1], '--', color='gray')
        ax2.plot([self.labeled_preds['label_value'].mean(), 1], [1, 1], '--', color='gray')
        ax2.plot(pct_above_per_thresh, recall, "#CC0000")
        ax2.set_ylabel('recall', color="#CC0000")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.title("fpr-recall at x-proportion")
        plt.show()

    def plot_precision_recall_n(self):
        """Plot recall and precision curves against depth into the list.
        """

        # TODO: Allow plot formatting arguments to be passed through

        if self.preds is None:
            self._get_preds()

        y_score = self.labeled_preds['score']
        precision_curve, recall_curve, pr_thresholds = \
            metrics.precision_recall_curve(self.labeled_preds['label_value'], y_score)

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
        fig, ax1 = plt.subplots()
        ax1.plot(pct_above_per_thresh, precision_curve, "#000099")
        ax1.set_xlabel('proportion of population')
        ax1.set_ylabel('precision', color="#000099")
        plt.ylim([0.0, 1.05])
        ax2 = ax1.twinx()
        ax2.plot(pct_above_per_thresh, recall_curve, "#CC0000")
        ax2.set_ylabel('recall', color="#CC0000")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title("Precision-recall at x-proportion")
        plt.show()

    def categorical_plot_and_stats(self, frame, cat_col, grp_col,
                                  y_axis_pct=False, figsize=[9,5],
                                  figtitle=None, xticks_map=None,
                                  figfontsize=14, figtitlesize=16):
        """Plot the distribution of categorical variable across a grouping variable,
           for instance, to compare the distribution of a feature across entities
           above and below a score threshold. Additionally runs a chi-squared test
           for independence of the distribution across the groups.

        Arguments:
            frame (DataFrame) -- the data frame containing the data to plot
            cat_col (string) -- name of the column with the categorical variable
            grp_col (string) -- name of the column with the grouping variable (e.g., above vs below threshold)
            y_axis_pct (bool) -- label the y-axis as a percent (as opposed to as a fraction)
            figsize (tuple) -- figure size to pass to matplotlib
            figtitle (string) -- optional title to pass to the plot
            xticks_map (dict) -- option mapping for more readable labels for the x-ticks
            figfontsize (int) -- font size to use for the figure
            figtitlesize (int) -- font size to use for the title
        """

        if figtitle is None:
            figtitle = cat_col

        # generate counts of the categorical column by the grouping column, unstacking and filling
        # NA's with 0's to generate a count table for the plot
        grouped = frame.groupby([grp_col, cat_col])[cat_col].count().unstack(grp_col).fillna(0)

        # calculate and plot frequencies
        grouped_counts = grouped.copy()  # preserve a copy with counts for chi2 test below
        for grp_val in grouped.columns:
            grouped[grp_val] = grouped[grp_val]/grouped[grp_val].sum()
        # sort columns to ensure consistent coloring in plot
        grouped.sort_index(axis=1, inplace=True)
        colors = ['#E59141', '#557BA5', 'royalblue', 'darkorchid', 'peru', 'red', 'yellow', 'green', 'darkblue']
        colors = colors[:grouped.shape[1]]
        ax = grouped.plot(kind='bar', stacked=False, width=0.9,
                     color=colors, alpha=1.00,
                     figsize=figsize, fontsize=figfontsize, rot=0)
        ax.set_xlabel('')
        plt.title(figtitle+'\n', fontsize=figtitlesize, fontweight='bold')

        # move the legend outside of the plot, under the title
        box = ax.get_position()
        ax.set_position([box.x0, box.y0,
                 box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=figfontsize).set_title(None)

        # optionally express the y-axis as a percentage
        if y_axis_pct:
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])

        # optionally replace the labels of the x-axis ticks with someting more readable
        if xticks_map is not None:
            vals = ax.get_xticks()
            ax.set_xticklabels([xticks_map.get(x, 'Ruh-Roh!') for x in vals])

        plt.show()


        # perform a chi-squared test if we only have 2 groups (e.g. above vs below threshold or by label)
        if grouped.shape[1]==2:
            # chi-squared test for independence above vs below threshold
            chi2, p_val, _, _ = scs.chi2_contingency(grouped_counts)

            print('Chi-Squared Test for Independence')
            print('Chi-Squared: {}'.format(chi2))
            print('p-value:     {}'.format(p_val))

    def continuous_plot_and_stats(self, frame, cont_col, grp_col, nbins=None,
                                  y_axis_pct=False, figsize=[9,5], figtitle=None,
                                  figfontsize=14, figtitlesize=16):
        """Plot the distribution of a continuous variable across a grouping variable,
           for instance, to compare the distribution of a feature across entities
           above and below a score threshold. Additionally, prints summary statistics
           for each group and, if there are only 2 groups, runs a t-test for independence
           of the distribution across the groups.

        Arguments:
            frame (DataFrame) -- the data frame containing the data to plot
            cont_col (string) -- name of the column with the continuous variable
            grp_col (string) -- name of the column with the grouping variable (e.g., above vs below threshold)
            nbins (int) -- optionally specify the number of bins to use for the histograms
            y_axis_pct (bool) -- label the y-axis as a percent (as opposed to as a fraction)
            figsize (tuple) -- figure size to pass to matplotlib
            figtitle (string) -- optional title to pass to the plot
            figfontsize (int) -- font size to use for the figure
            figtitlesize (int) -- font size to use for the title
        """

        if figtitle is None:
            figtitle = cont_col

        # unique group values (np.unique() returns a sorted array)
        grp_vals = np.unique(frame[grp_col])

        # histograms
        fig, ax = plt.subplots(figsize=figsize)
        # use numpy to calculate a reasonable set of bins across the entire range
        # to ensure plots are comparable (using 1/10 the smallest group bins if
        # not specified in the arguments)
        if nbins is None:
            nbins = frame.groupby(grp_col)[grp_col].count().min()/10
        bins_array = np.histogram(frame[cont_col], bins=nbins)[1]

        colors = ['#E59141', '#557BA5', 'royalblue', 'darkorchid', 'peru', 'red', 'yellow', 'green', 'darkblue']
        for i, grp_val in enumerate(grp_vals):
            ax.hist(frame[frame[grp_col]==grp_val][cont_col],
                    # these weights will make the histogram bars sum to 1 (default is an expected value of the x-variable of 1)
                    weights=1.0*np.ones_like(frame[frame[grp_col]==grp_val][cont_col])/len(frame[frame[grp_col]==grp_val]),
                    bins=bins_array,
                    color=colors[i % len(colors)], alpha=0.75,
                    label=str(grp_val),
                    fontsize=figfontsize
                    )
        plt.title(figtitle+'\n', fontsize=figtitlesize, fontweight='bold')

        # move the legend outside of the plot, under the title
        box = ax.get_position()
        ax.set_position([box.x0, box.y0,
                 box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=figfontsize).set_title(None)

        # optionally express the y-axis as a percentage
        if y_axis_pct:
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
        plt.show()

        # mean, p10, p25, median, p75, p90 for each
        print(frame.groupby(grp_col)[cont_col]\
                  .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])\
                  .unstack(grp_col)\
                  .sort_index(axis=1)
             )

        # perform a t-test if we only have 2 groups (e.g. above vs below threshold or by label)
        if len(grp_vals)==2:

            # mean diff, t-stat, p-value
            # could also consider test for difference in distributions?
            t_stat, p_val = scs.ttest_ind(
                frame[frame[grp_col]==grp_vals[0]][cont_col],
                frame[frame[grp_col]==grp_vals[1]][cont_col]
            )
            mean_diff = frame[frame[grp_col]==grp_vals[0]][cont_col].mean() -\
                        frame[frame[grp_col]==grp_vals[1]][cont_col].mean()

            print('\nT-Test for Difference in Means')
            print('Mean Difference: {}'.format(mean_diff))
            print('t-statistic:     {}'.format(t_stat))
            print('p-value:         {}'.format(p_val))


    # TODO: allow the cut params to optionally be either a single cut-point or
    #       a (top, bottom) tuple to ignore entities in the middle
    def test_feature_diffs(self, feature_type, name_or_prefix, suffix='',
                      score_col='score', entity_col='entity_id',
                      cut_n=None, cut_frac=None, cut_score=None,
                      nbins=None, thresh_labels=['Above Threshold', 'Below Threshold'],
                      figsize=[9,5], figtitle=None, xticks_map=None,
                      figfontsize=14, figtitlesize=16,
                      **aws_args
                     ):
        """
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
            **aws_args:     arguments to pass through to fetch_features_and_pred_matrix()
        """

        if self.pred_matrix is None:
            self.fetch_features_and_pred_matrix(entity_col, **aws_args)

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





class ModelGroupEvaluation(object):
    """The ModelGroupEvaluation class contains methods for looking at the performance of a
       single model group, including stack ranking plots an score histograms pooled across
       the models in the group.
    """
    def __init__(self, model_group_id, models_table='models'):
        self.model_group_id = model_group_id

        # Since the full model table may be cluttered with aborted/buggy runs, allow for
        # specifying a different models table to use (e.g. 'models_table_clean') that
        # contains only models associated with runs that should be trusted/considered.
        self.models_table = models_table

        # grab model group metadata from the database
        sel = """
                SELECT model_type, hyperparameters, model_config
                FROM results.model_groups
                WHERE model_group_id = {}
            """.format(model_group_id)

        engine = util.create_engine()

        mg = pd.read_sql(sel, engine)

        self.model_type = mg.loc[0, 'model_type']
        self.hyperparameters = json.loads(mg.loc[0, 'hyperparameters'])
        self.model_config = json.loads(mg.loc[0, 'model_config'])

        self.models = self._get_models(engine)

        # only pull these down as needed since they may be large
        self.preds = None

        engine.dispose()

    def _get_models(self, engine):
        """Fetch the information associated with all of the models in this model group
           from the database into a data frame.

        Arguments:
            engine -- the engine for the database connection

        Returns: data frame of the model info
        """
        sel = """
            SELECT *
            FROM results.{}
            WHERE model_group_id = {}
            ;
            """.format(self.models_table, self.model_group_id)

        m = pd.read_sql(sel, engine)

        return m

    def __str__(self):
        return "ModelGroupEvaluation object for model group id {}".format(self.model_group_id)

    def most_recent_model(self):
        """Fetch the information associated with the most recent model in this group.

        Returns: dataframe containing the record for the most recent model
        """
        return self.models.sort_values('train_end_time', ascending=False).head(1)

    def _get_all_preds(self):
        """Fetch all of the predictions for entities with labels associated with
           models in this group and store them in the object, along with some
           ntile cuts and bins on the scores.
        """

        engine = util.create_engine()

        # Note: only fetch predictions associated with labels since these data will be used
        #       for validation metrics.
        sel = """
                SELECT model_id, entity_id, score, label_value
                FROM results.predictions
                JOIN results.{} USING(model_id)
                WHERE model_group_id = {}
                      AND label_value IS NOT NULL
            """.format(self.models_table, self.model_group_id)

        self.preds = pd.read_sql(sel, engine)
        self.preds['score_bins'] = cut_bins(self.preds, 'score')
        self.preds['score_vigintile'] = pd.qcut(self.preds['score'], 20, labels=np.arange(1,21))
        self.preds['score_decile'] = pd.qcut(self.preds['score'], 10, labels=np.arange(1,11))
        self.preds['score_quintile'] = pd.qcut(self.preds['score'], 5, labels=np.arange(1,6))

        engine.dispose()

    # plot_type can be ['bins', 'decile', 'vigintile', 'quintile']
    def stack_rank_plot(self, plot_type='decile', y_axis_pct=False, figsize=[9,5]):
        """Generate a stack-ranking plot to compare predicted and observed outcome rates across
           the score distribution.

        Arguments:
            plot_type (string) -- how to bin the score in the test set for plotting,
                                  may be ['bins', 'decile', 'vigintile', 'quintile']
            y_axis_pct (bool) -- label the y-axis as a percent (as opposed to as a fraction)
            figsize (tuple) -- figure size to pass to matplotlib
        """

        # TODO: Allow more formatiing options to be passed in directly and make the titles and
        #       labels more flexible for other model types.

        if plot_type not in set(['bins', 'decile', 'vigintile', 'quintile']):
            raise ValueError("Plot type must be one of: ['bins', 'decile', 'vigintile', 'quintile']")

        if self.preds is None:
            self._get_all_preds()

        col = 'score_'+plot_type

        aggs = {
                'score': {'avg_score' : 'mean'},
                'label_value' : {'violation_rate' : 'mean'}
            }
        grp = self.preds.groupby(col).aggregate(aggs)
        grp.columns = grp.columns.droplevel()
        # prev colors: ['royalblue', 'peru']
        ax = grp.plot(kind='bar', ylim=[0,1.05], figsize=figsize, color=['#557BA5', '#E59141'], fontsize=14, rot=0)
        ax.set_title('Predicted vs Observed Violation Rates', fontsize=16)
        ax.set_xlabel('Score {}'.format(plot_type.title()), fontsize=14)
        ax.set_ylabel('Violation Rate', fontsize=14)
        plt.legend(fontsize=14, labels=['Predicted Rate', 'Actual Rate'])
        if y_axis_pct:
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
        plt.show()

    def score_hist(self):
        """Plot a histogram of the scores for labeled examples across all models associated
           with this model group.
        """

        # TODO: allow more plot formatting options to be passed through

        if self.preds is None:
            self._get_all_preds()

        plt.figure(figsize=(9,5))
        plt.hist(self.preds['score'],
                weights=np.ones_like(self.preds['score'])/len(self.preds),
                bins=50,
                color='royalblue', alpha=0.75
                )
        plt.title('Histogram of Scores')




class ExperimentEvaluation(object):
    """The ExperimentEvaluation class contains methods for looking at metrics accross model
       groups, including Jaccard Similarity of top N lists, plotting evaluation metrics over
       time by model type or hyperparameters, and identifying model configurations that are
       consistently performing well.
    """

    def __init__(self, model_groups, min_batch_date,
                 default_metric=None, default_metric_param=None,
                 models_table='models'):
        self.model_groups = model_groups
        self.model_groups_str = ', '.join(map(str, model_groups))
        self.min_batch_date = min_batch_date

        # Since the full model table may be cluttered with aborted/buggy runs, allow for
        # specifying a different models table to use (e.g. 'models_table_clean') that
        # contains only models associated with runs that should be trusted/considered.
        self.models_table = models_table

        # by default, the experiment evaluation will focus on precion@300, but this can be over-ridden
        # either when instantiating the object, by calling set_default_metric(), or manually when
        # calling one of the plotting methods.
        if default_metric is None:
            self.default_metric = 'precision@'
        else:
            self.default_metric = default_metric

        if default_metric_param is None:
            self.default_metric_param = '300_abs'
        else:
            self.default_metric_param = default_metric_param

        # may be large, so will defer fetching these until needed
        self.df_bestdist = None
        self.bestdist_params = {}

    def __str__(self):
        return "ExperimentEvaluation Object:\n\nModel Groups: {model_groups}\n\nMin Batch Date: {min_batch_date}".format(model_groups=self.model_groups_str, min_batch_date=self.min_batch_date)

    def set_default_metric(default_metric, default_metric_param):
        """Allow for setting the default metric and parameter after the object has been instantiated.

        Arguments:
            default_metric (string) -- the metric to use for plots/analyses by default (e.g., 'precision@')
            default_metric_param (string) -- the metric parameter for plots/analyses by default (e.g., '300_abs')
        """
        self.default_metric = default_metric
        self.default_metric_param = default_metric_param

    def _check_best_dist(self, metric, metric_param, model_type, hyperparam):
        """Helper function to test whether the best distance dataframe associated with the object
           matches certain parameters. The object caches the best distance information mmost
           recently used to allow for faster additional analysis, so we need to keep track of
           the metadata associated with that data frame to ensure we only re-create it when
           necessary.

        Arguments:
            metric (string) -- model evaluation metric, such as 'precision@'
            metric_param (string) -- model evaluation metric parameter, such as '300_abs'
            model_type (string) -- model type, such as sklearn.ensemble.RandomForestClassifier (may be None)
            hyperparam (string) -- model hyperparameter, such as max_depth (may be None)

        Returns: boolean to indicate whether the current df_bestdist has the input parameters
        """
        check_dict = {
                'metric': metric,
                'metric_param': metric_param,
                'model_type': model_type,
                'hyperparam': hyperparam
        }
        return (self.df_bestdist is not None and self.bestdist_params == check_dict)

    def _hyperparam_sql(self, hyperparam):
        """Helper function to generate SQL snippet for pulling different types of model configuration
           and hyperparameter information out of the database.

        Arguments:
            hyperparam (string) -- name of the hyperparameter to query

        Returns: string SQL snippet for querying hyperparameter
        """

        if hyperparam is None:
            hyperparam_sql = 'NULL::VARCHAR(64) AS hyperparam,'
        elif hyperparam == 'feature_hash':
            hyperparam_sql = "md5(mg.feature_list::VARCHAR) AS hyperparam,"
        else:
            hyperparam_sql = "COALESCE(mg.hyperparameters->>'{}', mg.model_config->>'{}') AS hyperparam,".format(hyperparam, hyperparam)

        return hyperparam_sql

    def _get_best_dist(self, metric, metric_param, model_type, hyperparam):
        """Fetch a best distance data frame from the database. This may be a relatively intensive
           query and result in a large amount of data depending on the number of model groups
           being considered as well as the time range and modeling frequency. As a result, we
           only update this data frame when necessary and simply return if the best distance
           data associated with a set of parameters has already been read in. See the comments
           on plot_best_dist() for more information.

        Arguments:
            metric (string) -- model evaluation metric, such as 'precision@'
            metric_param (string) -- model evaluation metric parameter, such as '300_abs'
            model_type (string) -- model type, such as sklearn.ensemble.RandomForestClassifier (may be None)
            hyperparam (string) -- model hyperparameter, such as max_depth (may be None)
        """

        # TODO: Allow for parameters not measured on a 0-1 scale?

        # only hold onto one best dist dataframe at a time to keep memory from getting
        # unreasonable, so overwrite any existing one

        # if we've already read this, skip talking to the db
        if self._check_best_dist(metric, metric_param, model_type, hyperparam):
            return

        engine = util.create_engine()

        # optionally filter down to a certain type of model within the experiment, for instance,
        # to look at random forest-specific hyperparameters.
        if model_type is None:
            model_type_sql = ''
        else:
            model_type_sql = "AND m.model_type='{}'".format(model_type)

        # grab sql snippet for querying hyperparameter from model_groups data
        hyperparam_sql = self._hyperparam_sql(hyperparam)

        sel_params = {
            'metric': metric,
            'metric_param': metric_param,
            'batch_run_dt': self.min_batch_date,
            'model_groups': self.model_groups_str,
            'hyperparam_sql': hyperparam_sql,
            'model_type_sql': model_type_sql,
            'models_table': self.models_table
        }

        # TODO: allow for ranking based on ASC parameters, too, such as error rates, etc.

        # query to generate for each model group a series of data at 1pp intervals of the cumulative
        # percent of the time that the model from that model group is within X percentage points of
        # the best model on the basis of the metric & metric parameter being considered.
        #   * model_ranks simply calculates the ranking of the model groups at each train_end_time by the metric
        #   * model_tols finds the best best value of the metric by train end time
        #   * x_vals provides a series from 0 to 1 by 0.01 associated with each model_group_id
        #   * the final select rolls up the raction of models in the group that are within x pp of the best value across train_end_times
        sel = """
                WITH model_ranks AS (
                  SELECT m.model_group_id, m.model_id, m.model_type, m.train_end_time, ev.value,
                         {hyperparam_sql}
                         row_number() OVER (PARTITION BY m.train_end_time ORDER BY ev.value DESC, RANDOM()) AS rank
                  FROM results.evaluations ev
                  JOIN results.{models_table} m USING(model_id)
                  JOIN results.model_groups mg USING(model_group_id)
                  WHERE m.batch_run_time >= '{batch_run_dt}' AND ev.metric='{metric}' AND ev.parameter='{metric_param}'
                        AND m.model_group_id IN ({model_groups})
                        {model_type_sql}
                ),
                model_tols AS (
                  SELECT train_end_time, model_group_id, model_id, model_type, hyperparam,
                         rank,
                         value,
                         first_value(value) over (partition by train_end_time order by rank ASC) AS best_val
                  FROM model_ranks
                ),
                x_vals AS (
                  SELECT m.model_group_id, s.pct_diff
                  FROM
                  (
                  SELECT GENERATE_SERIES(0,100) / 100.0 AS pct_diff
                  ) s
                  CROSS JOIN
                  (
                  SELECT DISTINCT model_group_id FROM results.{models_table}
                  ) m
                )
                SELECT model_group_id, model_type, hyperparam, pct_diff,
                       COUNT(*) AS num_models,
                       AVG(CASE WHEN best_val - value <= pct_diff THEN 1 ELSE 0 END) AS pct_of_time
                FROM model_tols
                JOIN x_vals USING(model_group_id)
                GROUP BY 1,2,3,4
                ORDER BY 1,2,3,4
                ;
            """.format(**sel_params)

        self.df_bestdist = pd.read_sql(sel, engine)
        self.bestdist_params = {
                'metric': metric,
                'metric_param': metric_param,
                'model_type': model_type,
                'hyperparam': hyperparam
            }

        engine.dispose()


    def plot_best_dist(self, metric=None, metric_param=None,
                      model_type=None, hyperparam=None,
                      **plt_format_args):
        """Generates a plot of the percentage of time that a model group is within X percentage points
           of the best-performing model group using a given metric. At each point in time that a set of
           model groups is evaluated, the performance of the best model is calculated and the difference
           in performace for all other models found relative to this. An (x,y) point for a given model
           group on the plot generated by this method means that across all of those tets sets, the model
           from that model group performed within X percentage points of the best model in y% of the test
           sets.

           The plot will contain a line for each model group in the ExperimentEvaluation object
           representing the cumulative percent of the time that the group is within Xpp of the best group
           for each value of X between 0 and 100. All groups ultimately reach (1,1) on this graph (as every
           model group must be within 100pp of the best model 100% of the time), and a model specification
           that always dominated the others in the experiment would start at (0,1) and remain at y=1
           across the graph.

        Arguments:
            metric (string) -- model evaluation metric, such as 'precision@'; if not specified the object's
                               default_metric will be used.
            metric_param (string) -- model evaluation metric parameter, such as '300_abs'; if not specified
                                     the object's default_metric_param will be used.
            model_type (string) -- optional model type, such as sklearn.ensemble.RandomForestClassifier, to
                                   subset the model groups considered, for instance if wanting to look at
                                   the relative performance of random forest-specific hyperparamteres
            hyperparam (string) -- optional model hyperparameter, such as max_depth, to use to color the
                                   lines in the plot; if not specified, model_type will be used instead.
            **plt_format_args -- formatting arguments passed through to plot_cats()
        """

        if metric is None:
            metric = self.default_metric

        if metric_param is None:
            metric_param = self.default_metric_param

        # get the best dist data frame associated with these parameters if it hasn't been read in already
        self._get_best_dist(metric, metric_param, model_type, hyperparam)

        cat_col = 'model_type' if hyperparam is None else 'hyperparam'
        plt_title = 'Fraction of models X pp worse than best {} {}'.format(metric, metric_param)
        if hyperparam is not None:
            plt_title += '\n{} by {}'.format(model_type, hyperparam)

        plot_cats(self.df_bestdist, 'pct_diff', 'pct_of_time', cat_col=cat_col,
          title=plt_title,
          x_label='decrease in {} from best model'.format(metric),
          y_label='fraction of models',
          x_ticks=np.arange(0,1.1,0.1),
          **plt_format_args)


    def plot_metric_over_time(self, metric=None, metric_param=None, min_support=0,
                              hyperparam=None, **plt_format_args):
        """Generate a time-series plot for a given metric, optionally restricting only to observations
           with a certain amount of support (e.g., number of labeled examples above the threshold) and
           coloring model groups either by type of model (the default) or a specified hyperparameter.

        Arguments:
            metric (string) -- model evaluation metric, such as 'precision@'; if not specified the object's
                               default_metric will be used.
            metric_param (string) -- model evaluation metric parameter, such as '300_abs'; if not specified
                                     the object's default_metric_param will be used.
            min_support (int) -- minimum number of labeled examples above the threshold to include an
                                 observation in the dataset.
            hyperparam (string) -- optional model hyperparameter, such as max_depth, to use to color the
                                   lines in the plot; if not specified, model_type will be used instead.
            **plt_format_args -- formatting arguments passed through to plot_cats()
        """


        # TODO: allow for subsetting by a model type

        if metric is None:
            metric = self.default_metric

        if metric_param is None:
            metric_param = self.default_metric_param

        engine = util.create_engine()

        # Can use this to plot a metric over time with a hyperparameter, but not that might
        # not have meaningful results if the hyperparameter isn't shared by all model types
        # in the experiment...
        hyperparam_sql = self._hyperparam_sql(hyperparam)
        cat_col = 'model_type' if hyperparam is None else 'hyperparam'

        sel_params = {
            'metric': metric,
            'metric_param': metric_param,
            'hyperparam_sql': hyperparam_sql,
            'batch_run_dt': self.min_batch_date,
            'model_groups': self.model_groups_str,
            'min_support': min_support,
            'models_table': self.models_table
        }


        # Read the evaluation data in from the database. Note that this query implicitly assumes
        # only one evaluation per model group & evaluation_start_time and would need to be
        # edited to allow for selecting a certain evaluation if this were not the case.
        sel = """
                  SELECT m.model_group_id, m.model_id, m.model_type, {hyperparam_sql}
                         ev.evaluation_start_time::DATE, ev.value, ev.num_labeled_above_threshold
                  FROM results.evaluations ev
                  JOIN results.{models_table} m USING(model_id)
                  JOIN results.model_groups mg USING(model_group_id)
                  WHERE m.batch_run_time >= '{batch_run_dt}' AND metric='{metric}' AND parameter='{metric_param}'
                        AND m.model_group_id IN ({model_groups})
                        AND ev.num_labeled_above_threshold >= {min_support}
                  ;
                """.format(**sel_params)

        df_ts = pd.read_sql(sel, engine)

        plot_cats(df_ts, 'evaluation_start_time', 'value', cat_col=cat_col,
          title='Model Group {} {} Over Time'.format(metric, metric_param),
          x_label='Evaluation Start Time',
          y_label='{} {}'.format(metric, metric_param),
          **plt_format_args)


        engine.dispose()
        del(df_ts)


    def best_model_groups(self, bestdist_cut, n_groups=10, info=False):
        """Return the top model groups from the best distance data frame looking at a certain slice
           of that data set (in terms of the percentage point difference from the best group).

           Note that this method currently only returns the best model groups for the bestdist data
           frame currently stored in the ExperimentEvaluation object, which may vary based on the
           parameters that were last used for plot_best_dist(). For the moment, adding a print
           statement to alert the user of the current parameters being used, but we should refactor
           to be more explicit about controlling the output here.

        Arguments:
            bestdist_cut (float) -- the slice of the best distance data set (in terms of percentage point
                                    difference from the best) to rank model groups based on.
            n_groups (int) -- number of model groups to return
            info (bool) -- when False, returns a list of model_group_ids, when True, returns the
                           information in the best distance data frame associated with these groups as well

        Returns: A list of model_group_ids if info=False; a data frame of model groups and best distance
                 information if info=True
        """

        # TODO: be more explicit in returning this data about the current best distance parameters!

        # if no best distance data has yet been read, get the data associated with the default parameters
        if self.df_bestdist is None:
            self._get_best_dist(self.default_metric, self.default_metric_param, None, None)

        print("Returning best distance model groups for parameters: {}".format(str(self.bestdist_params)))

        # look at the desired slice of the data set and rank model groups by percent of time they're that
        # distance from the best group
        cut_sorted = self.df_bestdist[self.df_bestdist['pct_diff']==bestdist_cut].sort_values('pct_of_time', ascending=False)

        if info:
            return cut_sorted.head(n_groups)
        else:
            return list(cut_sorted['model_group_id'].head(n_groups))

    def plot_jaccard(self, train_end_date, top_n,
                      sim_model_groups=None, bestdist_cut=None, n_groups=10,
                      metric=None, metric_param=None,
                      model_type=None, hyperparam=None):
        """Plot a heatmap of the Jaccard Similarity between the top n entities for each model with a given
           train_end_date associated with either a specified set of model groups or the model groups
           identified from the best distance criteria.

           This method doesn't actually assume a single prediction per model and entity (as is done
           elsewhere), but rather assumes a single prediction per (model_id, entity_id, as_of_date)
           tuple and picks the one with the first as_of_date.

        Arguments:
            train_end_date (string) -- consider the model in each model group associated with this date
            top_n (int) -- the number of entities, ranked by their scores, to classify as targets from
                           from each model for the calculating the Jaccard similarity
            sim_model_groups (list) -- a list of model groups to calculate the similarity across. Must
                                       specify either this or bestdist_cut and n_groups.
            bestdist_cut (float) -- the slice of the best distance data set (in terms of percentage point
                                    difference from the best) to rank model groups based on. Must specify
                                    either this or sim_model_groups.
            n_groups (int) -- if using bestdist_cut, the number of model groups to return.
            metric (string) -- model evaluation metric, such as 'precision@'; if not specified the object's
                               default_metric will be used.
            metric_param (string) -- model evaluation metric parameter, such as '300_abs'; if not specified
                                     the object's default_metric_param will be used.
            model_type (string) -- optional model type, such as sklearn.ensemble.RandomForestClassifier, to
                                   subset the model groups considered if using bestdist_cut.
            hyperparam (string) -- optional model hyperparameter, such as max_depth, if using bestdist_cut
                                   (note that the hyperparam passed won't affect the result of the jaccard
                                   plot, but is used to determine if the best distance data has already been
                                   downloaded).
        """

        # TODO: allow for passing plot formatting parameters

        if (sim_model_groups is None) and (bestdist_cut is None):
            raise ValueError("Must specify either sim_model_groups or bestdist_cut")

        if metric is None:
            metric = self.default_metric

        if metric_param is None:
            metric_param = self.default_metric_param

        engine = util.create_engine()


        # if using bestdist_cut to get the model groups, read the appropriate best distance
        # data if necessary and determine the groups
        if bestdist_cut is not None:
            self._get_best_dist(metric, metric_param, model_type, hyperparam)
            sim_model_groups = self.best_model_groups(bestdist_cut, n_groups, False)

        sim_params = {
            'sim_model_groups' : ', '.join(map(str, sim_model_groups)),
            'train_end_dt' : train_end_date,
            'models_table' : self.models_table
        }

        # Select the models, predictions, and rankings from the database
        sim_sel = """
            WITH mods AS (
              SELECT model_group_id, model_id, run_time, train_end_time,
                     row_number() OVER (PARTITION BY model_group_id ORDER BY train_end_time DESC, run_time DESC) AS rn_mod
              FROM results.{models_table}
              WHERE model_group_id IN ({sim_model_groups}) AND train_end_time::DATE = '{train_end_dt}'
            ),
            preds AS (
              SELECT m.model_group_id, m.model_id, p.entity_id, p.score,
                     row_number() OVER (PARTITION BY m.model_id, p.entity_id ORDER BY as_of_date ASC) AS rn_ent
              FROM mods m
              JOIN results.predictions p USING(model_id)
              WHERE rn_mod=1
            )
            SELECT model_group_id, model_id, entity_id, score,
                   row_number() OVER (PARTITION BY model_id ORDER BY score DESC, RANDOM()) AS score_rank
            FROM preds
            ;
        """.format(**sim_params)

        # read the data from the database and label the top n for each model
        df_sim = pd.read_sql(sim_sel, engine)
        df_sim['above_thresh'] = (df_sim['score_rank'] <= top_n).astype(int)
        df_sim_piv = df_sim.pivot(index='entity_id', columns='model_group_id', values='above_thresh')

        # calculate jaccard similarity between the models
        res = pdist(df_sim_piv[sim_model_groups].T, 'jaccard')
        df_jac = pd.DataFrame(1-squareform(res), index=sim_model_groups, columns=sim_model_groups)

        # plot the similarity matrix
        plt.figure(figsize=(11,9))
        sns.heatmap(df_jac, cmap='Greens', vmin=0, vmax=1, annot=True, linewidth=0.1)

        engine.dispose()
