"""
Model Evaluator

This module will use the model_id as a modeling unit of analysis to answer
different type of questions related with the quality and behavior of my
predictions. The ModelEvaluator (and by extension de ModelGroupEvaluator) are
part of the triage framework, hence they depend on the results schema
structure, and the rest of the triage elements.

ModelEvaluator will use a tuple: (model_group_id, model_id) to run diffent
functions that will help the triage user to explore the final models, ideally
selected by the Audition module. To initiate this classes, the user need to
have a tuple and a SQLAlchemy engine object to call the necesary data from the
SQL database.
"""

import pandas as pd
import numpy as np
import pickle
import graphviz
import matplotlib.patches as mpatches
import seaborn as sns
from adjustText import adjust_text
from functools import partial
from matplotlib import pyplot as plt
from descriptors import cachedproperty
from sklearn import metrics
from sklearn import tree
from triage.component.catwalk.storage import ProjectStorage, ModelStorageEngine, MatrixStorageEngine


class ModelEvaluator:
    '''
    ModelExtractor class calls the model metadata from the database
    and hold model_id metadata features on each of the class attibutes.
    This class will contain any information about the model, and will be
    used to make comparisons and calculations across models.

    A pair of (model_group_id, model_id) is needed to instate the class. These
    can be feeded from the get models_ids.
    '''
    def __init__(self, model_group_id, model_id, engine):
        self.engine = engine
        self.model_id = model_id
        self.model_group_id = model_group_id

    @cachedproperty
    def metadata(self):
        return next(self.engine.execute(
                    f'''WITH
                    individual_model_ids_metadata AS(
                    SELECT m.model_id,
                           m.model_group_id,
                           m.hyperparameters,
                           m.model_hash,
                           m.train_end_time,
                           m.train_matrix_uuid,
                           m.training_label_timespan,
                           m.model_type,
                           mg.model_config
                        FROM triage_metadata.models m
                        JOIN triage_metadata.model_groups mg
                        USING (model_group_id)
                        WHERE model_group_id = {self.model_group_id}
                        AND model_id = {self.model_id}
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
                    USING(model_id);''')
        )

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
    def pred_matrix_uuid(self):
        return self.metadata['matrix_uuid']

    @property
    def as_of_date(self):
        return self.metadata['as_of_date']

    def __repr__(self):
        return f"ModelEvaluator(model_group_id={self.model_group_id}, model_id={self.model_id})"

    @cachedproperty
    def predictions(self):
        preds = pd.read_sql(
            f'''
            SELECT model_id,
                   entity_id,
                   as_of_date,
                   score,
                   label_value,
                   COALESCE(rank_abs_with_ties, RANK() OVER(ORDER BY score DESC)) AS rank_abs,
                   COALESCE(rank_pct_with_ties, percent_rank()
                   OVER(ORDER BY score DESC)) * 100 as rank_pct,
                   test_label_timespan
            FROM test_results.predictions
            WHERE model_id = {self.model_id}
            AND label_value IS NOT NULL
            ''', con=self.engine)

        if preds.empty:
            raise RuntimeError("No predictions were retrieved from the database."
                               "Some functionality will not be available without predictions."
                               "Please run catwalk.Predictor for each desired model and test matrix"
                               )
        return preds

    def _feature_importance_slr(self, path):
        '''
        Calculate feature importances for ScaledLogisticRegression

        Since triage do not calculate the feature importances of
        ScaledLogisticRegression by default, this function will call the model
        object and calculate a feature importance proxy using the coefficients
        of the regression.

        Arguments:
            - path (str): triage's project path, or a path to find the model
            objects

        Return:
            - Dataframe with feature_importances
        '''

        storage = ProjectStorage(path)
        model_obj = ModelStorageEngine(storage).load(self.model_hash)

        test_matrix = self.preds_matrix(path)
        feature_names = [x for x in test_matrix.column.tolist() 
                         if x not in self.predictions.column.tolist()]

        raw_importances = pd.DataFrame(
            {'feature': feature_names,
             'model_id':  test_matrix['model_id'],
             'feature_importance':  np.abs(1 - np.exp(model_obj.coef_.squeeze())),
             'feature_group': [x.split('_entity_id')[0] for x in
                               feature_names]
            }
        )
        raw_importances['rank_abs'] = raw_importances['feature_importance'].\
                                      rank(method='max')

        return raw_importances


    def feature_importances(self, path=None):
        if "ScaledLogisticRegression" in self.model_type:
            features = self._feature_importance_slr(path)
        else:
            features = pd.read_sql(
                f'''
                SELECT model_id,
                       feature,
                       feature_importance,
                       CASE
                       WHEN feature like 'Algorithm does not support a standard way to calculate feature importance.'
		    	       THEN 'No feature group'
		               ELSE split_part(feature, '_', 1)
		    	       END AS feature_group,
                       rank_abs
                FROM train_results.feature_importances
                WHERE model_id = {self.model_id}
                ''', con=self.engine)
        return features


    def feature_group_importances(self, path=None):
        if "ScaledLogisticRegression" in self.model_type:
            raw_importances = self._feature_importance_slr(path)
            feature_groups = raw_importances.\
                              groupby(['feature_group', 'model_id'])['feature_importance']\
                                      .mean()\
                                      .reset_index()
            feature_groups = feature_groups.rename(index=str, 
                                                   columns={"feature_importance":"importance_aggregate"})
        else:
            feature_groups = pd.read_sql(
            f'''
			WITH
			raw_importances AS(
			SELECT
			model_id,
			feature,
			feature_importance,
			CASE
			WHEN feature like 'Algorithm does not support a standard way to calculate feature importance.'
			THEN 'No feature group'
			ELSE split_part(feature, '_', 1)
			END AS feature_group,
			rank_abs
			FROM train_results.feature_importances
			WHERE model_id = {self.model_id}
			)
			SELECT
			model_id,
			feature_group,
			max(feature_importance) as importance_aggregate
			FROM raw_importances
			GROUP BY feature_group, model_id
			ORDER BY model_id, feature_group
            ''', con=self.engine)

        return feature_groups

    @cachedproperty
    def test_metrics(self):
        model_test_metrics = pd.read_sql(
            f'''
            SELECT model_id,
                   metric,
                   parameter,
                   stochastic_value as value,
                   num_labeled_examples,
                   num_labeled_above_threshold,
                   num_positive_labels
            FROM test_results.evaluations
            WHERE model_id = {self.model_id}
            ''', con=self.engine)
        return model_test_metrics

    @cachedproperty
    def train_metrics(self):
        model_train_metrics = pd.read_sql(
            f'''
            SELECT model_id,
                   metric,
                   parameter,
                   stochastic_value as value,
                   num_labeled_examples,
                   num_labeled_above_threshold,
                   num_positive_labels
            FROM test_results.evaluations
            WHERE model_id = {self.model_id}
            ''', con=self.engine)
        return model_train_metrics

    @cachedproperty
    def crosstabs(self):
        model_crosstabs = pd.read_sql(
            f'''
            SELECT model_id,
                   as_of_date,
                   metric,
                   feature_column,
                   value,
                   threshold_unit,
                   threshold_value
            FROM test_results.crosstabs
            WHERE model_id = {self.model_id}
            ''', con=self.engine)

        return model_crosstabs

    def preds_matrix(self, path):
        '''
        Load predictions matrices using the catwalk.storage.ProjectStorage.
        This class allow the user to define a project path that can be either a
        system local path or an s3 path and it will handle itself this
        different infrastructures.

        Once defined, we can pass this object to the MatrixStorageEngine that
        will read this object and return a MatrixSotre object with a set of
        handy methods to handle matrices

        Arguments:
            path: project path to initiate the ProjectStorage object
        '''
        cache = self.__dict__.setdefault('_preds_matrix_cache', {})
        try:
            return cache[path]
        except KeyError:
            pass

        storage = ProjectStorage(path)
        matrix_storage = MatrixStorageEngine(storage).get_store(self.pred_matrix_uuid)
        mat = matrix_storage.design_matrix

        # Merge with predictions table and return complete matrix
        merged_df = pd.merge(mat,
                             self.predictions,
                             on=['entity_id', 'as_of_date'],
                             how='inner',
                             suffixes=('test', 'pred'))

        cache[path] = merged_df
        return merged_df

    def train_matrix(self, path):
        '''
        Load predictions matrices using the catwalk.storage.ProjectStorage.
        This class allow the user to define a project path that can be either a
        system local path or an s3 path and it will handle itself this
        different infrastructures.

        Once defined, we can pass this object to the MatrixStorageEngine that
        will read this object and return a MatrixSotre object with a set of
        handy methods to handle matrices

        Arguments:
            path: project path to initiate the ProjectStorage object
        '''
        cache = self.__dict__.setdefault('_train_matrix_cache', {})
        try:
            return cache[path]
        except KeyError:
            pass

        storage = ProjectStorage(path)
        matrix_storage = MatrixStorageEngine(storage).get_store(self.train_matrix_uuid)
        mat = matrix_storage.design_matrix

        # Merge with predictions table and return complete matrix
        merged_df = mat.merge(self.predictions,
                             on='entity_id',
                             how='inner',
                             suffixes=('test', 'pred'))

        cache[path] = mat
        return mat

    def plot_score_distribution(self,
                                nbins=10,
                               figsize=(16,12),
                               fontsize=20):
        '''
        Generate an histograms with the raw distribution of the predicted
        scores for all entities.
            - Arguments:
                - nbins: bins to plot in histogram (default is 10).
                - label_names(tuple): define custom label names for class.
                - figsize (tuple): specify size of plot.
                - fontsize (int): define custom fontsize. 20 is set by default.

        '''

        df_ = self.predictions.filter(items=['score'])

        fig, ax = plt.subplots(1, figsize=figsize)
        plt.hist(df_.score,
                 bins=nbins,
                 density=True,
                 alpha=0.5,
                 color='blue')
        plt.axvline(df_.score.mean(),
                    color='black',
                    linestyle='dashed')
        ax.set_ylabel('Frequency', fontsize=fontsize)
        ax.set_xlabel('Score', fontsize=fontsize)
        plt.title('Score Distribution', y =1.2,
                  fontsize=fontsize)
        plt.show()

    def plot_score_label_distributions(self,
                                       nbins=10,
                                       label_names = ('Label = 0', 'Label = 1'),
                                       figsize=(16, 12),
                                       fontsize=20):
        '''
        Generate a histogram showing the label distribution for predicted
        entities:
            - Arguments:
                - label_names(tuple): define custom label names for class.
                - nbins: bins to plot in histogram (default is 10).
                - figsize (tuple): specify size of plot.
                - fontsize (int): define custom fontsize. 20 is set by default.
        '''

        df_predictions = self.predictions.filter(items=['score', 'label_value'])
        df__0 = df_predictions[df_predictions.label_value == 0]
        df__1 = df_predictions[df_predictions.label_value == 1]

        fig, ax = plt.subplots(1, figsize=figsize)
        plt.hist(df__0.score,
                 bins=nbins,
                 density=True,
                 alpha=0.5,
                 color='skyblue',
                 label=label_names[0])
        plt.hist(list(df__1.score),
                 bins=nbins,
                 density=True,
                 alpha=0.5,
                 color='orange',
                 label=label_names[1])
        plt.axvline(df__0.score.mean(),
                    color='skyblue',
                    linestyle='dashed')
        plt.axvline(df__1.score.mean(),
                    color='orange',
                    linestyle='dashed')
        plt.legend(bbox_to_anchor=(0., 1.005, 1., .102),
                   loc=8,
                   ncol=2,
                   borderaxespad=0.,
                   fontsize=fontsize-2)
        ax.set_ylabel('Frequency', fontsize=fontsize)
        ax.set_xlabel('Score', fontsize=fontsize)
        plt.title('Score Distribution across Labels', y =1.2,
                  fontsize=fontsize)
        plt.show()

    def plot_score_distribution_thresh(self,
                                       param_type,
                                       param,
                                       nbins = 10,
                                       label_names = ('Label = 0', 'Label = 1'),
                                       figsize=(16, 12),
                                       fontsize=20):
        '''
        Generate a histogram showing the label distribution for predicted
        entities using a threshold to classify between good prediction or not.
            - Arguments:
                - param_type (str): parameter to use,
                - param (int/float): threshold value to use
                - nbins: bins to plot in histogram (default is 10).
                - label_names(tuple): define custom label names for class.
                - figsize (tuple): specify size of plot.
                - fontsize (int): define custom fontsize. 20 is set by default.
        '''
        df_predictions = self.predictions

        if param_type == 'rank_abs':
            # Calculate residuals/errors
            preds_thresh = df_predictions.sort_values(['rank_abs'], ascending=True)
            preds_thresh['above_thresh'] = \
                    np.where(preds_thresh['rank_abs'] <= param/100, 1, 0)
        elif param_type == 'rank_pct':
            # Calculate residuals/errors
            preds_thresh = df_predictions.sort_values(['rank_pct'], ascending=True)
            preds_thresh['above_thresh'] = \
                    np.where(preds_thresh['rank_pct'] <= param/100, 1, 0)
        else:
            raise ValueError('''Error! You have to define a parameter type to
                                 set up a threshold
                                 ''')
        df__0 = df_predictions[df_predictions.label_value == 0]
        df__1 = df_predictions[df_predictions.label_value == 1]

        # Split dataframe for plotting
        #preds_above_thresh = preds_thresh[preds_thresh['above_thresh'] == 1]
        #preds_below_thresh = preds_thresh[preds_thresh['above_thresh'] == 0]
        threshold_value = preds_thresh[preds_thresh['above_thresh'] == 1].score.min()
        fig, ax = plt.subplots(1, figsize=figsize)
        plt.hist(df__0.score,
                 bins=nbins,
                 density=True,
                 alpha=0.5,
                 color='skyblue',
                 label=label_names[0])
        plt.hist(df__1.score,
                 bins=nbins,
                 density=True,
                 alpha=0.5,
                 color='orange',
                 label=label_names[1])
        plt.axvline(threshold_value,
                    color='black',
                    linestyle='dashed')
        plt.legend(bbox_to_anchor=(0., 1.005, 1., .102),
                   loc=8,
                   ncol=2,
                   borderaxespad=0.,
                   fontsize=fontsize-2)
        ax.set_ylabel('Density', fontsize=fontsize)
        ax.set_xlabel('Score', fontsize=fontsize)
        plt.title('Score Distribution using prediction threshold', y=1.2,
                  fontsize=fontsize)


    def plot_feature_importances(self,
                                 path,
                                 n_features_plots=30,
                                 figsize=(16, 12),
                                 fontsize=20):
        '''
        Generate a bar chart of the top n feature importances (by absolute value)
        Arguments:
                - save_file (bool): save file to disk as png. Default is False.
                - path (str): path to triage project_dir 
                - name_file (string): specify name file for saved plot.
                - n_features_plots (int): number of top features to plot
                - figsize (tuple): figure size to pass to matplotlib
                - fontsize (int): define custom fontsize for labels and legends.
        '''

        importances = self.feature_importances(path=path)
        importances = importances.filter(items=['feature', 'feature_importance'])
        importances = importances.set_index('feature')

        # Sort by the absolute value of the importance of the feature
        importances['sort'] = abs(importances['feature_importance'])
        importances = \
                importances.sort_values(by='sort', ascending=False).drop('sort', axis=1)
        importances = importances[0:n_features_plots]

        # Show the most important positive feature at the top of the graph
        importances = importances.sort_values(by='feature_importance', ascending=True)

        fig, ax = plt.subplots(figsize=figsize)
        ax.tick_params(labelsize=16)
        importances.plot(kind="barh", legend=False, ax=ax)
        ax.set_frame_on(False)
        ax.set_xlabel('Feature Importance', fontsize=20)
        ax.set_ylabel('Feature', fontsize=20)
        plt.tight_layout()
        plt.title(f'Top {n_features_plots} Feature Importances',
                  fontsize=fontsize).set_position([.5, 0.99])

    def plot_feature_importances_std_err(self,
                                         path,
                                         bar=True,
                                         n_features_plots=30,
                                         figsize=(16,21),
                                         fontsize=20):
        '''
        Generate a bar chart of the top n features importances showing the
        error bars. This plot is valid for ensemble classifiers (i.e. Random
        Forests or ExtraTrees), where many classifiers are bootstraped in
        estimation. This plot will allow the user to explore the feature
        importance variation inside the ensemble classifier.

        Arguments:
                - save_file (bool): save file to disk as png. Default is False.
                - name_file (string): specify name file for saved plot.
                - bar (bool): Should we plot a barplot or a scatter plot. If
                true, it will print a bar plot (set True by default).
                - n_features_plots (int): number of top features to plot.
                - figsize (tuple): figuresize to pass to matplotlib.
                - fontsize (int): define a custom fontsize for labels and legends.
                - *path: path to retrieve model pickle
        '''
        if 'sklearn.ensemble' in self.model_type: 

            storage = ProjectStorage(path)
            model_object = ModelStorageEngine(storage).load(self.model_hash)
            matrix_object = MatrixStorageEngine(storage).get_store(self.pred_matrix_uuid)

            # Calculate errors from model
            importances = model_object.feature_importances_
            std = np.std([tree.feature_importances_ for tree in model_object.estimators_],
                        axis=0)
            # Create dataframe and sort select the most relevant features (defined
            # by n_features_plot)
            importances_df = pd.DataFrame({
                'feature_name': matrix_object.columns(),
                'std': std,
                'feature_importance': importances
            }).set_index('feature_name')

            importances_sort = \
            importances_df.sort_values(['feature_importance'], ascending=False)
            importances_filter = importances_sort[:n_features_plots]

            # Plot features in order
            importances_ordered = \
            importances_filter.sort_values(['feature_importance'], ascending=True)


            if bar:
                # Plot features with sd bars
                fig, ax = plt.subplots(figsize=figsize)
                ax.tick_params(labelsize=16)
                importances_ordered['feature_importance'].\
                                   plot.barh(legend=False, 
                                             ax=ax,
                                             xerr=importances_ordered['std'],
                                             color='b')
                ax.set_frame_on(False)
                ax.set_xlabel('Feature Importance', fontsize=20)
                ax.set_ylabel('Feature', fontsize=20)
                plt.tight_layout()
                plt.title(f'Top {n_features_plots} Feature Importances with SD',
                          fontsize=fontsize).set_position([.5, 0.99])

            else:
                fig, ax = plt.subplots(figsize=figsize)
                ax.tick_params(labelsize=16)
                importances_ordered.plot.scatter(x = 'std',
                                                 y = 'feature_importance',
                                                 legend=False,
                                                 ax=ax)
                ax.set_xlabel('Std. Error', fontsize=20)
                ax.set_ylabel('Feature Importance', fontsize=20)
                plt.title(f'Top {n_features_plots} Feature Importances against SD',
                          fontsize=fontsize).set_position([.5, 0.99])
                feature_labels = []
                for k, v in importances_ordered.iterrows():
                    feature_labels.append(plt.text(v[0], v[1], k))
                adjust_text(feature_labels,
                            arrow_props=dict(arrowstype='->',
                                            color='r',
                                            lw=1))
        else:
            raise ValueError(f'''
            This plot is only available for Ensemble models, not
            {self.model_type}
            ''')

    def plot_feature_group_aggregate_importances(self,
                                               n_features_plots=30,
                                               figsize=(16, 12),
                                               fontsize=20,
                                               path=None):
        '''
        Generate a bar chart of the aggregate feature group importances (by absolute value)
        Arguments:
                - save_file (bool): save file to disk as png. Default is False.
                - path (str): path to the triage's project_path
                - name_file (string): specify name file for saved plot.
                - n_features_plots (int): number of top features to plot
                - figsize (tuple): figure size to pass to matplotlib
                - fontsize (int): define custom fontsize for labels and legends.
        '''

        fg_importances = self.feature_group_importances(path=path)
        fg_importances = fg_importances.filter(items=['feature_group', \
                                               'importance_aggregate'])
        fg_importances = fg_importances.set_index('feature_group')

        # Sort by the absolute value of the importance of the feature
        fg_importances['sort'] = abs(fg_importances['importance_aggregate'])
        fg_importances = \
                fg_importances.sort_values(by='sort', ascending=False).drop('sort', axis=1)

        # Show the most important positive feature at the top of the graph
        importances = fg_importances.sort_values(by='importance_aggregate', ascending=True)

        fig, ax = plt.subplots(figsize=figsize)
        ax.tick_params(labelsize=16)
        importances.plot(kind="barh", legend=False, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=15)
        ax.set_xlabel('Feature Importance', fontsize=20)
        ax.set_ylabel('Feature Group', fontsize=20)
        plt.tight_layout()
        plt.title(f'Feature Group Importances',
                  fontsize=fontsize).set_position([.5, 1.0])


    def cluster_correlation_features(self,
                                     path,
                                     feature_group_subset_list=None,
                                     cmap_color_fgroups='Accent',
                                     cmap_heatmap='mako',
                                     figsize=(16,16),
                                     fontsize=12):
        '''
        Plot correlation in feature space

        This function simply renders the correlation between features and uses
        hierarchical clustering to identify cluster of features. The idea bhind
        this plot is not only to explore the correlation in the space, but also
        to explore is this correlation is comparable with the feature groups.
        Arguments:
            - path (string): Project directory where to find matrices and
            models. Usually is under 'triage/outcomes/'
            - feature_group_subset_list (list): list of feature groups to plot.
            By default, the function uses all the feature groups.
            - cmap_color_fgroups (string): matplotlib pallete to color the
            feature groups.
            - cmap_heatmap (string):seaborn/matplotlib pallete to color the
            correlation/clustering matrix
            - figsize (tuple): define size of the plot (please use square
            dimensions)
            - fontsize (string): define size of plot title and axes.
         '''
        if feature_group_subset_list is None:
             feature_group_subset = self.feature_importances(path).feature_group.unique()
             feature_regex = '|'.join(feature_group_subset)
        else:
             feature_group_subset = feature_group_subset_list
             feature_regex = '|'.join(feature_group_subset_list)

        # Load Prediction Matrix
        test_matrix = self.preds_matrix(path)

        # Define feature space (remove predictions)
        storage = ProjectStorage(path)
        matrix_storage = \
        MatrixStorageEngine(storage).get_store(self.pred_matrix_uuid)
        feature_columns = matrix_storage.columns()

        # Prepare and calculate feature correlation
        test_matrix = test_matrix[feature_columns]
        test_matrix_filter = test_matrix.filter(regex=feature_regex)
        feature_groups = [x.split('_', 1)[0] for x in \
                          test_matrix_filter.columns.tolist()]
        corr = test_matrix_filter.corr()

        # Define feature groups and colors
        feature_groups = pd.DataFrame(feature_groups,
                              columns = ['feature_group'],
                              index = corr.index.tolist())

        cmap = plt.get_cmap(cmap_color_fgroups)
        colors = cmap(np.linspace(0, 1,
                                  len(feature_groups.feature_group.unique())))
        lut = dict(zip(feature_groups.feature_group.unique(), colors))
        row_colors = feature_groups.feature_group.map(lut)

        legend_feature_groups = \
                [mpatches.Patch(color=value, label=key) for key,value in \
                 lut.items()]

        # Plot correlation/cluster map
        ax = sns.clustermap(corr.fillna(0),
                       row_colors=row_colors,
                       cmap=cmap_heatmap)
        plt.title(f'Clustered Correlation matrix plot for {self.model_id}',
                  fontsize=fontsize)
        plt.legend(handles=legend_feature_groups,
                   title= 'Feature Group',
                   bbox_to_anchor=(0., 1.005, 1., .102),
                   loc=7,
                   borderaxespad=0.)


    def cluster_correlation_sparsity(self,
                                     path,
                                     figsize=(20,20),
                                     fontsize=12):
         '''
        Plot sparcity in feature space

        This function simply renders the correlation between features and uses
        hierarchical clustering to identify cluster of features. The idea bhind
        this plot is not only to explore the correlation in the space, but also
        to explore is this correlation is comparable with the feature groups.
        Arguments:
            - path (string): Project directory where to find matrices and
            models. Usually is under 'triage/outcomes/'
            - cmap_heatmap (string):seaborn/matplotlib pallete to color the
            correlation/clustering matrix
            - figsize (tuple): define size of the plot (please use square
            dimensions)
            - fontsize (string): define size of plot title and axes.
         '''

         # Load Prediction Matrix
         test_matrix = self.preds_matrix(path)

         # Define feature space (remove predictions)
         storage = ProjectStorage(path)
         matrix_storage = \
         MatrixStorageEngine(storage).get_store(self.pred_matrix_uuid)
         feature_columns = matrix_storage.columns()

         # Prepare and calculate feature correlation
         test_matrix = test_matrix[feature_columns]

         # Create sparse matrix
         # 1: Values with more than 0, and 0 to values with 0
         sparse_feature_matrix = test_matrix.where(test_matrix == 0).fillna(1)
         sparse_feature_matrix_filter = sparse_feature_matrix.apply(lambda x: \
                                                            x.sort_values().values)

         sparse_feature_matrix = test_matrix.where(test_matrix == 0).fillna(1)
         sparse_feature_matrix_filter = \
                sparse_feature_matrix.apply(lambda x: x.sort_values().values)
         num_zeros = sparse_feature_matrix.sum(axis=0)
         sparse_feature_matrix_filter_columns = \
                 sparse_feature_matrix_filter[num_zeros. \
                                              sort_values(ascending=False).index.values]

		 # Plot matrix
         fig, ax = plt.subplots(figsize=figsize)
         plt.title(f'Feature space sparse matrix for {self.model_id}',
                   fontsize=fontsize)
         ax.set_xlabel('Features', fontsize=fontsize)
         ax.set_ylabel('Entity ID', fontsize=fontsize)
         cbar_kws = {'ticks': range(2)}
         sns.heatmap(sparse_feature_matrix_filter_columns,
                     cmap=sns.color_palette("hls", 2),
                     cbar_kws=cbar_kws)


    def compute_AUC(self):
        '''
        Utility function to generate ROC and AUC data to plot ROC curve
        Returns (tuple):
            - (false positive rate, true positive rate, thresholds, AUC)
        '''

        label_ = self.predictions.label_value
        score_ = self.predictions.score
        fpr, tpr, thresholds = metrics.roc_curve(
            label_, score_, pos_label=1)

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


        _labels = self.predictions.filter(items=['label_value', 'score'])

        # since tpr and recall are different names for the same metric, we can just
        # grab fpr and tpr from the sklearn function for ROC
        fpr, recall, thresholds, auc = self.compute_AUC()

        # turn the thresholds into percent of the list traversed from top to bottom
        pct_above_per_thresh = []
        num_scored = float(len(_labels.score))
        for value in thresholds:
            num_above_thresh = len(_labels.loc[_labels.score >= value, 'score'])
            pct_above_thresh = num_above_thresh / num_scored
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)

        # plot the false positive rate, along with a dashed line showing the optimal bounds
        # given the proportion of positive labels
        plt.clf()
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot([_labels.label_value.mean(), 1], [0, 1], '--', color='gray')
        ax1.plot([0, _labels.label_value.mean()], [0, 0], '--', color='gray')
        ax1.plot(pct_above_per_thresh, fpr, "#000099")
        plt.title('Deconstructed ROC Curve\nRecall and FPR vs. Depth',fontsize=fontsize)
        ax1.set_xlabel('Proportion of Population', fontsize=fontsize)
        ax1.set_ylabel('False Positive Rate', color="#000099", fontsize=fontsize)
        plt.ylim([0.0, 1.05])


    def plot_precision_recall_n(self,
                                figsize=(16, 12),
                                fontsize=20):
        """
        Plot recall and precision curves against depth into the list.
        """


        _labels = self.predictions.filter(items=['label_value', 'score'])

        y_score = _labels.score
        precision_curve, recall_curve, pr_thresholds = \
            metrics.precision_recall_curve(_labels.label_value, y_score)

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

    def _error_labeler(self,
                      path,
                      param=None,
                      param_type=None):
        '''
        Explore the underlying causes of errors using decision trees to explain the
        residuals base on the same feature space used in the model. This
        exploration will get the most relevant features that determine y - y_hat
        distance and may help to understand the outomes of some models.

        This function will label the errors and return two elements relevant to
        model these. First, a feature matrix (X) with all the features used by
        the model. Second, an iterator with different labeled errors: FPR, FRR,
        and the general error.

        Arguments:
            - param_type: (str) type of parameter to define a threshold. Possible
              values come from triage evaluations: rank_abs, or rank_pct
            - param: (int) value
            - path: path for the ProjectStorage class object
        '''

        test_matrix = self.preds_matrix(path)

        if param_type == 'rank_abs':
            # Calculate residuals/errors
            test_matrix_thresh = test_matrix.sort_values(['rank_abs'], ascending=True)
            test_matrix_thresh['above_thresh'] = \
                    np.where(test_matrix_thresh['rank_abs'] <= param, 1, 0)
            test_matrix_thresh['error'] = test_matrix_thresh['label_value'] - \
                    test_matrix_thresh['above_thresh']
        elif param_type == 'rank_pct':
            # Calculate residuals/errors
            test_matrix_thresh = test_matrix.sort_values(['rank_pct'], ascending=True)
            test_matrix_thresh['above_thresh'] = \
                    np.where(test_matrix_thresh['rank_pct'] <= param, 1, 0)
            test_matrix_thresh['error'] = test_matrix_thresh['label_value'] - \
                    test_matrix_thresh['above_thresh']
        else:
            raise AttributeError('''Error! You have to define a parameter type to
                                 set up a threshold
                                 ''')

        # Define labels using the errors
        dict_errors = {'FP': (test_matrix_thresh['label_value'] == 0) &
                              (test_matrix_thresh['above_thresh'] == 1),
                       'FN': (test_matrix_thresh['label_value'] == 1) &
                              (test_matrix_thresh['above_thresh'] == 0),
                       'TP':  (test_matrix_thresh['label_value'] == 1) &
                              (test_matrix_thresh['above_thresh'] == 1),
                       'TN':  (test_matrix_thresh['label_value'] == 0) &
                              (test_matrix_thresh['above_thresh'] == 0)
                      }
        test_matrix_thresh['class_error'] = np.select(condlist=dict_errors.values(),
                                                     choicelist=dict_errors.keys(),
                                                     default=None)

        # Split data frame to explore FPR/FNR against TP and TN
        test_matrix_thresh_0 = \
        test_matrix_thresh[test_matrix_thresh['label_value'] == 0]
        test_matrix_thresh_1 = \
        test_matrix_thresh[test_matrix_thresh['label_value'] == 1]
        test_matrix_predicted_1 = \
        test_matrix_thresh[test_matrix_thresh['above_thresh'] == 1]

        dict_error_class = {'FPvsAll': (test_matrix_thresh['class_error'] == 'FP'),
                            'FNvsAll': (test_matrix_thresh['class_error'] == 'FN'),
                            'FNvsTP': (test_matrix_thresh_1['class_error'] == 'FN'),
                            'FPvsTN': (test_matrix_thresh_0['class_error'] == 'FP'),
                            'FPvsTP': (test_matrix_predicted_1['class_error'] == 'FP')}

        # Create label iterator
        Y = [(np.where(condition, 1, -1), label) for label, condition in \
             dict_error_class.items()]

        # Define feature space to model: get the list of feature names
        storage = ProjectStorage(path)
        matrix_storage = MatrixStorageEngine(storage).get_store(self.pred_matrix_uuid)
        feature_columns = matrix_storage.columns()

        # Build error feature matrix
        matrices = [test_matrix_thresh,
                    test_matrix_thresh,
                    test_matrix_thresh_1,
                    test_matrix_thresh_0,
                    test_matrix_predicted_1]
        X = [matrix[feature_columns] for matrix in matrices]

        return zip(Y, X)

    def _error_modeler(self,
                      depth=None,
                      view_plots=False,
                      **kwargs):
       '''
       Model labeled errors (residuals) by the error_labeler (FPR, FNR, and
       general residual) using a RandomForestClassifier. This function will
       yield a plot tree for each of the label numpy arrays return by the
       error_labeler (Y).
       Arguments:
           - depth: max number of tree partitions. This is passed directly to
             the classifier.
           - view_plot: the plot is saved to disk by default, but the
             graphviz.Source also allow to load the object and see it in the
             default OS image renderer
           - **kwargs: more arguments passed to the labeler: param indicating
             the threshold value, param_type indicating the type of threshold,
             and the path to the ProjectStorage.
       '''

       # Get matrices from the labeler
       zip_data = self._error_labeler(param_type = kwargs['param_type'],
                                  param = kwargs['param'],
                                  path=kwargs['path'])

       # Model tree and output tree plot
       for error_label, matrix in zip_data:

           dot_path = 'error_analysis_' + \
                      str(error_label[1]) + '_' + \
                      str(self.model_id) + '_' + \
                      str(kwargs['param_type']) + '@'+ \
                      str(kwargs['param']) +  '.gv'

           clf = tree.DecisionTreeClassifier(max_depth=depth)
           clf_fit = clf.fit(matrix, error_label[0])
           tree_viz = tree.export_graphviz(clf_fit,
                                           out_file=None,
                                           feature_names=matrix.columns.values,
                                           filled=True,
                                           rounded=True,
                                           special_characters=True)
           graph = graphviz.Source(tree_viz)
           graph.render(filename=dot_path,
                        directory='error_analysis',
                        view=view_plots)

           print(dot_path)

    def error_analysis(self, threshold, **kwargs):
        '''
        Error analysis function for ThresholdIterator objects. This function
        have the same functionality as the _error.modeler method, but its
        implemented for iterators, which can be a case use of this analysis.
        If no iterator object is passed, the function will take the needed
        arguments to run the _error_modeler.
        Arguments:
            - threshold: a threshold and threshold parameter combination passed
            to the PostmodelingParamters. If multiple parameters are passed,
            the function will iterate through them.
            -**kwags: other arguments passed to _error_modeler
        '''

        error_modeler = partial(self._error_modeler,
                               depth = kwargs['depth'],
                               path = kwargs['path'],
                               view_plots = kwargs['view_plots'])

        if isinstance(threshold, dict):
           for threshold_type, threshold_list in threshold.items():
               for threshold in threshold_list:
                   print(threshold_type, threshold)
                   error_modeler(param_type = threshold_type,
                                 param = threshold)
        else:
           error_modeler(param_type=kwargs['param_type'],
                         param=kwargs['param'])


    def crosstabs_ratio_plot(self,
                             n_features=30,
                             figsize=(12,16),
                             fontsize=20):
        '''
        Plot to visualize the top-k features with the highest mean ratio. This
        plot will show the biggest quantitative differences between the labeled/predicted
        groups
        '''
        crosstabs_ratio = self.crosstabs.loc[self.crosstabs.metric == \
                          'ratio_predicted_positive_over_predicted_negative']
        crosstabs_ratio_subset = crosstabs_ratio.sort_values(by=['value'],
                                 ascending=False)[:n_features]
        crosstabs_ratio_plot = \
        crosstabs_ratio_subset.filter(items=['feature_column','value']).\
                set_index('feature_column').\
                sort_values(['value'])

        fig, ax = plt.subplots(figsize=figsize)
        ax.tick_params(labelsize=16)
        crosstabs_ratio_plot.plot(kind="barh", legend=False, ax=ax)
        ax.set_frame_on(False)
        ax.set_xlabel('Ratio means (positive/negative)', fontsize=20)
        ax.set_ylabel('Feature', fontsize=20)
        plt.tight_layout()
        plt.title(f'Top {n_features} features with higher mean ratio',
                  fontsize=fontsize).set_position([.5, 0.99])

    def plot_feature_distribution(self,
                                  path,
                                  feature_list=None):
        '''
        Plot feature distributions (and compare feature distributions across
        labels)
        '''

        if feature_list is None:
            f_importances = self.feature_importances(path)
            top_f = f_importances[f_importances['rank_abs'] <= 10]['feature'].tolist()
            feature_list = top_f

        n = len(feature_list)

        fig, axs = plt.subplots(n, 3, figsize=(20,7*n))
        axs = axs.ravel()

        matrix = self.preds_matrix(path=path)
 
        for idx,feature in enumerate(feature_list):
            i1 = 3*idx 
            i2 = 3*idx + 1
            i3 = 3*idx + 2
            f_0 = matrix[matrix.label_value==0][feature]
            f_1 = matrix[matrix.label_value==1][feature]

            if len(matrix[feature].unique()) == 2:
                axs[i1].hist(f_0, bins=20,density=True,alpha=0.5, 
                             label=str(yr), color=colors[yr], histtype='step')
                axs[i2].hist(f_1, bins=20,density=True,alpha=0.5, 
                             label=str(yr), color=colors[yr], linestyle="--", histtype='step')
                axs[i3].hist(f_0, bins=20,density=True,alpha=0.8,histtype='step',
                             label=str(yr), color=colors[yr])
                axs[i3].hist(f_1, bins=20,density=True,alpha=1, linestyle="--", histtype='step',
                             label=str(yr), color=colors[yr])
            else:
                sns.distplot(matrix[matrix.label_value == 0][feature], 
                             hist=False, 
                             kde=True, 
                             kde_kws={'linewidth': 2}, 
                             ax=axs[i1]
                        )
                sns.distplot(matrix[matrix.label_value == 1][feature], 
                             hist=False, 
                             kde=True, 
                             kde_kws={'linewidth': 2, 'linestyle':'--'}, 
                             ax=axs[i2]
                        )
                sns.distplot(f_0, 
                             hist=False, 
                             kde=True, 
                             kde_kws={'linewidth': 2} , 
                             ax=axs[i3])
                sns.distplot(f_1, 
                             hist=False, 
                              kde=True, 
                             kde_kws={'linewidth': 2, 'linestyle':'--'}, 
                             ax=axs[i3])

            axs[i1].legend()
            axs[i1].set_title("0 class")
            axs[i1].set_xlabel(feature)
            axs[i2].legend()
            axs[i2].set_title("1 class")
            axs[i2].set_xlabel(feature)
            axs[i3].legend()
            axs[i3].set_title("All classes")
            axs[i3].set_xlabel(feature)
        plt.tight_layout()
        plt.show()

