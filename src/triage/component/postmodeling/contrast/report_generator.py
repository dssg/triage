import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools


from descriptors import cachedproperty
from scipy.stats import spearmanr

from triage.component.postmodeling.contrast.model_class import ModelAnalyzer
from triage.component.postmodeling.error_analysis import generate_error_analysis


class PostmodelingReport: 

    def __init__(self, engine, model_groups, experiment_hashes, project_path=None, use_all_model_groups=False) -> None:
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
        pass

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
        for m in models:
            if m['model_group_id'] in d:
                d[m['model_group_id']][m['train_end_time']] = ModelAnalyzer(m['model_id'], self.engine)
            else:
                d[m['model_group_id']] = {m['train_end_time']: ModelAnalyzer(m['model_id'], self.engine)}

        return d 
    
    def _get_subplots(self, subplot_width=3, subplot_len=None, n_rows=None, n_cols=None):
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
            squeeze=False # to handle only one model group
        )

        return fig, axes


    def _make_plot_grid(self, plot_type, subplot_width=3, subplot_len=None, **kw):
        """
            Abstracts out generating the plot grid (time x model group) for comparing
            model_scores,  
        """
        fig, axes = self._get_subplots(subplot_width=subplot_width, subplot_len=subplot_len)

        for j, mg in enumerate(self.models):
            for i, train_end_time in enumerate(self.models[mg]):
                model_analyzer = self.models[mg][train_end_time]

                plot_func = getattr(model_analyzer, plot_type)

                plot_func(ax=axes[i, j], **kw)

                if j==0:
                    axes[i, j].set_ylabel(f'{train_end_time}')
                else:
                    axes[i, j].set_ylabel('')
                
                axes[i, j].set_xlabel('')

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

    def plot_prk_curves(self):
        self._make_plot_grid(plot_type='plot_precision_recall_curve')
  
    def plot_bias_threshold(self, attribute_name, attribute_values, bias_metric):
        """
            Plot bias_metric for the specified list of attribute_values for a particular attribute_name across different thresholds (list %)
        """
        fig, axes = self._get_subplots(subplot_width=6, n_rows=len(attribute_values))
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
        fig, axes = self._get_subplots(subplot_width=6, n_rows=1)
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

            positive_support_threshold (float, optional): The threshold of pct support for the feature (instances with non-zero values) among predicted positives 
        """
        
        fig, axes = self._get_subplots(
            subplot_width=4,
            subplot_len=1 + (display_n_features*2) / 5,
            n_rows=len(self.models[self.model_groups[0]]) * len(self.model_groups),
            n_cols=1
        )

        dfs = dict()

        for i, mg in enumerate(self.models):
            for j, train_end_time in enumerate(self.models[mg]):
                model_analyzer = self.models[mg][train_end_time]

                idx = i * len(self.models[self.model_groups[0]]) + j 
                try:
                    df = model_analyzer.display_crosstabs_pos_vs_neg(
                        threshold_type=threshold_type,
                        threshold=threshold,
                        display_n_features=display_n_features,
                        filter_features=filter_features,
                        support_threshold=support_threshold,
                        table_name=table_name,
                        ax=axes[idx],
                        return_df=True
                    )

                    dfs[model_analyzer.model_id] = df
                except ValueError as e:
                    logging.error('Please run calculate_crosstabs_pos_vs_neg function to calculate crosstabs first for all models!')
                    raise e


        fig.suptitle(
            f'{display_n_features} Features with highest & lowest pos:neg mean ratio',
            x=-0.1,
            fontsize=11
        )

        if return_dfs:
            return dfs

    def pairwise_list_comparison(self, threshold_type, threshold, train_end_time, matrix_uuid=None, plot=True):
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
            results[m] = pd.DataFrame(index=self.model_groups, columns=self.model_groups)


        for model_group_pair in pairs:
            logging.info(f'Comparing {model_group_pair[0]} and {model_group_pair[1]}')

            df1 = lists[model_group_pair[0]]
            df2 = lists[model_group_pair[1]]

            # calculating jaccard similarity and overlap
            entities_1 = set(df1.entity_id)
            entities_2 = set(df2.entity_id)

            inter = entities_1.intersection(entities_2)
            un = entities_1.union(entities_2)    
            results['jaccard'].loc[model_group_pair[1], model_group_pair[0]] = len(inter)/len(un)

            # If the list sizes are not equal, using the smallest list size to calculate simple overlap
            results['overlap'].loc[model_group_pair[1], model_group_pair[0]] = len(inter)/ min(len(entities_1), len(entities_2))

            # calculating rank correlation
            df1.sort_values('score', ascending=False, inplace=True)
            df2.sort_values('score', ascending=False, inplace=True)

            # only returning the corr coefficient, not the p-value
            results['rank_corr'].loc[model_group_pair[0], model_group_pair[1]] = spearmanr(df1.entity_id.iloc[:], df2.entity_id.iloc[:])[0]

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

            fig.tight_layout()
        return results

    def execute_error_analysis(self):
        """Generates the error analysis of a model
        """
        model_ids = self.models
        db_conn = self.engine.connection()
        for model_id in model_ids:
            generate_error_analysis(model_id, db_conn, self.project_path)

    