import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from IPython.display import display


from descriptors import cachedproperty
from scipy.stats import spearmanr

from triage.component.postmodeling.model_analyzer import ModelAnalyzer
from triage.component.postmodeling.error_analysis import generate_error_analysis, output_all_analysis


class PostmodelingAnalyzer: 

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
                    d[m['model_group_id']][m['train_end_time']] = ModelAnalyzer(m['model_id'], self.engine)
                else:
                    d[m['model_group_id']] = {m['train_end_time']: ModelAnalyzer(m['model_id'], self.engine)}

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


    