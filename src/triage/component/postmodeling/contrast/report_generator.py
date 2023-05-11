import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from descriptors import cachedproperty

from triage.component.postmodeling.contrast.model_class import ModelAnalyzer


class PostmodelingReport: 

    def __init__(self, engine, model_groups, experiment_hashes, project_path=None, use_all_model_groups=False) -> None:
        self.model_groups = model_groups
        self.experiment_hashes = experiment_hashes # TODO made experiment hashes into a list to plot models from different experiments for MVESC, there's probably a better way to generalize this
        self.engine = engine
        self.project_path = project_path
        if use_all_model_groups: # shortcut to listing out all model groups for an experiment
            self.use_all_model_groups()
        self.models = self.get_model_ids()
        
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
        data_df = pd.DataFrame(data_dict, columns=['model_group_id', 'train_end_time', 'model_id', 'model_type', 'hyperparameters'])
        return data_df

    def get_model_ids(self):
        """for the model group ids, fetch the model_ids and initialize the datastructure"""

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

        if n_rows is None: 
            n_rows = len(self.model_groups)
        
        if n_cols is None: 
            n_cols = len(self.models[self.model_groups[0]])
        
        
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize = (subplot_width*n_cols, subplot_len*n_rows),
            dpi=100,
            squeeze=False # to handle only one model group
        )

        return fig, axes

    def plot_score_distributions(self, use_labels):
        """for the model group ids plot score grid"""

        # FIXME -- This breaks if only one model group is given to the report generator
        fig, axes = self._get_subplots()

        for i, mg in enumerate(self.models):
            for j, train_end_time in enumerate(self.models[mg]):
                mode_analyzer = self.models[mg][train_end_time]

                if use_labels:
                    mode_analyzer.plot_score_label_distribution(ax=axes[i, j])
                else:
                    mode_analyzer.plot_score_distribution(
                        ax=axes[i, j]
                    )

                if j==0:
                    axes[i, j].set_ylabel(f'Mod Grp: {mg}')
                else:
                    axes[i, j].set_ylabel('')
                
                if i == 0:
                    axes[i, j].set_title(f'{train_end_time}({mode_analyzer.model_id})')
                else:
                    axes[i, j].set_title('')

        fig.suptitle('Score Distributions')
        fig.tight_layout()
        

    def plot_calibration_curves(self):
        fig, axes = self._get_subplots()

        for i, mg in enumerate(self.models):
            for j, train_end_time in enumerate(self.models[mg]):
                mode_analyzer = self.models[mg][train_end_time]

                mode_analyzer.plot_calibration_curve(
                    ax=axes[i, j]
                )

                if j==0:
                    axes[i, j].set_ylabel(f'Mod Grp: {mg}')
                else:
                    axes[i, j].set_ylabel('')
                
                if i == 0:
                    axes[i, j].set_title(f'{train_end_time} ({mode_analyzer.model_id})')
                else:
                    axes[i, j].set_title('')

        fig.suptitle('Calibration Curves')
        fig.tight_layout()

    def plot_prk_curves(self):
        fig, axes = self._get_subplots()

        for i, mg in enumerate(self.models):
            for j, train_end_time in enumerate(self.models[mg]):
                mode_analyzer = self.models[mg][train_end_time]

                mode_analyzer.plot_precision_recall_curve(
                    ax=axes[i, j]
                )

                if j==0:
                    axes[i, j].set_ylabel(f'Mod Grp: {mg}')
                else:
                    axes[i, j].set_ylabel('')
                
                if i == 0:
                    axes[i, j].set_title(f'{train_end_time} ({mode_analyzer.model_id})')
                else:
                    axes[i, j].set_title('')

        fig.suptitle('Precision-Recall with Positive Prediction %')
        fig.tight_layout()

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

        # For readability, we will plot the feature importance with time advancing on the vertical
        fig, axes = self._get_subplots(
            subplot_width=7,
            n_rows = len(self.models[self.model_groups[0]]), # train end tomes
            n_cols = len(self.model_groups) # mdoel groups
        )

        for j, mg in enumerate(self.models):
            for i, train_end_time in enumerate(self.models[mg]):
                mode_analyzer = self.models[mg][train_end_time]

                mode_analyzer.plot_feature_importance(
                    ax=axes[i, j],
                    n_top_features=n_top_features
                )

                if j==0:
                    axes[i, j].set_ylabel(f'{train_end_time}') # first column
                else:
                    axes[i, j].set_ylabel('')
                
                # if i == 0:
                #     # axes[i, j].set_title(f'{train_end_time} ({mode_analyzer.model_id})')
                #     axes[i, j].set_title(f'Mod Grp: {mg}') # Top row

                # # else:
                #     # axes[i, j].set_title('')

        # fig.suptitle(f'{n_top_features} Features with highest importance (magnitude)')
        fig.tight_layout()

    def plot_feature_group_importance(self, n_top_groups=20):
        """ plot all feature group importance  """

        # For readability, we will plot the feature importance with time advancing on the vertical
        fig, axes = self._get_subplots(
            subplot_width=7,
            n_rows = len(self.models[self.model_groups[0]]), # train end tomes
            n_cols = len(self.model_groups) # mdoel groups
        )

        for j, mg in enumerate(self.models):
            for i, train_end_time in enumerate(self.models[mg]):
                mode_analyzer = self.models[mg][train_end_time]

                mode_analyzer.plot_feature_group_importance(
                    ax=axes[i, j],
                    n_top_groups=n_top_groups
                )

                if j==0:
                    axes[i, j].set_ylabel(f'{train_end_time} ({mode_analyzer.model_id})') # first column
                else:
                    axes[i, j].set_ylabel('')
                
                if i == 0:
                    # axes[i, j].set_title(f'{train_end_time} ({mode_analyzer.model_id})')
                    axes[i, j].set_title(f'Mod Grp: {mg}') # Top row

                else:
                    axes[i, j].set_title('')

        fig.suptitle(f'Feature groups with highest importance (magnitude)')
        fig.tight_layout()



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

        




    
    




    