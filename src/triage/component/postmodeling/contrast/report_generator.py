import logging
import pandas as pd
import matplotlib.pyplot as plt

from descriptors import cachedproperty

from triage.component.postmodeling.contrast.model_class import ModelAnalyzer


class PostmodelingReport: 

    def __init__(self, engine, model_groups, experiment_hash, project_path=None) -> None:
        self.model_groups = model_groups
        self.experiment_hash = experiment_hash
        self.engine = engine
        self.project_path = project_path
        self.models = self.get_model_ids()
        
    @property
    def model_ids(self):
        pass

    @property
    def model_types(self):
        pass

    def get_model_ids(self):
        """for the model group ids, fetch the model_ids and initialize the datastructure"""

        model_groups = "', '".join([str(x) for x in self.model_groups])
        q = f"""
            select distinct on (model_group_id, train_end_time)
                model_id, 
                train_end_time::date,
                model_group_id
            from triage_metadata.models 
                join triage_metadata.experiment_models using(model_hash)
            where experiment_hash='{self.experiment_hash}'
            and model_group_id in ('{model_groups}')        
            """  

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
            dpi=100
        )

        return fig, axes

    def plot_score_distributions(self):
        """for the model group ids plot score grid"""
        fig, axes = self._get_subplots()

        for i, mg in enumerate(self.models):
            for j, train_end_time in enumerate(self.models[mg]):
                mode_analyzer = self.models[mg][train_end_time]

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


    def plot_feature_importance(self, n_top_features=20):
        """ plot all feature importance  """

        fig, axes = self._get_subplots(subplot_width=7)

        for i, mg in enumerate(self.models):
            for j, train_end_time in enumerate(self.models[mg]):
                mode_analyzer = self.models[mg][train_end_time]

                mode_analyzer.plot_feature_importance(
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

        fig.suptitle(f'{n_top_features} Features with highest importance (magnitude)')
        fig.tight_layout()


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

        




    
    




    