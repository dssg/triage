import pandas as pd
import matplotlib.pyplot as plt

from descriptors import cachedproperty

from triage.component.postmodeling.contrast.model_class import ModelAnalyzer


class PostmodelingReport: 

    def __init__(self, engine, model_groups, experiment_hash) -> None:
        self.model_groups = model_groups
        self.experiment_hash = experiment_hash
        self.engine = engine
        self.models = self.get_model_ids()

    @cachedproperty
    def model_ids(self):
        pass

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
            
    def _get_subplots(self, subplot_width=3, subplot_len=None):
        """"""

        if subplot_len is None:
            subplot_len = subplot_width

        row = len(self.model_groups)
        col = len(self.models[self.model_groups[0]])
        fig, axes = plt.subplots(
            row,
            col,
            figsize = (subplot_width*col, subplot_len*row)
        )

        return fig, axes

    def score_distributions(self):
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
        

    def prk_curves(self):
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


    
    




    