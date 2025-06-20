import pandas as pd
import logging

import altair as alt



class ModelGroupComparison:
    
    def __init__(self, model_group_ids, engine):
        """
        Initialize the ModelGroupComparison with model group IDs and a database engine.
        
        :param model_group_ids: List of model group IDs to compare.
        :param engine: Database engine for executing SQL queries.
        """
        
        self.model_group_ids = model_group_ids
        self.engine = engine
        
    def bias_and_performance(self, performance_metric='precision@', bias_metric='tpr_disparity', parameter='1_pct'):
        """
        Compare model groups based on performance and bias metrics.
        
        :param performance_metric: The performance metric to use for comparison.
        :param bias_metric: The bias metric to use for comparison.
        :param parameter: The threshold for determining significant differences.
        :return: DataFrame with comparison results.
        """
        
        # check if there are bias audit results in the DB (directly from aequitas table)
        q = f'''
        
            select 
            distinct a.attribute_name, a.attribute_value
            from triage_metadata.models m join test_results.aequitas a using (model_id)
            where m.model_group_id in ({', '.join([str(x) for x in self.model_group_ids])})
            order by 1, 2
        '''
        
        rg = pd.read_sql(q, self.engine)
        
        if rg.empty:
            logging.warning("No bias audit results found in the database. Returning.")
            return None
        
        attributes = rg['attribute_name'].unique()
        attribute_values = rg['attribute_value'].unique()
        
        parameter_ae = parameter
        
        if 'pct' in parameter:
            t = round(float(parameter.split('_')[0]) / 100, 2)
            parameter_ae =  f'{t}_pct'
        
        q = f'''
            select 
                m.model_id,
                m.model_group_id,
                m.model_type,
                m.hyperparameters,
                m.train_end_time::date,
                e.metric,
                e."parameter",
                e.num_labeled_examples,
                e.num_positive_labels,
                e.stochastic_value as "{performance_metric}{parameter}",
                a.tpr,
                a.{bias_metric},
                a.attribute_name,
                a.attribute_value,
                a.group_size,
                a.prev as baserate,
                total_entities
            from triage_metadata.models m left join test_results.evaluations e 
                    on m.model_id = e.model_id 
                    and e.metric = '{performance_metric}'
                    and e."parameter" = '{parameter}'
                    and e.subset_hash = ''
                        left join test_results.aequitas a 
                            on m.model_id = a.model_id 
                            and a."parameter" = '{parameter_ae}'
                            and a.attribute_name in ('{"','".join(attributes)}')
                            and a.attribute_value in ('{"','".join(attribute_values)}')
                            and a.tie_breaker= 'worst' 
            where m.model_group_id in ({", ".join([str(x) for x in self.model_group_ids])})
            and group_size::float / num_labeled_examples > 0.01
            and num_positive_labels > 0
        '''
        
        metrics = pd.read_sql(q, self.engine)
        metrics['model_type_display'] = metrics.model_type.str.split('.').str[-1]
        
        # grouping:
        df = round(metrics.groupby(['model_group_id', 'attribute_name', 'attribute_value']).agg(
            model_type=('model_type_display', 'max'),
            mean_bias=(f'{bias_metric}', 'mean'),
            sem_bias=(f'{bias_metric}', 'sem'),
            mean_perf=(f'{performance_metric}{parameter}', 'mean'),
            sem_perf=(f'{performance_metric}{parameter}', 'sem'),
            group_size=('group_size', 'mean')
        ), 3).reset_index()
        
        reference_group_filter = (df.mean_bias == 1)
        
        filtered_df = df[~reference_group_filter]
        
        points = alt.Chart(filtered_df).mark_point(filled=True, size=80, opacity=1).encode(
        x=alt.X('mean_perf', axis=alt.Axis(title='Performance Metric', grid=False)),
        y=alt.Y('mean_bias', axis=alt.Axis(title='Bias Metric', grid=False), scale=alt.Scale(domain=(0, 3))),
        color=alt.Color('model_group_id:N', title='Model'),
        shape=alt.Shape('attribute_value:N', title='Protected Groups'),
        tooltip=[
            alt.Tooltip('attribute_name', title='Attribute'), 
            alt.Tooltip('model_type', title='Model Type'),
            alt.Tooltip('group_size', title='Mean Group Size'), 
            alt.Tooltip('mean_bias', title=f'{bias_metric}'), 
            alt.Tooltip('mean_perf', title=f'{performance_metric}{parameter}')
            ],
        ).interactive()


        error_x = alt.Chart(filtered_df).mark_errorbar(orient='horizontal', opacity=0.5).encode(
            x=alt.X('mean_perf', title=''),
            xError='sem_perf',
            y='mean_bias',
            color='model_group_id:N'
        )

        error_y = alt.Chart(filtered_df).mark_errorbar(orient='vertical', opacity=0.5).encode(
            y=alt.Y('mean_bias', title=''),
            yError='sem_bias',
            x='mean_perf',
            color='model_group_id:N'
        )

        parity_band = alt.Chart(filtered_df).transform_calculate(y_min='0.8', y_max='1.2').mark_rect(opacity=0.01, color='gray').encode(
            y='y_min:Q',
            y2='y_max:Q'
        )

        rule = alt.Chart(filtered_df).transform_calculate(y='1').mark_rule(
            strokeDash=[4, 4],  # Dashed line: [dash, gap]
            color='gray'
        ).encode(
            y='y:Q'
        )

        # TODO - add annotations to the plot
        # annotations = pd.DataFrame(
        #     {
        #         'x': [0.15, 0.15],
        #         'y': [0.3, 2],
        #         'label': ['Models favor reference group', 'Models favor protected group']
        #     }
        # )

        # texts = alt.Chart(annotations).mark_text(
        #     color='black',
        #     align='center',
        #     opacity=0.5
        # ).encode(
        #     x='x:Q',
        #     y='y:Q',
        #     text='label:N'
        # )

        chart = (parity_band + rule + points + error_x + error_y).properties(
            width=300,
            title=''
        )
        
        return chart
