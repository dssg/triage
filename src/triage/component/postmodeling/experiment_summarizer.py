
""" This is a module for moving the ad-hoc code we wrote in generating the modeling report"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from matplotlib.lines import Line2D

import warnings

from triage.component.timechop.plotting import visualize_chops_plotly
from triage.component.timechop import Timechop
from triage.component.audition.plotting import plot_cats

from triage.component.postmodeling.base import SingleModelAnalyzer
# from triage.component.postmodeling.postmodeling_analyzer import PostmodelingAnalyzer

model_name_abbrev = {
    'DummyClassifier': 'Dummy',
    'RandomForestClassifier': 'RF',
    'DecisionTreeClassifier': 'DT',
    'BaselineRankMultiFeature': 'Ranker',
    'ScaledLogisticRegression': 'LR',
    'XGBClassifier': 'XGB',
    'LightGBMClassifier':'LGBM'
}

def _format_model_name(long_name, model_group_id):
    
    # Get the child name fromt he import path
    child_name = long_name.split('.')[-1] 
    
    if child_name in model_name_abbrev:
        return str(model_group_id) + ': ' + model_name_abbrev[child_name]
    else:
        return str(model_group_id) + ': ' + child_name


def list_all_experiments(engine):
    """Generates a dataframe that lists the 
        experiment hash, last experiment run time, model comment
        for all experiments that have completed so far

    Args:
        engine (SQLAlchemy): Database engine
    """
    
    q = '''
        select 
            id,
            run_hash as experiment_hash,
            max(start_time) as most_recent_start_time,
            max(e.config ->> 'model_comment') as model_comment
        from triage_metadata.triage_runs tr join triage_metadata.experiments e on
        tr.run_hash = e.experiment_hash
        where current_status = 'completed'
        group by 1 order by 1 desc
    '''
    
    return pd.read_sql(q, engine)

def load_config(engine, experiment_hash):
    "return the experiment config"
    
    q = f'''
        select 
        config 
        from triage_metadata.experiments
        where experiment_hash = '{experiment_hash}'
    '''
    
    return pd.read_sql(q, engine).config.at[0]
    

def load_report_parameters_from_config(engine, experiment_hash):
    config = load_config(engine, experiment_hash=experiment_hash)
    
    d = dict()
    d['performance_metric'] = config['scoring'].get('priority_metric')
    d['threshold'] = config['scoring'].get('priority_parameter')
    
    bias_audit = config.get('bias_audit')
    d['bias_metric'] = None
    d['priority_groups'] = None 
    
    if bias_audit is not None:
        d['bias_metric'] = bias_audit.get('priority_metric')
        d['priority_groups'] = bias_audit.get('priority_groups')
    
    return d 
  
def get_most_recent_experiment_hash(engine):
    q = '''
        select 
            run_hash as experiment_hash
        from triage_metadata.triage_runs
        where current_status = 'completed'
        order by start_time desc       
    '''
    experiment_hash = pd.read_sql(q, engine)['experiment_hash'].iloc[0]
    logging.info(f'Using the experiment hash: {experiment_hash}')
    
    return experiment_hash







class ExperimentReport:
    
    def __init__(self, engine, experiment_hashes, performance_priority_metric, threshold, bias_priority_metric, bias_priority_groups):
        self.engine = engine
        self.experiment_hashes = experiment_hashes
        
        # TODO - consider expanding to multiple metrics and/or thresholds
        self.performance_metric = performance_priority_metric
        self.threshold = threshold
        self.bias_metric = bias_priority_metric
        self.bias_groups = bias_priority_groups
    
    def timesplits(self):
        """Generate an interactive plot of the time splits used for cross validation

        Args:
            engine (SQLAlchemy): DB engine
            experiment_hash (str): Experiment hash we are interested in  
        """
        
        q = f'''
            select 
                config
            from triage_metadata.experiments where experiment_hash = '{self.experiment_hashes[0]}'
        '''

        experiment_config = pd.read_sql(q, self.engine).at[0, 'config']
        
        
        chops = Timechop(**experiment_config['temporal_config'])
        splits = len(chops.chop_time())

        print(f'There are {splits} train-validation splits')
        visualize_chops_plotly(
            chops
        )
           
    def cohorts(self, generate_plots=True):
        """Generate a summary of cohorts (size, baserate)

        Args:
            engine (SQLAlchemy): Database engine
            experiment_hash (str): a list of experiment hashes
        """
        
        get_labels_table_query = f"""
        select distinct labels_table_name from triage_metadata.triage_runs 
        where run_hash = '{self.experiment_hashes[0]}'
        """

        labels_table = pd.read_sql(get_labels_table_query, self.engine).labels_table_name.iloc[0]
        # print(labels_table)
        cohort_query = f"""
            select 
            label_name, 
            label_timespan, 
            as_of_date, 
            count(distinct entity_id) as cohort_size, 
            avg(label) as baserate 
            from public.{labels_table}
            group by 1,2,3 order by 1,2,3
        """

        df = pd.read_sql(cohort_query, self.engine)
        
        if generate_plots:
            fig, ax1 = plt.subplots(figsize=(7, 3), dpi=100)

            color='darkblue'
            sns.lineplot(
                data=df,
                x='as_of_date',
                y='cohort_size',
                ax=ax1,
                label='',
                color=color
            )

            ax1.axhline(y=df.cohort_size.mean(), color=color, alpha=0.4, linestyle='--', label='mean cohort size')
            ax1.set_title('Cohort Size and Baserate Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Cohort Size', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color='firebrick'

            sns.lineplot(
                data=df,
                x='as_of_date',
                y='baserate',
                ax=ax2,
                color=color
            )
            ax2.set_ylabel('Baserate (%)', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.axhline(y=df.baserate.mean(), color=color, alpha=0.4, linestyle='--')
                    
        return df

    def subsets(self):
        q = f'''
            select 
                distinct s.subset_hash, s.config, s.config ->> 'name' as subset_name,
                s.config ->> 'name'::text||'_'||s.subset_hash as table_name
            from triage_metadata.experiment_models join triage_metadata.models using(model_hash)
                left join test_results.evaluations e using(model_id)
                    inner join triage_metadata.subsets s on e.subset_hash = s.subset_hash 
            where experiment_hash in ('{"','".join(self.experiment_hashes)}')
        '''
        
        df = pd.read_sql(q, self.engine)
        
        df['table_name'] = df.table_name.apply(lambda x: f'subet_{x}')

        # TODO - add code to plot subset size relative to the cohort size
            
        return df.set_index('subset_hash')
        
    def model_groups(self):
        """Generate a summary of all the model groups (types and hyperparameters) built in the experiment

        Args:
            engine (SQLAlchemy): Database engine
            experiment_hashes (List[str]): List of experiment hashes
        """
                
        mg_query_selected_exp = ''' 
            select 
                model_group_id, 
                model_type, 
                hyperparameters,
                count(distinct model_id) as num_models, 
                count(distinct train_end_time) as num_time_splits
            from triage_metadata.experiment_models join triage_metadata.models using(model_hash)
            where experiment_hash in ('{}')
            group by 1, 2, 3
            order by 2
        '''

        model_groups = pd.read_sql(mg_query_selected_exp.format("','".join(self.experiment_hashes)), self.engine)
        
        #TODO - highlight model groups that don't have the "correct" number of models built. 
        # Some model objects can be missing, and some train_end_times can have multiple models
    
        return model_groups
    
    def models(self):
        """ List all of the models built in the experiment
        
        Args:
            engine (SQLAlchemy): Database engine
            experiment_hashes (List[str]): List of experiment hashes
        """
        
        q = f''' 
            select 
                model_id,
                model_group_id,
                model_type,
                hyperparameters,
                train_end_time
            from triage_metadata.experiment_models join triage_metadata.models using(model_hash)
            where experiment_hash in ('{"','".join(self.experiment_hashes)}')
            order by model_type
            '''

        df = pd.read_sql(q, self.engine)

        return df.sort_values(by=['model_group_id', 'train_end_time'])      
       
     
    def features(self):
        """ Generate a dataframe containing all the features that were built in an experimet
    
        Args:
            engine (SQLAlchemy): Database engine
            experiment_hash (str): Hash of the experiment 
                Limiting this to one experiment hash based on the assumption that if the feature 
                spaces are different the experiment hashes shouldn't be lumped together 
        """
        
        q = f'''
            select 
                config
            from triage_metadata.experiments where experiment_hash = '{self.experiment_hashes[0]}'
        '''

        experiment_config = pd.read_sql(q, self.engine).at[0, 'config']
        
        all_features = list()

        for fg in experiment_config['feature_aggregations']:
            
            for agg in fg.get('aggregates', {}):
                d = dict()
                d['feature_group'] = fg['prefix']
                if isinstance(agg['quantity'], dict):
                    d['feature_name'] = list(agg['quantity'].keys())[0]
                    
                else:
                    d['feature_name'] = agg['quantity']
            
                d['metrics'] = ', '.join(agg['metrics'])
                d['time_horizons'] = ', '.join(fg['intervals'])
                d['feature_type'] = 'continuous_aggregate'
                
                if agg.get('imputation') is None: 
                    imp = fg.get('aggregates_imputation')
                else:
                    imp = agg.get('imputation')
                d['imputation'] = json.dumps(imp)
                # d['num_predictors'] = len(agg['metrics']) * len(fg['intervals'])
                all_features.append(d)
                
            for cat in fg.get('categoricals', {}):
                d = dict()
                d['feature_group'] = fg['prefix']
                d['feature_name'] = cat['column']
                d['metrics'] = ', '.join(cat['metrics'])
                d['time_horizons'] = ', '.join(fg['intervals'])
                d['feature_type'] = 'categorical_aggregate'
                
                if agg.get('imputation') is None: 
                    imp = fg.get('categoricals_imputation')
                else:
                    imp = agg.get('imputation')
                d['imputation'] = json.dumps(imp)
                # d['num_predictors'] = 'number of categories'
                all_features.append(d)
                
        all_features = pd.DataFrame(all_features).sort_values('feature_group', ignore_index=True)
        
        return all_features

    def feature_missingness(self):
        """
        Generates the mean, min, max missingness degree for each feature. 
        Assumes that the current "features" schema holds the relevant features for the experiment
        """
        
        # TODO -- This code is not correct. This looks at all the feature tables rather than the config. 
        q = '''
        select 
        table_name 
        from information_schema.tables
        where table_schema = 'features'
        and table_name like '%%aggregation_imputed'
        '''

        feature_tables = pd.read_sql(q, self.engine)['table_name'].tolist()
        
        logging.info(f'Printing only features with missing values')
        
        column_names = dict()
        for table in feature_tables:
            q = f'''
                select 
                column_name
                FROM information_schema.columns
                WHERE table_schema = 'features'
                AND table_name   = '{table}'
                and (column_name like '%%_imp')   
            '''

            column_names[table] = pd.read_sql(q, self.engine).column_name.tolist()
            
        results = pd.DataFrame()

        for table_name, columns in column_names.items():
            # print(table_name)
            select_clause = '''
                select 
                as_of_date,
                count(distinct entity_id) as cohort_size
            '''
            imputation_counts = ''
            for col in columns:
                imputation_counts += f'''
                    ,(sum("{col}")::float / count(distinct entity_id)) * 100  as "{col[:-4]}"
                '''
            
            from_clause = f'''
                from features.{table_name}
                group by 1
            '''
            
            query = select_clause + imputation_counts + from_clause
            df = pd.read_sql(query, self.engine).set_index(['as_of_date', 'cohort_size'])

            if results.empty:
                results = df
            else:
                results = results.join(df)
                
        df = pd.concat([results.mean(), results.min(), results.max()], axis=1).fillna(0)
        df.columns = ['mean (%)', 'min (%)', 'max (%)']

        # df.style.applymap(lambda x: 'background-color : pink' if x>80 else '')
        df_ = df[df['mean (%)'] > 0]

        return df_.sort_values(by="mean (%)", ascending=False)
    
    
    def model_performance(self, metric=None, parameter=None, generate_plot=True):
        """ Generate an Audition type plot to display predictive performance over time for all model groups
        
        Args:
            metric (str): The metric we are intersted in (suppores 'precision@', 'recall@', 'auc_roc')
            parameter (str): The threshold, supports percentile ('_pct') and absolute ('_abs') thresholds
        """
        
        if metric is None:
            metric = self.performance_metric
        
        if parameter is None: 
            parameter = self.threshold
        
        # fetch model groups
        q = f'''
            with models as (
                select 
                    distinct model_id, 
                    train_end_time, 
                    model_group_id, 
                    model_type, 
                    hyperparameters
                from triage_metadata.experiment_models join triage_metadata.models using(model_hash)
                where experiment_hash in ('{"','".join(self.experiment_hashes)}')   
            )
            select 
                m.model_id, 
                train_end_time::date as train_end_time_dt,
                to_char(train_end_time, 'YYYY-MM-DD') as train_end_time,
                model_type, 
                model_group_id,
                stochastic_value as metric_value
            from models m left join test_results.evaluations e 
            on m.model_id = e.model_id
            and e.metric = '{metric}'
            and e.parameter = '{parameter}'
            and e.subset_hash = ''
        '''


        df = pd.read_sql(q, self.engine)
        df['train_end_time'] = pd.to_datetime(df.train_end_time, format='%Y-%m-%d')
        
        models_per_train_end_time = df.groupby(['model_group_id', 'train_end_time']).count()['model_id']
        
        if models_per_train_end_time[models_per_train_end_time > 1].empty:
            pass 
        else: 
            print(f'model groups with morel than one model id per train end time: \n{models_per_train_end_time[models_per_train_end_time > 1]}' )
        
        if generate_plot:
            # using the audition plotting base here
            plot_cats(
                frame=df,
                x_col='train_end_time',
                y_col='metric_value',
                cat_col='model_type',
                grp_col='model_group_id',
                highlight_grp=None,
                title=f'Model Performance Over Time - {metric}{parameter}',
                x_label='Time',
                y_label=f'Value - {metric}{parameter}',
                cmap_name="tab10",
                figsize=[12, 4],
                dpi=150,
                x_ticks=list(df.train_end_time.unique()),
                y_ticks=None,
                x_lim=None,
                y_lim=(0, 1.1),
                legend_loc=None,
                legend_fontsize=12,
                label_fontsize=12,
                title_fontsize=12,
                label_fcn=None,
                path_to_save=None,
                alpha=0.4,
                colordict=None,
                styledict=None
            )
            sns.despine()     
        
        # sort df by train_end_time
        return df.sort_values(by="train_end_time")
    
    
    def model_performance_subsets(self, metric=None, parameter=None, generate_plot=True):
        
        if metric is None:
            metric = self.performance_metric
        
        if parameter is None: 
            parameter = self.threshold
        
        q = f'''
            select 
                case when e.subset_hash is null then 'full_cohort' 
                else s.config ->> 'name' 
                end as "subset",
                e.subset_hash,
                m.model_id,
                m.model_group_id,
                m.model_type,
                m.train_end_time::date,
                e.stochastic_value as metric_value
            from triage_metadata.experiment_models join triage_metadata.models m using(model_hash)
                left join test_results.evaluations e
                on m.model_id = e.model_id
                and e.parameter = '{parameter}'
                and e.metric = '{metric}'
                    left join triage_metadata.subsets s on e.subset_hash = s.subset_hash 
            where experiment_hash in ('{"','".join(self.experiment_hashes)}')
        '''
        
        df = pd.read_sql(q, self.engine)
        
        if (df.empty) or (None in df.subset.unique()):
            return None
        
        df['model_type_child'] = df.apply(lambda x: _format_model_name(x['model_type'], x['model_group_id']), axis=1)
        df['model_type_short'] = df.apply(lambda x: x['model_type'].split('.')[-1], axis=1)
        # df.apply(lambda x: _format_model_name(x['model_type']) + ': ' + str(x['model_group_id']), axis=1) 
        
        
        get_labels_table_query = f"""
        select distinct labels_table_name from triage_metadata.triage_runs 
        where run_hash = '{self.experiment_hashes[0]}'
        """

        labels_table = pd.read_sql(get_labels_table_query, self.engine).labels_table_name.iloc[0]

        if generate_plot:        
            grpobj = df.groupby('subset')

            for grp, gdf in grpobj:
                fig, axes = plt.subplots(1, 2, figsize=(10,4), dpi=100)
                fig.suptitle(f'Subset Name: {grp}')
                
                # Plotting the subsetsize and the baserate
                subset_table_name = 'subset_' + gdf['subset'].iloc[0] + '_' + gdf['subset_hash'].iloc[0]
                q = f'''
                    select 
                    as_of_date::date, count(entity_id) as subset_size, avg(label)*100 as baserate
                    from {labels_table} inner join {subset_table_name} using(entity_id, as_of_date)
                    group by 1
                    order by 1
                '''
                subset_size = pd.read_sql(q, self.engine)
                
                # Subset sizes over time
                color='tab:blue'
                sns.barplot(
                    data=subset_size,
                    x='as_of_date',
                    y='subset_size',
                    ax=axes[0],
                    alpha=0.5,
                    color=color
                )
                axes[0].tick_params(axis='x', rotation=90)
                axes[0].set_title(f'Subset Size & Baserate')
                axes[0].set_xlabel('Time')
                axes[0].set_ylabel('Subset size', color=color)
                axes[0].tick_params(axis='y', labelcolor=color)
                axes[0].axhline(y=subset_size.subset_size.mean(), color=color, alpha=0.4, linestyle='--')
                
                
                ax2 = axes[0].twinx()
                color='tab:red'
                
                sns.lineplot(
                    data=subset_size,
                    x=axes[0].get_xticks(),
                    y='baserate',
                    ax=ax2,
                    alpha=0.6,
                    marker='o',
                    markersize=5,
                    color=color
                )
                ax2.set_ylabel('Baserate (%)', color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.axhline(y=subset_size.baserate.mean(), color=color, alpha=0.4, linestyle='--')
                
                colors={x: sns.color_palette().as_hex()[i] for i, x in enumerate(sorted(df.model_type_short.unique()))}
                legend_handles=[ Line2D([0], [0], label=k, color=v) for k, v in colors.items() ]

                model_groups = gdf.groupby('model_group_id')
                
                for mod_type, model_grp_df in model_groups:
                    model_grp_df = model_grp_df.sort_values('train_end_time')

                    # print(model_grp_df.head())
                    model_type = model_grp_df.model_type_short.iloc[0]
                    
                    model_grp_df.plot(x='train_end_time', y='metric_value', color=colors[model_type], ax=axes[1], alpha=0.4)
                    
                    # ticklabels = list(gdf.train_end_time.unique())
                    # axes[1].set_xticklabels(ticklabels, rotation=45)
                    axes[1].legend(handles=legend_handles, loc='upper left', frameon=False, ncol=1, fontsize='small', bbox_to_anchor=[1, 1])
                    axes[1].tick_params(axis='x', rotation=90)
                    axes[1].set_ylabel(f'{metric}{parameter}')
                    axes[1].set_xlabel('Time')
                    # axes[1].set_ylim(0, 0.3)
                    axes[1].set_title(f'Model Performance')    
                
                plt.tight_layout()

        # sort df by train_end_time     
        return df.sort_values(by="train_end_time")


    def efficiency_and_equity(self, efficiency_metric=None, equity_metric=None, parameter=None, groups=None, model_group_ids=None, bias_metric_tolerance=0.2, generate_plot=True):
        ''' Plot the performanc metric against the bias metric for all or selected models.
            Args:
                
        '''
        
        if efficiency_metric is None:
            efficiency_metric = self.performance_metric
        
        if equity_metric is None:
            equity_metric = self.bias_metric
            
        if parameter is None:
            parameter = self.threshold
            
        if groups is None:
            groups = self.bias_groups

            # check if there are bias audit results in the DB (directly from aequitas table)
            q = f'''
                with attr_values as (
                    select distinct 
                        attribute_name,
                        attribute_value
                    from triage_metadata.experiment_models a 
                    join triage_metadata.models b
                    using(model_hash)
                    join test_results.aequitas c 
                    using (model_id)
                    where experiment_hash = '{self.experiment_hashes[0]}'
                )

                select distinct 
                    attribute_name,
                    attribute_value
                from attr_values
                order by 1,2
            '''
            
            rg = pd.read_sql(q, self.engine)
            
            # if there are no bias audit results in the DB
            if rg.empty:
                logging.warning('No bias audit config or aequitas calculation was not completed! check the test_results.aequitas table. No plots generated')
                return
            else: 
                groups = dict()
                for attr, gdf in rg.groupby('attribute_name'):
                    groups[attr] = list(gdf['attribute_value'].unique())
            
        if model_group_ids is None: 
            # logging.warning('No model groups specified. Usign all model group ids')
            model_group_ids = self.model_groups().model_group_id.tolist()
            
            if not model_group_ids:
                logging.warning('No model groups belong to the experiment! Returning None')
                return None
            
        # If no groups are specified, we show results for all groups    
        if groups is None:
            
            # logging.info('No groups are specified. Showing results for all attributes and their values')
            groups = dict()
            # we can't trust on what is defined in the config, we take the values directly from the aequitas table
            q = f'''
                with attr_values as (
                        select distinct 
                            attribute_name,
                            attribute_value
                        from triage_metadata.experiment_models a 
                        join triage_metadata.models b
                        using(model_hash)
                        join test_results.aequitas c 
                        using (model_id)
                        where experiment_hash = '{self.experiment_hashes[0]}'
                    )

                select distinct 
                    attribute_name,
                    attribute_value
                from attr_values
                order by 1,2
            '''
            
            rg = pd.read_sql(q, self.engine)
            
            if rg.empty:
                logging.warning('No bias audit config or aequitas calculation was not completed! check the test_results.aequitas table. No plots generated')
                return
            
            groups = dict()
            for attr, gdf in rg.groupby('attribute_name'):
                groups[attr] = list(gdf['attribute_value'].unique())
                
            logging.debug(f'Plotting bias for following groups and values {rg}')
            
        
        parameter_ae = parameter
        
        if 'pct' in parameter:
            t = round(float(parameter.split('_')[0]) / 100, 2)
            parameter_ae =  f'{t}_pct'
            
        attributes = groups.keys()
        attribute_values = [v for v_list in groups.values() for v in v_list]
            
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
                    e.stochastic_value as "{efficiency_metric}{parameter}",
                    a.tpr,
                    a.{equity_metric},
                    a.attribute_name,
                    a.attribute_value,
                    a.group_size,
                    a.prev as baserate,
                    total_entities
                from triage_metadata.experiment_models em join triage_metadata.models m 
                    on em.model_hash = m.model_hash
                    and m.model_group_id in ({", ".join([str(x) for x in model_group_ids])})
                    left join test_results.evaluations e 
                        on m.model_id = e.model_id 
                        and e.metric = '{efficiency_metric}'
                        and e."parameter" = '{parameter}'
                        and e.subset_hash = ''
                            left join test_results.aequitas a 
                                on m.model_id = a.model_id 
                                and a."parameter" = '{parameter_ae}'
                                and a.attribute_name in ('{"','".join(attributes)}')
                                and a.attribute_value in ('{"','".join(attribute_values)}')
                                and a.tie_breaker= 'worst' 
                where experiment_hash in  ('{"','".join(self.experiment_hashes)}')
                
        '''
        
        #and a.attribute_value in ('{', '.join(attribute_values)}')
        metrics = pd.read_sql(q, self.engine)

        # metrics['Model Class'] = metrics['model_type'].apply(lambda x: x.split('.')[-1])
        # metrics['model_label'] = metrics.apply(lambda x: f"{x['model_group_id']}: {x['model_type'].split('.')[-1]}", axis=1)
        metrics['model_label'] = metrics.apply(lambda x: _format_model_name(x['model_type'], x['model_group_id']), axis=1)
        metrics['model_type_short'] = metrics.apply(lambda x: x['model_type'].split('.')[-1], axis=1)
        
        if generate_plot: 
            # Metric means
            mean = metrics.groupby(['model_label', 'attribute_value', 'model_type_short'])[[f'{efficiency_metric}{parameter}', f'{equity_metric}']].mean().reset_index().sort_values('model_label')
            
            # Metric standard errors
            sem = metrics.groupby(['model_label', 'attribute_value', 'model_type_short'])[[f'{efficiency_metric}{parameter}', f'{equity_metric}']].sem().reset_index().sort_values('model_label')
            labels = sorted(mean.model_label.unique())
            
            # n_attrs = sum([len(x) for x in groups.values()])
            n_attrs = len(attribute_values)
            ax_cntr = 0
            # bias_tolerance = 0.2
            fig, axes = plt.subplots(1, n_attrs, figsize=(4*n_attrs + 1, 4), sharey=True, sharex=True, dpi=100)
            # colors=sns.color_palette().as_hex()[:len(mean.model_label.unique())]
            colors={x: sns.color_palette().as_hex()[i] for i, x in enumerate(sorted(metrics.model_type_short.unique()))}
            legend_handles=[ Line2D([0], [0], label=k, marker='o', color=v) for k, v in colors.items() ]

            
            for group, attrs in groups.items():
                for attr in attrs:
                    msk = mean['attribute_value'] == attr
                    x = mean[msk][f'{efficiency_metric}{parameter}'].tolist()
                    y = mean[msk][f'{equity_metric}'].tolist()

                    # mean[msk]['model_type'].tolist()

                    msk = sem['attribute_value'] == attr
                    yerr = sem[msk][f'{equity_metric}'].tolist()
                    xerr = sem[msk][f'{efficiency_metric}{parameter}'].tolist()

                    # print(x)
                    
                    for i in range(len(x)):
                        axes[ax_cntr].errorbar(x[i], y[i], yerr[i], xerr[i], fmt=' ', linewidth=1, capsize=2, color=colors[mean[msk]['model_type_short'].iloc[i]], alpha=0.3)
                        axes[ax_cntr].scatter(x[i], y[i], color=colors[mean[msk]['model_type_short'].iloc[i]], label=labels[i], alpha=0.5)
                        # axes[ax_cntr].set(title=f'{group} | {attr}', xlabel='Performance Metric', ylabel='Bias Metric', ylim=[0, 3])
                        axes[ax_cntr].set(title=f'{group} | {attr}', xlabel='Efficiency Metric', ylabel='Equity Metric')
                        axes[ax_cntr].axhline(y=1, color='dimgray', linestyle='--', alpha=0.1)
                        axes[ax_cntr].axhline(y=1+bias_metric_tolerance, color='dimgray', linestyle=':', alpha=0.09)
                        axes[ax_cntr].axhline(y=1-bias_metric_tolerance, color='dimgray', linestyle=':', alpha=0.09)
                    ax_cntr += 1
                axes[-1].legend(handles=legend_handles, bbox_to_anchor=(1,1), loc='upper left', frameon=False)
                sns.despine()
            
            # Plotting the Group sizes and baserates
            fig, axes = plt.subplots(1, n_attrs, figsize=(4*n_attrs + 1, 4), sharex=True, sharey=True, dpi=100)
            ax_cntr=0
            for group, attrs in groups.items():
                for attr in attrs:
                    msk = metrics['attribute_value'] == attr

                    grouped = metrics[msk].groupby('train_end_time')[['baserate', 'group_size', 'total_entities']].mean().reset_index()
                    color='tab:blue'
                    sns.barplot(
                        data=grouped,
                        x='train_end_time',
                        y='group_size',
                        ax=axes[ax_cntr],
                        alpha=0.8,
                        color=color,
                        label='Group Size'
                    )

                    sns.barplot(
                        data=grouped,
                        x='train_end_time',
                        y='total_entities',
                        ax=axes[ax_cntr],
                        alpha=0.2,
                        color='gray',
                        label='Cohort Size'
                    )

                    axes[ax_cntr].tick_params(axis='x', rotation=90)
                    axes[ax_cntr].set(xlabel='Time', title=f'{group} | {attr}')
                    axes[ax_cntr].set_ylabel('Entities', color=color)
                    axes[ax_cntr].tick_params(axis='y', labelcolor=color)

                    ax2 = axes[ax_cntr].twinx()
                    color='tab:red'

                    sns.lineplot(
                        data=grouped,
                        x=axes[ax_cntr].get_xticks(),
                        y='baserate',
                        ax=ax2,
                        alpha=0.6,
                        marker='o',
                        markersize=5,
                        color=color,
                        # label='Baserate'
                    )
                    ax2.set_ylabel('Baserate (%)', color=color)
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax_cntr += 1
            axes[-1].legend(bbox_to_anchor=(1.1,1.1), loc='upper left', frameon=False)
            plt.tight_layout()
        
        return metrics
    
      
    def experiment_stats(self):
        q = f'''
            select 
                max(time_splits) as timesplits_from_temporal_config,
                max(matrices_needed/2) as validation_splits,
                max(as_of_times) as as_of_times,
                max(feature_blocks) as feature_groups,
                max(total_features) as features,
                max(matrices_needed) as matrices_needed,
                sum(grid_size) as grid_size,
                sum(models_needed) as models_needed,
                max(case when (matrices_needed/2) < time_splits then 1 else 0 end) as implemented_fewer_splits
            from triage_metadata.experiments
            where experiment_hash in ('{"','".join(self.experiment_hashes)}')
        '''
        
        return pd.read_sql(q, self.engine).to_dict(orient='records')[0]
    
          
    def generate_summary(self, metric=None, parameter=None, equity_metric=None):
        '''
        Summarize:
         - Number of time splits
         - Number of as of dates
         - Average cohort size and baserate
         - Number of features and feature groups
         - Number of model types, model groups, and models built
         - Number of model groups that are missing models 
         - Best model performance and the model group
         - model performance for subsets
         - Best model's bias metric, and whether other models are within the bias metric threshold
        
        '''
        
        if metric is None: 
            metric = self.performance_metric
        
        if parameter is None: 
            parameter = self.threshold
            
        if equity_metric is None:
            equity_metric = self.bias_metric
        
        
        stats = self.experiment_stats()
          
        if stats['implemented_fewer_splits'] == 1:
            print(f"Temporal config suggests {stats['timesplits_from_temporal_config']} temporal splits, but experiment implemented only {stats['validation_splits']} splits. Was this intentional?")
        else:
            print(f'Experiment contained {stats["timesplits_from_temporal_config"]} temporal splits')
            
        
        print(f"Experiment contained {stats['as_of_times']} distinct as_of_times")
    
        cohorts = self.cohorts(generate_plots=False)
        print(f'On average, your cohorts contained around {round(cohorts.cohort_size.mean())} entities with a baserate of {round(cohorts.baserate.mean(), 3)}')
    
        print(f"You built {stats['features']} features organized into {stats['feature_groups']} groups/blocks")
        
        print(f"Your model grid specification contained {stats['grid_size']} model types with {stats['models_needed']} individual models")
        
        ## Models
        num_models = len(self.models())
        if num_models < stats['models_needed']:
            print(f"However, the experiment only built {num_models} models. You are missing {stats['models_needed'] - num_models} models")
            
        else:
            print(f"You successfully built all the {num_models} models")
        
        # Model Performance
        performance = self.model_performance(metric=metric, parameter=parameter, generate_plot=False)
        best_performance = performance.groupby(['model_group_id', 'model_type'])['metric_value'].mean().max()
        best_model_group = performance.groupby(['model_group_id', 'model_type'])['metric_value'].mean().idxmax()[0]
        best_model_type = performance.groupby(['model_group_id', 'model_type'])['metric_value'].mean().idxmax()[1]
            
        print(f"Your models acheived a best average {metric}{parameter} of {round(best_performance, 3)} over the {stats['validation_splits']} validation splits, with the Model Group {best_model_group},{best_model_type}. Note that model selection is more nuanced than average predictive performance over time. You could use Audition for model selection.")
        
        ## Subsets
        subset_performance = self.model_performance_subsets(metric=metric, parameter=parameter, generate_plot=False)
        if subset_performance is not None:
            grpobj = subset_performance.groupby('subset')
            res = []
            for subset, gdf in grpobj:
                d = dict()
                d['subset'] = subset
                d['best_perf'] = round(gdf.groupby(['model_group_id', 'model_type'])['metric_value'].mean().max(),3)
                d['best_mod'] = gdf.groupby(['model_group_id', 'model_type'])['metric_value'].mean().idxmax()

                res.append(d)
            
            if len(res) > 0:
                print(f"You created {len(res)} subsets of your cohort -- {', '.join([x['subset'] for x in res])}")
                for d in res:
                    print(f"For subset '{d['subset'] }', Model Group {d['best_mod'][0]}, {d['best_mod'][1]} achieved the best average {metric}{parameter} of {d['best_perf']}")
        else:
            print("No subsets defined.") 

        ## Bias
        equity_metrics = self.efficiency_and_equity(
            efficiency_metric=metric,
            equity_metric=equity_metric,
            parameter=parameter,
            generate_plot=False
        )
        
        if equity_metrics is not None:
            grpobj = equity_metrics[(equity_metrics.baserate > 0) & (equity_metrics.model_group_id == best_model_group)].groupby('attribute_name')
            for attr, gdf in grpobj:
                print(f'Measuring biases across {attr} groups using {equity_metric} for the best performing model:')
                d = gdf.groupby('attribute_value')[equity_metric].mean()
                print(", ".join(f"{k}: {round(v, 3)}" for k, v, in d.to_dict().items()))
        else:
            print(f"No bias audit results were found in the database for the experiment.")
            

    def precision_recall_curves(self, plot_size=(3,3)):
        
        n_splits = self.experiment_stats()['validation_splits']
        n_groups = len(self.model_groups())     
        
        # n_groups x n_splits grid 
        fig, axes = plt.subplots(
            n_groups, 
            n_splits,
            figsize=(n_splits*plot_size[0], n_groups*plot_size[1])
        )
        
        axes = axes.flatten()
        
        grp_obj = self.models().groupby('model_group_id')
        
        ax_idx = 0
        for _, gdf in grp_obj:
            model_ids = gdf.model_id.tolist()
            
            for mod_id in model_ids:
                tmp = SingleModelAnalyzer(engine=self.engine, model_id=mod_id)
                tmp.plot_precision_recall_curve(
                    ax=axes[ax_idx]
                )
                ax_idx += 1
                
            # Making sure that models that aren't built are skipped in the grid
            if ax_idx % n_splits > 0:
                ax_idx += (n_splits - (ax_idx % n_splits)) 

        plt.show()
        plt.tight_layout()


    def get_best_hp_config_for_each_model_type(self):
        
        q = f'''
            with avg_perf as (
                select 
                    model_group_id, model_type, hyperparameters, avg(stochastic_value) as mean_performance
                from triage_metadata.experiment_models join triage_metadata.models m using(model_hash)
                    left join test_results.evaluations e
                    on m.model_id = e.model_id
                    and e.metric = '{self.performance_metric}'
                    and e.parameter = '{self.threshold}'
                    and e.subset_hash = ''
                where experiment_hash in ('{"','".join(self.experiment_hashes)}') 
                and model_type not like '%%Dummy%%'
                group by 1, 2, 3    
            )
            select distinct on(model_type)
            model_group_id, model_type, hyperparameters, mean_performance
            from avg_perf
            order by model_type, mean_performance desc
        '''

        best_models = pd.read_sql(q, self.engine).set_index('model_group_id').sort_values(by='mean_performance', ascending=False)
        best_models['model_type'] = best_models['model_type'].str.split('.').apply(lambda x: x[-1])
            
        return best_models
    
    
    def model_groups_w_best_mean_performance(self, n_model_groups=5):
        """ Return the model groups with the best mean performance """
        
        q = f'''
            with models as (
                select 
                    distinct model_id, train_end_time, model_group_id, model_type, hyperparameters
                from triage_metadata.experiment_models join triage_metadata.models using(model_hash)
                where experiment_hash in ('{"','".join(self.experiment_hashes)}')   
            )
            select 
                m.model_group_id, 
                model_type, 
                hyperparameters,
                avg(stochastic_value) as mean_metric_value
            from models m left join test_results.evaluations e 
                on m.model_id = e.model_id
                and e.metric = '{self.performance_metric}'
                and e.parameter = '{self.threshold}'
                and e.subset_hash = ''
            group by 1, 2, 3
            limit {n_model_groups};
        '''
        
        df = pd.read_sql(q, self.engine)
        
        return df.model_group_id.tolist(), df
    

    def feature_importance(self, plot_size=(2,5), n_features=20):
        n_splits = self.experiment_stats()['validation_splits']
        n_groups = len(self.model_groups())     

        # n_groups x 1 grid 
        fig, axes = plt.subplots(
            n_groups,
            1,
            figsize=(plot_size[0], n_groups*plot_size[1])
        )

        axes = axes.flatten()

        grp_obj = self.models().groupby('model_group_id')

        ax_idx = 0
        for model_group_id, gdf in grp_obj:
            model_ids = gdf.model_id.tolist()

            feature_importances_group = list()
            for mod_id in model_ids:
                tmp = SingleModelAnalyzer(engine=self.engine, model_id=mod_id)

                fi = tmp.get_feature_importances(n_top_features=100)

                feature_importances_group.append(fi)

            feature_importances_group = pd.concat(feature_importances_group)
                        
            if len(feature_importances_group.feature.unique()) > 1: 
                agg_df = feature_importances_group.groupby('feature')['feature_importance'].agg(['mean', 'sem']).nlargest(n_features, 'mean').reset_index()
                
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sns.barplot(
                        data=agg_df,
                        x='mean',
                        xerr=agg_df['sem'],
                        y='feature',
                        color = 'royalblue',
                        capsize = 0.5,
                        ax = axes[ax_idx]
                    )
                    
                    axes[ax_idx].set_title(f'{_format_model_name(tmp.model_type, model_group_id)}')
                    axes[ax_idx].set_xlabel('')
                    axes[ax_idx].set_ylabel('')
                
            else:
                fig.delaxes(axes[ax_idx])
                logging.warning(f'Not a model with multiple features: {model_group_id} ')
                
            ax_idx += 1

        plt.show()
        plt.tight_layout()
        
    # def feature_group_importance(self):
        