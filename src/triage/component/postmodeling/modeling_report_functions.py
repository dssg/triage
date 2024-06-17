
""" This is a module for moving the ad-hoc code we wrote in generating the modeling report"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from triage.component.timechop.plotting import visualize_chops_plotly
from triage.component.timechop import Timechop
from triage.component.audition.plotting import plot_cats


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


def visualize_validation_splits(engine, experiment_hash):
    """Generate an interactive plot of the time splits used for cross validation

    Args:
        engine (SQLAlchemy): DB engine
        experiment_hash (str): Experiment hash we are interested in  
    """
    
    q = f'''
        select 
            config
        from triage_metadata.experiments where experiment_hash = '{experiment_hash}'
    '''

    experiment_config = pd.read_sql(q, engine).at[0, 'config']
    
    
    chops = Timechop(**experiment_config['temporal_config'])
    splits = len(chops.chop_time())

    print(f'There are {splits} train-validation splits')
    visualize_chops_plotly(
        chops
    )
    

def summarize_cohorts(engine, experiment_hash):
    """Generate a summary of cohorts (size, baserate)

    Args:
        engine (SQLAlchemy): Database engine
        experiment_hash (str): a list of experiment hashes
    """
    
    get_labels_table_query = f"""
    select distinct labels_table_name from triage_metadata.triage_runs 
    where run_hash = '{experiment_hash}'
    """

    labels_table = pd.read_sql(get_labels_table_query, engine).labels_table_name.iloc[0]
    print(labels_table)
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

    df = pd.read_sql(cohort_query, engine)
    
    return df

    # dfs = list()
    # for table in labels_tables:
    #     cohort_query = """
    #         select 
    #         label_name, 
    #         label_timespan, 
    #         as_of_date, 
    #         count(distinct entity_id) as cohort_size, 
    #         avg(label) as baserate 
    #         from public.{}
    #         group by 1,2,3 order by 1,2,3
    #     """
        
    #     df = pd.read_sql(cohort_query.format(table), engine)
        
    #     dfs.append(df)
        
    # return dfs[0].append(dfs[1:], ignore_index=True)


def summarize_model_groups(engine, experiment_hashes):
    """Generate a summary of all the model groups (types and hyperparameters) built in the experiment

    Args:
        engine (SQLAlchemy): Database engine
        experiment_hashes (List[str]): List of experiment hashes
    """
            
    mg_query_selected_exp = ''' 
        select 
            distinct model_group_id, model_type, hyperparameters
        from triage_metadata.experiment_models join triage_metadata.models using(model_hash)
        where experiment_hash in ('{}')
        order by 2
    '''

    model_groups = pd.read_sql(mg_query_selected_exp.format("','".join(experiment_hashes)), engine)
    
    return model_groups


def list_all_features(engine, experiment_hash):
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
        from triage_metadata.experiments where experiment_hash = '{experiment_hash}'
    '''

    experiment_config = pd.read_sql(q, engine).at[0, 'config']
    
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


def list_all_models(engine, experiment_hashes):
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
        where experiment_hash in ('{"','".join(experiment_hashes)}')
        order by model_type
        '''

    return pd.read_sql(q, engine)


def plot_performance_all_models(engine, experiment_hashes, metric, parameter, **kw):
    """ Generate an Audition type plot to display predictive performance over time for all model groups
    
    Args:
        engine (SQLAlchemy): Database engine
        experiment_hashes (List[str]): List of experiment hashes
        metric (str): The metric we are intersted in (suppores 'precision@', 'recall@', 'auc_roc')
        parameter (str): The threshold, supports percentile ('_pct') and absolute ('_abs') thresholds
    """
    
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
            where experiment_hash in ('{"','".join(experiment_hashes)}')   
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
    
    df = pd.read_sql(q, engine)
    df['train_end_time'] = pd.to_datetime(df.train_end_time, format='%Y-%m-%d')
    
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
    
    
def summarize_all_model_performance():
    pass
    

def list_all_subsets(engine, experiment_hashes):
    
    q = f'''
        select 
            distinct s.subset_hash, s.config, s.config ->> 'name' as subset_name,
            s.config ->> 'name'::text||'_'||s.subset_hash as table_name
        from triage_metadata.experiment_models join triage_metadata.models using(model_hash)
            left join test_results.evaluations e using(model_id)
                inner join triage_metadata.subsets s on e.subset_hash = s.subset_hash 
        where experiment_hash in ('{"','".join(experiment_hashes)}')
    '''
    
    df = pd.read_sql(q, engine)
    df['table_name'] = df.table_name.apply(lambda x: f'subet_{x}')

    # TODO - add code to plot subset size relative to the cohort size
        
    return df
        
    

def plot_subset_performance(engine, experiment_hashes, parameter, metric):

    q = f'''
        select 
            case when e.subset_hash is null then 'full_cohort' 
            else s.config ->> 'name' 
            end as "subset",
            e.subset_hash,
            m.model_id,
            m.model_group_id,
            m.model_type,
            m.train_end_time,
            e.stochastic_value as metric_value
        from triage_metadata.experiment_models join triage_metadata.models m using(model_hash)
            left join test_results.evaluations e
            on m.model_id = e.model_id
            and e.parameter = '{parameter}'
            and e.metric = '{metric}'
                left join triage_metadata.subsets s on e.subset_hash = s.subset_hash 
        where experiment_hash in ('{"','".join(experiment_hashes)}')
    '''
    
    df = pd.read_sql(q, engine)
    df['model_type_child'] = df.model_type.apply(lambda x: x['model_type'].split('.')[-1] + ': ' + str(x['model_group_id']), axis=1) 
    
    
    grpobj = df.groupby('subset')

    for grp, gdf in grpobj:
        fig, ax = plt.subplots(figsize=(8,3), dpi=100)
        sns.lineplot(
            data=gdf,
            x='train_end_time',
            y='metric_value',
            hue='model_type_child',
            # style='model_type_child',
            ax=ax,
            alpha=0.7
        )
        
        l = ax.legend(bbox_to_anchor=(1,1), loc='upper left', frameon=True)
        ax.set_ylabel(f'{metric}{parameter}')
        ax.set_xlabel('Time')
        ax.set_ylim(0, 0.3)
        ax.set_title(f'Model performance over cohort subset: {grp}')
        sns.despine()


    
def model_groups_w_best_mean_performance(engine, experiment_hashes, metric, parameter, n_model_groups):
    """ Return the model groups with the best mean performance """
    
    q = f'''
        with models as (
            select 
                distinct model_id, train_end_time, model_group_id, model_type, hyperparameters
            from triage_metadata.experiment_models join triage_metadata.models using(model_hash)
            where experiment_hash in ('{"','".join(experiment_hashes)}')   
        )
        select 
            m.model_group_id, 
            model_type, 
            hyperparameters,
            avg(stochastic_value) as mean_metric_value
        from models m left join test_results.evaluations e 
            on m.model_id = e.model_id
            and e.metric = '{metric}'
            and e.parameter = '{parameter}'
            and e.subset_hash = ''
        group by 1, 2, 3
        limit {n_model_groups};
    '''
    
    df = pd.read_sql(q, engine)
    
    return df.model_group_id.tolist(), df
    
    



def plot_performance_against_bias(engine, experiment_hashes, metric, parameter, bias_metric, groups, model_group_ids=None, selected_models=None):
    ''' Plot the performanc metric against the bias metric for all or selected models.
        Args:
            engine: DB connection
            experiment_hashes ([])
    '''
    
    if model_group_ids is None: 
        logging.warning('No model groups specified. Usign all model group ids')
        model_group_ids = summarize_model_groups(engine, experiment_hashes).model_group_id.tolist()
    
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
                e.stochastic_value as "{metric}{parameter}",
                a.tpr,
                a.{bias_metric},
                a.attribute_name,
                a.attribute_value
            from triage_metadata.experiment_models em join triage_metadata.models m 
                on em.model_hash = m.model_hash
                and m.model_group_id in ({", ".join([str(x) for x in model_group_ids])})
                left join test_results.evaluations e 
                    on m.model_id = e.model_id 
                    and e.metric = '{metric}'
                    and e."parameter" = '{parameter}'
                    and e.subset_hash = ''
                        left join test_results.aequitas a 
                            on m.model_id = a.model_id 
                            and a."parameter" = '{parameter_ae}'
                            and a.attribute_name in ('{', '.join(attributes)}')
                            and a.attribute_value in ('{', '.join(attribute_values)}')
                            and a.tie_breaker= 'worst' 
            where experiment_hash in  ('{"','".join(experiment_hashes)}')
            
    '''
    
    metrics = pd.read_sql(q, engine)
    
    metrics['Model Class'] = metrics['model_type'].apply(lambda x: x.split('.')[-1])
    
    