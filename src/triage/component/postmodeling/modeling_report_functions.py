
""" This is a module for moving the ad-hoc code we wrote in generating the modeling report"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

from triage.component.timechop.plotting import visualize_chops_plotly
from triage.component.timechop import Timechop


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


def plot_performance_all_models(engine, experiment_hashes, metric, parameter):
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
                distinct model_id, train_end_time, model_group_id, model_type, hyperparameters
            from triage_metadata.experiment_models join triage_metadata.models using(model_hash)
            where experiment_hash in ('{"','".join(experiment_hashes)}')   
        )
        select 
            model_id, 
            train_end_time,
            model_type, 
            stochastic_value as metric_value
        from models m left join test_results.evaluations e 
        on m.model_id = e.model_id
        and e.metric = '{metric}'
        and e.parameter = '{parameter}'
        and e.subset_hash = ''
    '''
    
    df = pd.read_sql(q, engine)
    
    fig, ax = plt.subplots(figsize=(6,3), dpi=150)
    
    sns.lineplot(
        data=df,
        x='train_end_time',
        y='metric_value',
        hue='model_type',
        alpha=0.7,
        ax=ax
    )
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Model Performance Over Time - {metric}{parameter}')    
    
    sns.despine()
    
def summarize_all_model_performance():
    pass
    
    
def model_selection():
    pass


def plot_performance_against_bias(engine, experiment_hashes, performance_metric):
    pass