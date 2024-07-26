
""" This is a module for moving the ad-hoc code we wrote in generating the modeling report"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from matplotlib.lines import Line2D

from triage.component.timechop.plotting import visualize_chops_plotly
from triage.component.timechop import Timechop
from triage.component.audition.plotting import plot_cats
from triage.component.postmodeling.report_generator import PostmodelingReport


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
    

def load_config(engine, experiment_hash):
    "return the experiment config"
    
    q = f'''
        select 
        config 
        from triage_metadata.experiments
        where experiment_hash = '{experiment_hash}'
    '''
    
    return pd.read_sql(q, engine).config.at[0]
    

def summarize_cohorts(engine, experiment_hash, generate_plots=True):
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
    
    if df.empty:
        return None
    
    df['model_type_child'] = df.apply(lambda x: _format_model_name(x['model_type'], x['model_group_id']), axis=1)
    # df.apply(lambda x: _format_model_name(x['model_type']) + ': ' + str(x['model_group_id']), axis=1) 
    
    
    get_labels_table_query = f"""
    select distinct labels_table_name from triage_metadata.triage_runs 
    where run_hash = '{experiment_hashes[0]}'
    """

    labels_table = pd.read_sql(get_labels_table_query, engine).labels_table_name.iloc[0]
    
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
        subset_size = pd.read_sql(q, engine)
        
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

        sns.lineplot(
        data=gdf,
        x='train_end_time',
        y='metric_value',
        hue='model_type_child',
        ax=axes[1],
        alpha=0.7,
        # marker=''
        )
        # ticklabels = list(gdf.train_end_time.unique())
        # axes[1].set_xticklabels(ticklabels, rotation=45)
        axes[1].legend(loc='upper left', frameon=False, ncol=3, fontsize='small')
        axes[1].tick_params(axis='x', rotation=90)
        axes[1].set_ylabel(f'{metric}{parameter}')
        axes[1].set_xlabel('Time')
        axes[1].set_ylim(0, 0.3)
        axes[1].set_title(f'Model Performance')    
        
        plt.tight_layout()

    
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


def get_best_hp_config_for_each_model_type(engine, experiment_hashes, metric, parameter):
        
    q = f'''
        with avg_perf as (
            select 
                model_group_id, model_type, hyperparameters, avg(stochastic_value) as mean_performance
            from triage_metadata.experiment_models join triage_metadata.models m using(model_hash)
                left join test_results.evaluations e
                on m.model_id = e.model_id
                and e.metric = '{metric}'
                and e.parameter = '{parameter}'
                and e.subset_hash = ''
            where experiment_hash in ('{"','".join(experiment_hashes)}') 
            and model_type not like '%%Dummy%%'
            group by 1, 2, 3    
        )
        select distinct on(model_type)
        model_group_id, model_type, hyperparameters, mean_performance
        from avg_perf
        order by model_type, mean_performance desc
    '''

    best_models = pd.read_sql(q, engine).set_index('model_group_id').sort_values(by='mean_performance', ascending=False)
    best_models['model_type'] = best_models['model_type'].str.split('.').apply(lambda x: x[-1])
        
    return best_models
    


def plot_performance_against_bias(engine, experiment_hashes, metric, parameter, bias_metric, groups=None, model_group_ids=None, selected_models=None, bias_metric_tolerance=0.2):
    ''' Plot the performanc metric against the bias metric for all or selected models.
        Args:
            engine: DB connection
            experiment_hashes ([])
    '''
    
    if model_group_ids is None: 
        logging.warning('No model groups specified. Usign all model group ids')
        model_group_ids = summarize_model_groups(engine, experiment_hashes).model_group_id.tolist()
        
        if not model_group_ids:
            logging.warning('No model groups belong to the experiment! Returning None')
            return None
        
    # If no groups are specified, we show results for all groups    
    if groups is None:
        
        logging.info('No groups are specified. Showing results for all attributes and their values')
        groups = dict()
        
        q = f'''
        select 
            distinct attribute_name, attribute_value  
        from test_results.aequitas a
        where attribute_name::varchar in (
            select replace(jsonb_array_elements(config -> 'bias_audit_config' -> 'attribute_columns')::varchar, '"', '')   as attr
            from triage_metadata.experiments e 
            where experiment_hash = '{experiment_hashes[0]}'
        ) 
        order by 1, 2
        '''
        
        rg = pd.read_sql(q, engine)
        
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
                            and a.attribute_name in ('{"','".join(attributes)}')
                            and a.attribute_value in ('{"','".join(attribute_values)}')
                            and a.tie_breaker= 'worst' 
            where experiment_hash in  ('{"','".join(experiment_hashes)}')
            
    '''
    
    #and a.attribute_value in ('{', '.join(attribute_values)}')
    metrics = pd.read_sql(q, engine)

    # metrics['Model Class'] = metrics['model_type'].apply(lambda x: x.split('.')[-1])
    # metrics['model_label'] = metrics.apply(lambda x: f"{x['model_group_id']}: {x['model_type'].split('.')[-1]}", axis=1)
    metrics['model_label'] = metrics.apply(lambda x: _format_model_name(x['model_type'], x['model_group_id']), axis=1)
        
        # str(x['model_group_id']) + ': ' + x['model_type']), axis=1) 
    # Metric means
    mean = metrics.groupby(['model_label', 'attribute_value']).mean()[['precision@100_abs', 'tpr_disparity']].reset_index().sort_values('model_label')
    
    # Metric standard errors
    sem = metrics.groupby(['model_label', 'attribute_value']).sem()[['precision@100_abs', 'tpr_disparity']].reset_index().sort_values('model_label')
    labels = sorted(mean.model_label.unique())
    
    # n_attrs = sum([len(x) for x in groups.values()])
    n_attrs = len(attribute_values)
    ax_cntr = 0
    # bias_tolerance = 0.2
    fig, axes = plt.subplots(1, n_attrs, figsize=(4*n_attrs + 1, 4), sharey=True, sharex=True, dpi=100)
    colors=sns.color_palette().as_hex()[:len(mean.model_label.unique())]
    
    for group, attrs in groups.items():
        for attr in attrs:
            msk = mean['attribute_value'] == attr
            x = mean[msk]['precision@100_abs'].tolist()
            y = mean[msk]['tpr_disparity'].tolist()

            msk = sem['attribute_value'] == attr
            yerr = sem[msk]['tpr_disparity'].tolist()
            xerr = sem[msk]['precision@100_abs'].tolist()
            
            for i in range(len(x)):
                axes[ax_cntr].errorbar(x[i], y[i], yerr[i], xerr[i], fmt=' ', linewidth=1, capsize=2, color=colors[i], alpha=0.5)
                axes[ax_cntr].scatter(x[i], y[i], c=colors[i], label=labels[i])
                axes[ax_cntr].set(title=f'{group} | {attr}', xlabel='Performance Metric', ylabel='Bias Metric', ylim=[0, 3])
                axes[ax_cntr].axhline(y=1, color='gray', linestyle='--', alpha=0.1)
                axes[ax_cntr].axhline(y=1+bias_metric_tolerance, color='gray', linestyle=':', alpha=0.01)
                axes[ax_cntr].axhline(y=1-bias_metric_tolerance, color='gray', linestyle=':', alpha=0.01)
            ax_cntr += 1
        axes[-1].legend(bbox_to_anchor=(1,1), loc='upper left', frameon=False)
        sns.despine()
    
    
    # msk = metrics.attribute_value.str.contains('|'.join(attribute_values))

    # g = sns.FacetGrid(metrics[msk].sort_values('train_end_time'), row='attribute_value', col="train_end_time", hue='model_label', height=2.5)
    # g.map(sns.scatterplot, f'{metric}{parameter}', f"{bias_metric}")
    # g.add_legend(title='')

    # # drawing the parity reference line
    # g.map(plt.axhline, y=1, color='gray', linestyle='--', alpha=0.1)

    # # Drawing the tolerance bounds set
    # g.map(plt.axhline, y=1+bias_metric_tolerance, color='gray', linestyle=':', alpha=0.01)
    # g.map(plt.axhline, y=1-bias_metric_tolerance, color='gray', linestyle=':', alpha=0.01)

    # g.figure.set_dpi(300)
    # g.set_titles(template='{row_name}\n{col_name}')
    # g.set_axis_labels(f"{metric}{parameter}", 'TPR Ratio')

    # g.tight_layout()
    # g.set(ylim=(0, 1.8))

    
def plot_prk_curves(engine, experiment_hashes, model_groups=None, step_size=0.01):
    
    if model_groups is None:
        q = f'''
            select 
            distinct model_group_id
            from triage_metadata.experiment_models join triage_metadata.models using(model_hash)
            where experiment_hash in  ('{"','".join(experiment_hashes)}')        
        '''
        
        model_groups = pd.read_sql(q, engine)['model_group_id'].tolist()
        
        if not model_groups:
            logging.warning('No model groups belong to the experiment! Returning None')
            return None
            
    
    
    rep = PostmodelingReport(
        engine=engine,
        experiment_hashes=experiment_hashes,
        model_groups=model_groups
    )
    
    rep.plot_prk_curves(
        pct_step_size=step_size
    )
    

def feature_missingness_stats(engine):
    """
    Generates the mean, min, max missingness degree for each feature. 
    Assumes that the current "features" schema holds the relevant features for the experiment
    """
    
    q = '''
    select 
    table_name 
    from information_schema.tables
    where table_schema = 'features'
    and table_name like '%%aggregation_imputed'
    '''

    feature_tables = pd.read_sql(q, engine)['table_name'].tolist()
    
    logging.info(f'{len(feature_tables)} Tables')
    
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

        column_names[table] = pd.read_sql(q, engine).column_name.tolist()
        
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
        df = pd.read_sql(query, engine).set_index(['as_of_date', 'cohort_size'])

        if results.empty:
            results = df
        else:
            results = results.join(df)
            
    df = pd.concat([results.mean(), results.min(), results.max()], axis=1).fillna(0)
    df.columns = ['mean (%)', 'min (%)', 'max (%)']

    # df.style.applymap(lambda x: 'background-color : pink' if x>80 else '')
    
    return df 