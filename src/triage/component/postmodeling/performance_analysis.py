""" temp file containing performance and feature analysis to be combined with other files later """ 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def get_score_distribution(db_conn, model_group_ids, top_k=0):
    """
    Plot the distribution of predicted scores for all entities across model groups and train_end_time
    Optionally, show only the top k predicted scores.

    Args:
        db_conn (sqlalchemy.engine): Database connection engine
        model_group_ids (List[int]): list of model group ids 
        top_k (int): if 0, displays all scores. Otherwise displays just the top_k scores.
    """

    # fetch all model ids for the model groups
    q = """
    select 
        model_id, to_char(train_end_time, 'YYYY-MM-DD') as train_end_time, model_group_id
    from triage_metadata.models where model_group_id in ({})
    """.format(','.join([str(x) for x in model_group_ids]))

    models = pd.read_sql(q, db_conn)
    train_end_times = list(models['train_end_time'].unique())

    fig, axes = plt.subplots( 
        len(model_group_ids), 
        len(train_end_times),
        figsize=(len(train_end_times)*3, len(model_group_ids)*3),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    # plot scores
    fig.suptitle('Comparing score distributions across time for the model groups')
    for j, end_time in enumerate(train_end_times):
        for i, model_group in enumerate(model_group_ids):
            msk = (models['model_group_id'] == model_group) & (models['train_end_time'] == end_time)
            model_id = models[msk]['model_id'].iloc[0]
            
            if top_k > 0: # restrict query to top_k 
                q = f"""
                select 
                    score, label_value
                from test_results.predictions 
                where model_id={model_id} and rank_abs_no_ties <= {top_k}
                """
            else:
                q = f"""
                    select 
                        score, label_value
                    from test_results.predictions 
                    where model_id={model_id}
                """
            scores = pd.read_sql(q, db_conn)
            sns.histplot(data=scores, x='score', hue='label_value', ax=axes[i, j], bins=20) 
            axes[i, j].set_ylabel('')
            axes[i, j].set_xlabel('')
            if i==0:
                axes[i, j].set_title(end_time)
            if j==0:
                axes[i,j].set_ylabel('model_group: {}'.format(model_group))
            if i>0 or j<len(train_end_times)-1:
                axes[i,j].get_legend().remove()
            axes[i,j].title.set_size(10)
    plt.subplots_adjust(top=0.85)


def plot_calibration_curve(db_conn, model_group_ids, n_bins=20):
    """
    Plot the calibration curve of predicted probability scores across model groups
    
    Args:
        db_conn (sqlalchemy.engine): Database connection engine
        model_group_ids (List[int]): list of model group ids 
        n_bins (int): the number of bins to define the calibration curve
    """

    # get model_ids and train_end_times
    q = """
    select 
        model_id, to_char(train_end_time, 'YYYY-MM-DD') as train_end_time, model_group_id
    from triage_metadata.models where model_group_id in ({})
    """.format(','.join([str(x) for x in model_group_ids]))

    models = pd.read_sql(q, db_conn)
    train_end_times = list(models['train_end_time'].unique())

    fig, axes = plt.subplots(
        1,
        len(train_end_times), 
        figsize=(len(train_end_times)*3, len(model_group_ids)*2),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    # plot calibration curves
    for j, end_time in enumerate(train_end_times):
        plot_info_list = []
        for i, model_group in enumerate(model_group_ids):
            # get model name and description
            q = f"""
                    select model_type, hyperparameters::TEXT as hyperparameter_str
                    from triage_metadata.models where model_group_id = {model_group}
                """
            model_info = pd.read_sql(q, db_conn)

            # get scores and labels
            msk = (models['model_group_id'] == model_group) & (models['train_end_time'] == end_time)
            model_id = models[msk]['model_id'].iloc[0]
            q = f"""
                select 
                    score, label_value
                from test_results.predictions 
                where model_id={model_id}
                """
            
            scores = pd.read_sql(q, db_conn)
            cal_x, cal_y = calibration_curve(scores['label_value'], scores['score'], n_bins=n_bins)
            
            for (x, y) in zip(cal_x, cal_y):
                plot_info_list.append([model_group, model_id, model_info.iloc[0]['model_type'], model_info.iloc[0]['hyperparameter_str'], x, y])
        plot_info_df = pd.DataFrame(plot_info_list, columns=['model_group_id', 'model_id', 'model_type', 'hyperparameter_str', 'cal_x', 'cal_y'])
        # plot all calibration curves for this end_time
        sns.lineplot(
            data = plot_info_df,
            x='cal_x',
            y='cal_y', 
            hue=plot_info_df[['model_type', 'hyperparameter_str']].apply(tuple, axis=1),
            marker='o', 
            ax=axes[0, j], 
        )
        # plot perfectly calibrated line y = x for comparison
        probabilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        sns.lineplot(
            data = pd.DataFrame(list(zip(probabilities, probabilities)), columns=['x', 'y']),
            x = 'x', 
            y = 'y', 
            color = 'gray',
            linestyle='dashed',
            ax = axes[0, j]
        )
        axes[0, j].set_title(end_time)
        axes[0,j].get_legend().remove()
        axes[0,j].set_aspect('equal', adjustable='box')
    plt.legend(bbox_to_anchor=(-6, -0.3), loc='upper left', borderaxespad=0)

def get_performance_over_time(db_conn, metric, parameter, model_group_ids):
    """
    Plot the performance over time for a specified metric (e.g. "precision@100")
    
    Args:
        db_conn (sqlalchemy.engine): Database connection engine
        metric (str): the target metric
        parameter (str): the point at which to evaluate the target metric (e.g. '20_pct', '100_abs')
        model_group_ids (List[int]): list of model group ids 
    """

    # retrieve stochastic values for all model_group_ids for the particular metric at all timestamps 
    q = """
            select 
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                model_type,
                hyperparameters,
                hyperparameters::TEXT as hyperparameter_str,
                stochastic_value as metric_value
            from triage_metadata.models join test_results.evaluations e using(model_id)
            where metric='{}'
            and parameter='{}'
            and subset_hash=''
            and model_group_id in ({})
    """.format(
        metric, parameter, ','.join([str(x) for x in model_group_ids])
    )

    evaluations = pd.read_sql(q, db_conn)

    # plot metric over time
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.lineplot(
        data=evaluations, 
        x='train_end_time', 
        y='metric_value', 
        hue=evaluations[['model_type', 'hyperparameter_str']].apply(tuple, axis=1),
        marker='o', 
        estimator='mean', ci='sd')
    plt.legend(bbox_to_anchor=(1.05, 1))
    ax.set_title('{}{} over time'.format(metric, parameter))


def plot_pr_curves(db_conn, model_group_ids):
    """
    Plots precision-recall curves at each train_end_time for all model groups
    
    Args:
        db_conn (sqlalchemy.engine): Database connection engine
        model_group_ids (List[int]): list of model group ids 
    """

    # get train_end_times
    q = """
    select 
        model_id, to_char(train_end_time, 'YYYY-MM-DD') as train_end_time, model_group_id
    from triage_metadata.models where model_group_id in ({})
    """.format(','.join([str(x) for x in model_group_ids]))

    models = pd.read_sql(q, db_conn)
    train_end_times = list(models['train_end_time'].unique())

    fig, axes = plt.subplots(
        len(model_group_ids), 
        len(train_end_times), 
        figsize=(len(train_end_times)*3, len(model_group_ids)*2),
        sharex=True,
        squeeze=False
    )

    # plot PR curves
    for j, end_time in enumerate(train_end_times):
        for i, model_group in enumerate(model_group_ids):
            # get precision and recall for this model_group 
            msk = (models['model_group_id'] == model_group) & (models['train_end_time'] == end_time)
            model_id = models[msk]['model_id'].iloc[0]
            q = """
                select 
                    model_id,
                    parameter,
                    metric,
                    stochastic_value
                from test_results.evaluations
                where model_id={} and subset_hash='' 
            """.format(model_id)
            
            eval_model = pd.read_sql(q, db_conn)
            
            eval_model['perc_points'] = [x.split('_')[0] for x in eval_model['parameter'].tolist()]
            eval_model['perc_points'] = pd.to_numeric(eval_model['perc_points'])
            
            msk_prec = eval_model['metric']=='precision@'
            msk_recall = eval_model['metric']=='recall@'
            msk_pct = eval_model['parameter'].str.contains('pct')

            # plot precision
            sns.lineplot(
                x='perc_points',
                y='stochastic_value', 
                data=eval_model[msk_pct & msk_prec], 
                label='precision@k',
                ax=axes[i, j], 
                estimator='mean', ci='sd'
            )
            # plot recall
            sns.lineplot(
                x='perc_points', 
                y='stochastic_value', 
                data=eval_model[msk_pct & msk_recall], 
                label='recall@k', 
                ax=axes[i,j], 
                estimator='mean', 
                ci='sd'
            )
            axes[i, j].set_xlabel('List size percentage (k%)')
            axes[i,j].set_ylabel('Metric Value')
            
            if i==0:
                axes[i, j].set_title(end_time)
            if j==0:
                axes[i,j].set_ylabel('model_group: {}'.format(model_group))
    fig.tight_layout()


def get_feature_importance(db_conn, model_group_ids, n_top_features=20):
    """
    Plot the top most important individual features across model groups and train end times

    Args:
        db_conn (sqlalchemy.engine): Database connection engine
        model_group_ids (List[int]): list of model group ids 
        n_top_features (int): the number of features to display
    """

    # get train_end_times
    q = """
    select 
        model_id, to_char(train_end_time, 'YYYY-MM-DD') as train_end_time, model_group_id
    from triage_metadata.models where model_group_id in ({})
    """.format(','.join([str(x) for x in model_group_ids]))

    models = pd.read_sql(q, db_conn)
    train_end_times = list(models['train_end_time'].unique())

    # get feature importances for all models
    q = """
        select
            feature,
            feature_importance, 
            rank_abs, 
            rank_pct
        from train_results.feature_importances
        where model_id={model_id}
        and rank_abs <= {n_top_features}
        order by feature_importance desc;
    """

    fig, axes = plt.subplots(
        len(train_end_times), 
        len(model_group_ids), 
        figsize=(len(model_group_ids)*6, len(train_end_times)*4),
        squeeze=False
    #     sharex=True
    )

    # plot feature importance
    for i, end_time in enumerate(train_end_times):
        for j, model_group in enumerate(model_group_ids):
            msk = (models['model_group_id'] == model_group) & (models['train_end_time'] == end_time)
            model_id = models[msk]['model_id'].iloc[0]
            feature_importance_scores = pd.read_sql(q.format(model_id=model_id, n_top_features=n_top_features), db_conn)
            
            sns.barplot(
                data=feature_importance_scores,
                x='feature_importance',
                y='feature',
                color='royalblue',
                ax=axes[i,j]
            )
            
            if j==0:
                axes[i, j].set_ylabel(end_time)
            if i==0:
                axes[i,j].set_title('model_group: {}'.format(model_group))
    fig.tight_layout()

def get_feature_importance_by_group(db_conn, experiment_hash, model_group_ids):
    """
    Plot the top most important feature groups as identified by the maximum importance of any feature in the group

    Args:
        db_conn (sqlalchemy.engine): Database connection engine
        experiment_hash (str): the experiment hash
        top_k (int): if 0, displays all scores. Otherwise displays just the top_k scores.
    """

    # get train_end_times
    q = """
    select 
        model_id, to_char(train_end_time, 'YYYY-MM-DD') as train_end_time, model_group_id
    from triage_metadata.models where model_group_id in ({})
    """.format(','.join([str(x) for x in model_group_ids]))

    models = pd.read_sql(q, db_conn)
    train_end_times = list(models['train_end_time'].unique())

    # get feature group names
    q = """
        select 
            config->'feature_aggregations' as feature_groups
        from triage_metadata.experiments where experiment_hash = '{}'
    """.format(experiment_hash)

    feature_groups = [i['prefix'] for i in pd.read_sql(q, db_conn)['feature_groups'].iloc[0]]#['prefix']#.iloc[0]#['prefix']
    feature_groups
    case_part = ''
    for fg in feature_groups:
        case_part = case_part + "\nWHEN feature like '{fg}%%' THEN '{fg}'".format(fg=fg)
    fig, axes = plt.subplots(
        len(train_end_times), 
        len(model_group_ids), 
        figsize=(len(model_group_ids)*5, len(train_end_times)*5),
        squeeze=False
    )

    # get feature group importances
    q = """
        with raw_importances as (
            select 
                model_id,
                feature,
                feature_importance,
                CASE {case_part}
                ELSE 'No feature group'
                END as feature_group
            FROM train_results.feature_importances
            WHERE model_id = {model_id}
        )
        SELECT
        model_id,
        feature_group,
            max(feature_importance) as importance_aggregate
        FROM raw_importances
        GROUP BY feature_group, model_id
        ORDER BY importance_aggregate desc;
    """

    # plot feature group importances
    for i, end_time in enumerate(train_end_times):
        for j, model_group in enumerate(model_group_ids):
            msk = (models['model_group_id'] == model_group) & (models['train_end_time'] == end_time)
            model_id = models[msk]['model_id'].iloc[0]
            
            feature_group_importance = pd.read_sql(q.format(case_part=case_part, model_id=model_id), db_conn)
            
            sns.barplot(
                data=feature_group_importance,
                x='importance_aggregate',
                y='feature_group',
                color='royalblue',
                ax=axes[i,j]
            )
            
            if j==0:
                axes[i, j].set_ylabel(end_time)
            if i==0:
                axes[i,j].set_title('model_group: {}'.format(model_group))
    fig.tight_layout()