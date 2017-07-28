from audition.metric_directionality import greater_is_better, best_in_series, idxbest


def _mg_best_avg_by(df, value_col, metric):
    """Best model group in dataframe by average of some column

    Args:
        df (pandas.DataFrame)
        value_col (str) The column which contains the value to be averaged
        metric (str) the name of the column
    """
    return getattr(
        df.groupby(['model_group_id'])[value_col].mean().sample(frac=1),
        idxbest(metric)
    )()


def best_current_value(df, train_end_time, metric, param):
    """Pick the model group with the best current metric value

    Arguments:
        metric (string) -- model evaluation metric, such as 'precision@'
        param (string) -- model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp) -- current train end time
        df (pandas.DataFrame) -- dataframe containing the columns:
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                dist_from_best_case
    Returns: (int) the model group id to select, with highest current raw metric value
    """
    curr_df = df.loc[
                (df['train_end_time'] == train_end_time) &
                (df['metric'] == metric) &
                (df['parameter'] == param)
              ]
    # sample(frac=1) to shuffle rows so we don't accidentally introduce bias in breaking ties

    best_raw_value = getattr(curr_df['raw_value'], best_in_series(metric))()
    return curr_df\
        .loc[curr_df['raw_value'] == best_raw_value, 'model_group_id']\
        .sample(frac=1)\
        .tolist()[0]


def best_average_value(df, train_end_time, metric, param):
    """Pick the model with the highest average metric value so far

    Arguments:
        metric (string) -- model evaluation metric, such as 'precision@'
        param (string) -- model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp) -- current train end time
        df (pandas.DataFrame) -- dataframe containing the columns:
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                dist_from_best_case
    Returns: (int) the model group id to select, with highest mean raw metric value
    """

    met_df = df.loc[
                (df['metric'] == metric) &
                (df['parameter'] == param)
            ]
    # sample(frac=1) to shuffle rows so we don't accidentally introduce bias in breaking ties
    return _mg_best_avg_by(met_df, 'raw_value', metric)


def most_frequent_best_dist(df, train_end_time, metric, param, dist_from_best_case):
    """Pick the model that is most frequently within `dist_from_best_case` from the
    best-performing model group across test sets so far

    Arguments:
        dist_from_best_case (float) -- distance from the best performing model
        metric (string) -- model evaluation metric, such as 'precision@'
        param (string) -- model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp) -- current train end time
        df (pandas.DataFrame) -- dataframe containing the columns:
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                below_best
    Returns: (int) the model group id to select, with highest mean raw metric value
    """

    met_df = df.loc[
                (df['metric'] == metric) &
                (df['parameter'] == param)
            ]
    met_df['within_dist'] = (df['dist_from_best_case'] <= dist_from_best_case).astype('int')
    # sample(frac=1) to shuffle rows so we don't accidentally introduce bias in breaking ties
    return met_df.groupby(['model_group_id'])['within_dist'].mean().sample(frac=1).idxmax()


def best_average_two_metrics(
    df,
    train_end_time,
    metric1,
    param1,
    metric2,
    param2,
    metric1_weight=0.5
):
    """Pick the model with the highest average combined value to date
    of two metrics weighted together using `metric1_weight`

    Arguments:
        metric1_weight (float) -- relative weight of metric1, between 0 and 1
        metric1 (string) -- model evaluation metric, such as 'precision@'
        param1 (string) -- model evaluation metric parameter,
            such as '300_abs'
        metric2 (string) -- model evaluation metric, such as 'precision@'
        param2 (string) -- model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp) -- current train end time
        df (pandas.DataFrame) -- dataframe containing the columns:
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                below_best
    Returns: (int) the model group id to select, with highest mean raw metric value
    """

    if metric1_weight < 0 or metric1_weight > 1:
        raise ValueError("Metric weight must be between 0 and 1")

    metric1_dir = greater_is_better(metric1)
    metric2_dir = greater_is_better(metric2)
    if metric1_dir != metric2_dir:
        raise ValueError("Metric directionalities must be the same")

    met_df = df.loc[
                (
                    (df['metric'] == metric1) &
                    (df['parameter'] == param1)
                ) |
                (
                    (df['metric'] == metric2) &
                    (df['parameter'] == param2)
                )
            ]

    met_df.loc[
        (met_df['metric'] == metric1) & (met_df['parameter'] == param1),
        'weighted_raw'
    ] = met_df.loc[
        (met_df['metric'] == metric1) & (met_df['parameter'] == param1),
        'raw_value'
    ] * metric1_weight

    met_df.loc[
        (met_df['metric'] == metric2) & (met_df['parameter'] == param2),
        'weighted_raw'
    ] = met_df.loc[
        (met_df['metric'] == metric2) & (met_df['parameter'] == param2),
        'raw_value'
    ] * (1.0 - metric1_weight)

    met_df_wt = met_df.groupby(['model_group_id', 'train_end_time'], as_index=False).sum()

    # sample(frac=1) to shuffle rows so we don't accidentally introduce bias in breaking ties
    return _mg_best_avg_by(met_df_wt, 'weighted_raw', metric1)


SELECTION_RULES = {
    'best_current_value': best_current_value,
    'best_average_value': best_average_value,
    'most_frequent_best_dist': most_frequent_best_dist,
    'best_average_two_metrics': best_average_two_metrics,
}
