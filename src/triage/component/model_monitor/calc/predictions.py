import numpy as np
import pandas as pd
import itertools
from ..metrics import jaccard_similarity, spearman_rank_corr, kendall_tau


def extract_date_lags(df, lag, join_keys=('model_id',)):
    # create timedelta from lag

    # NOTE: assumes lag is in the format '<int><unit>'
    num_time_units = int(lag[:-1])
    time_units = lag[-1]
    lag_delta = pd.Timedelta(num_time_units, unit=time_units)

    # round to the nearest day
    # NOTE: certain time units are adjusted for equal spacing / leap years
    lag_delta_rounded = pd.Timedelta(lag_delta.days, unit='d')

    # clean df columns
    df.loc[:, 'as_of_date'] = pd.DatetimeIndex(df['as_of_date']).normalize().to_series()

    # add lags and self join
    df.loc[:, 'as_of_date_lag'] = df['as_of_date'] - lag_delta_rounded

    left_join_keys = join_keys + ('as_of_date',)
    right_join_keys = join_keys + ('as_of_date_lag',)
    merged_df = pd.merge(df, df,
                         how='inner',
                         left_on=left_join_keys,
                         right_on=right_join_keys,
                         suffixes=("", "_lag"))

    return merged_df


def apply_rank_calc(s1, s2, metric):
    if metric == 'jaccard':
        return jaccard_similarity(s1, s2)
    elif metric == 'spearman':
        return spearman_rank_corr(s1, s2)
    elif metric == 'kendall':
        return kendall_tau(s1, s2)
    else:
        raise ValueError("Unknown metric '{}'".format(metric))


def extract_subset(group_df,
                   k_threshold,
                   use_top_entities=True,
                   use_lag_as_reference=False):
    # if threshold between 0 and 1, filter by quantile
    if 0.0 < k_threshold < 1.0:
        # parse percent rank relative to proper side
        if use_top_entities:
            if use_lag_as_reference:
                odf = group_df[group_df['rank_pct_lag'] >= 1 - k_threshold]
            else:
                odf = group_df[group_df['rank_pct'] >= 1 - k_threshold]
        else:
            if use_lag_as_reference:
                odf = group_df[group_df['rank_pct_lag'] <= k_threshold]
            else:
                odf = group_df[group_df['rank_pct'] <= k_threshold]
        return odf[['rank_pct', 'rank_pct_lag']]
    else:
        # parse absolute rank relative to proper side
        max_entity_rank = group_df['rank_abs'].max()

        if use_top_entities:
            if use_lag_as_reference:
                odf = group_df[group_df['rank_abs_lag'] >= max_entity_rank - k_threshold]
            else:
                odf = group_df[group_df['rank_abs'] >= max_entity_rank - k_threshold]
        else:
            if use_lag_as_reference:
                odf = group_df[group_df['rank_abs_lag'] <= k_threshold]
            else:
                odf = group_df[group_df['rank_abs'] <= k_threshold]
        return odf[['rank_abs', 'rank_abs_lag']]


def create_prediction_stability_calcs(prediction_df,
                                      mm_config):
    # read config for target metrics
    subset_entities_config = mm_config['prediction_metrics']['subset_entities']
    top_thresholds = subset_entities_config.get('top_entities', [])
    bottom_thresholds = subset_entities_config.get('bottom_entities', [])
    subset_metrics = subset_entities_config.get('metrics', [])

    all_entities_config = mm_config['prediction_metrics']['all_entities']
    all_entities_metrics = all_entities_config.get('metrics', [])

    results = []

    # apply calculations to each model
    for model_id, group_df in prediction_df.groupby("model_id"):

        # apply all top entities metrics
        for (top_threshold, subset_metric) in itertools.product(top_thresholds, subset_metrics):
            subset_df = extract_subset(group_df,
                                       k_threshold=top_threshold,
                                       use_top_entities=True)
            calc_result = apply_rank_calc(subset_df[subset_df.columns[0]],
                                          subset_df[subset_df.columns[1]],
                                          subset_metric)

            results.append((model_id, top_threshold, subset_metric, True, calc_result))

        # apply all top entities metrics
        for (bottom_threshold, subset_metric) in itertools.product(bottom_thresholds, subset_metrics):
            subset_df = extract_subset(group_df,
                                       k_threshold=bottom_threshold,
                                       use_top_entities=True)
            calc_result = apply_rank_calc(subset_df[subset_df.columns[0]],
                                          subset_df[subset_df.columns[1]],
                                          subset_metric)

            results.append((model_id, bottom_threshold, subset_metric, False, calc_result))

        # apply top entities metrics
        for all_entities_metric in all_entities_metrics:
            calc_result = apply_rank_calc(group_df['rank_abs'],
                                          group_df['rank_abs_lag'])
            results.append((model_id, 0, all_entities_metric, False, calc_result))

        odf = pd.DataFrame.from_records(results,
                                        columns=['model_id',
                                                 'k_threshold',
                                                 'metric_type',
                                                 'use_top_entities',
                                                 'metric_value'])

        return odf





