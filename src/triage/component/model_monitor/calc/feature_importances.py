import numpy as np
import scipy.stats as stats
import itertools
import pandas as pd
import warnings
import re

from triage.component.model_monitor.calc.shared import apply_rank_calc, extract_subset

timeagg_regex = re.compile('_([0-9]+[dwmy])_')
agg_func_regex = re.compile('_(min|max|sum|avg|var)$')


def _feature_to_timeagg(feat, raise_err=False):
    try:
        feat_remainder, agg_func = [i for i in agg_func_regex.split(feat) if i]
        source_table, time_agg, feature_name = timeagg_regex.split(feat_remainder)
        return pd.Series(
            {'feature': feat,
             'source_table': source_table,
             'time_agg': time_agg,
             'feature_name': feature_name,
             'agg_func': agg_func}
        )
    except Exception as e:
        if raise_err:
            raise
        else:
            msg = "Error encountered while converting, skipping '{feat}'...\n{excp}".format(
                feat=feat,
                excp=e
            )
            warnings.warn(msg, UserWarning)
            return pd.Series(
                {'feature': feat,
                 'source_table': '',
                 'time_agg': '',
                 'feature_name': '',
                 'agg_func': ''}
            )


def extract_block_features(features, clean=True):
    # assumes format <source_table>_<time_aggregation>_<feature_name>_<agg_func>
    res = pd.Series(features).apply(_feature_to_timeagg)
    if clean:
        return res.replace('', np.NaN).dropna()
    else:
        return res


def create_feature_stability_calcs(feature_importance_df,
                                   mm_config):
    # read config for target metrics
    feature_importance_config = mm_config['feature_importance_metrics']

    subset_entities_config = feature_importance_config['subset_entities']
    top_thresholds = subset_entities_config.get('top_entities', [])
    subset_metrics = subset_entities_config.get('metrics', [])

    all_entities_config = feature_importance_config['all_entities']
    all_entities_metrics = all_entities_config.get('metrics', [])

    use_feature_blocking = feature_importance_config['use_feature_blocking']
    if use_feature_blocking:
        feature_block_aggregations = tuple(feature_importance_config['feature_importance_aggregations'])
    else:
        feature_block_aggregations = ('__ALL__',)

    results = []

    # apply calculations to each model
    for model_id, group_df in feature_importance_df.groupby('model_id'):

        # for each feature block
        for feature_block_aggregation in feature_block_aggregations:
            # perform appropriate aggregation
            if feature_block_aggregation == ('__ALL__',):
                aggregated_features = group_df
            else:
                # get block feature definitions and join with original results
                unique_features = feature_importance_df['feature'].unique()
                block_features = extract_block_features(unique_features)
                mdf = pd.merge(group_df,
                               block_features,
                               on='feature',
                               how='inner')

                agg_groups = feature_block_aggregation.split('+')
                group_columns = agg_groups + ['as_of_date']

                # recalculate feature ranks
                aggregated_features = mdf.groupby(group_columns).agg({
                    'feature_importance': np.sum,
                }).reset_index()
                aggregated_features.loc[:, 'rank_abs'] = aggregated_features['feature_importance'].rank()
                aggregated_features.loc[:, 'rank_pct'] = aggregated_features['feature_importances'].apply(
                    lambda t: stats.percentileofscore(aggregated_features['feature_importances'], t)
                )

            # apply all top entities metrics
            for (top_threshold, subset_metric) in itertools.product(top_thresholds, subset_metrics):
                subset_df = extract_subset(group_df,
                                           k_threshold=top_threshold,
                                           use_top_entities=True)
                calc_result = apply_rank_calc(subset_df[subset_df.columns[0]],
                                              subset_df[subset_df.columns[1]],
                                              subset_metric)

                results.append((model_id, top_threshold, subset_metric, True, calc_result))

            # apply all entities metrics
            for all_entities_metric in all_entities_metrics:
                calc_result = apply_rank_calc(group_df['rank_abs'],
                                              group_df['rank_abs_lag'],
                                              all_entities_metric)
                results.append((model_id, 0, all_entities_metric, False, calc_result))

    return pd.DataFrame.from_records(results,
                                     columns=['model_id',
                                              'k_threshold',
                                              'metric_type',
                                              'use_top_entities',
                                              'metric_value'])
