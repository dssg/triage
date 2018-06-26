import pandas as pd
import itertools
from triage.component.model_monitor.calc.shared import apply_rank_calc, extract_subset


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

        # apply all bottom entities metrics
        for (bottom_threshold, subset_metric) in itertools.product(bottom_thresholds, subset_metrics):
            subset_df = extract_subset(group_df,
                                       k_threshold=bottom_threshold,
                                       use_top_entities=True)
            calc_result = apply_rank_calc(subset_df[subset_df.columns[0]],
                                          subset_df[subset_df.columns[1]],
                                          subset_metric)

            results.append((model_id, bottom_threshold, subset_metric, False, calc_result))

        # apply all entities metrics
        for all_entities_metric in all_entities_metrics:
            calc_result = apply_rank_calc(group_df['rank_abs'],
                                          group_df['rank_abs_lag'],
                                          all_entities_metric)
            results.append((model_id, 0, all_entities_metric, False, calc_result))

    odf = pd.DataFrame.from_records(results,
                                    columns=['model_id',
                                             'k_threshold',
                                             'metric_type',
                                             'use_top_entities',
                                             'metric_value'])

    return odf
