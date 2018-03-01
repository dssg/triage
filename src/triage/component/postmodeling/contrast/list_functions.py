import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import stats
import itertools
import numpy as np
from . import error_analysis

from . import result_schema_helper


def get_score_distribution(conn, model_id, test_time, plot=False, filename=None):
    """
    This methods retrieves the score for given model_id and test_time. It returns a list
    and saves the distribution if requested. When no filename is specified the format used
    is: score_model_id_(value)_test_time_(value).png
    """
    df_score_label = result_schema_helper.get_prediction_score(
        conn=conn,
        model_id=model_id,
        as_of_date=test_time)

    df_score = df_score_label[['entity_id', 'score']]

    if plot:
        plt.style.use('ggplot')

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))

        axs.hist(df_score.score, bins=100, normed=True, alpha=0.5)
        axs.set_title('Score distribution ')
        axs.set_ylabel('Frequency')
        axs.set_xlabel('Score')

        if filename:
            plt.savefig(filename)
        else:
            plt.savefig('score_model_id_' + str(model_id) + '_test_time_' + str(test_time) + '.png')

    return df_score


def jaccard_similarity(x, y):
    '''
    http://www.pressthered.com/adding_dates_and_times_in_python/
    '''
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))

    if union_cardinality != 0:
        val = intersection_cardinality / float(union_cardinality)
    else:
        val = 0
    return val


def get_list_comparision(conn, model_ids, test_time, top_k):
    df_score_label = result_schema_helper.get_prediction_score(
        conn=conn,
        model_id=model_ids,
        as_of_date=test_time)

    df_score = df_score_label[['model_id', 'entity_id', 'score', 'rank_abs', 'label_value']]

    list_results = []

    for model_pair in itertools.combinations(model_ids, 2):
        # create the ranks to join after on feature name
        df_model_0 = df_score[df_score['model_id'] == model_pair[0]][['entity_id', 'rank_abs', 'score', 'label_value']]
        df_model_0.columns = ['entity_id', 'rank_abs_0', 'score_0', 'label_value_0']

        df_model_1 = df_score[df_score['model_id'] == model_pair[1]][['entity_id', 'rank_abs', 'score', 'label_value']]
        df_model_1.columns = ['entity_id', 'rank_abs_1', 'score_1', 'label_value_1']

        # merge both lists based on entity_id
        table_df = df_model_0.merge(df_model_1, how='inner', on=['entity_id'])

        top_k_set_0 = table_df.sort_values(['rank_abs_0'], axis=0)[['entity_id', 'label_value_0']][:top_k]
        top_k_set_1 = table_df.sort_values(['rank_abs_1'], axis=0)[['entity_id', 'label_value_1']][:top_k]

        jaccard_topk = jaccard_similarity(top_k_set_0['entity_id'],
                                          top_k_set_1['entity_id'])

        jaccard_topk_0 = jaccard_similarity(top_k_set_0[top_k_set_0.label_value_0 == 0].entity_id,
                                            top_k_set_1[top_k_set_1.label_value_1 == 0].entity_id)

        jaccard_topk_1 = jaccard_similarity(top_k_set_0[top_k_set_0.label_value_0 == 1].entity_id,
                                            top_k_set_1[top_k_set_1.label_value_1 == 1].entity_id)

        rank_order_overall = stats.spearmanr(table_df.rank_abs_0, table_df.rank_abs_1)

        # compare each topk against each other (not all features are overlapping therefore it is not symmetric)
        table_df_0_vs_1 = table_df.sort_values(['rank_abs_0'], axis=0)
        rank_order_0_vs_1 = stats.spearmanr(table_df_0_vs_1.rank_abs_0[:top_k], table_df_0_vs_1.rank_abs_1[:top_k])

        table_df_1_vs_0 = table_df.sort_values(['rank_abs_1'], axis=0)
        rank_order_1_vs_0 = stats.spearmanr(table_df_1_vs_0.rank_abs_0[:top_k], table_df_1_vs_0.rank_abs_1[:top_k])

        # calculate the KL divergence for the feature distribution, outer join to handle the case if
        # different features were used
        table_df_full = df_model_0.merge(df_model_1, how='outer', on=['entity_id'], )

        # Zero imputation
        table_df_full.fillna(0, inplace=True)

        # normalize the sum to 1
        score_dist_0 = _probnorm(table_df_full.score_0)
        score_dist_1 = _probnorm(table_df_full.score_1)

        kl_0_vs_1 = _KL(score_dist_0, score_dist_1)
        kl_1_vs_0 = _KL(score_dist_1, score_dist_0)

        result = {'model_0_id': model_pair[0],
                  'model_1_id': model_pair[1],
                  'jaccard': jaccard_topk,
                  'jaccard_0_class': jaccard_topk_0,
                  'jaccard_1_class': jaccard_topk_1,
                  'overall_rank_corr': rank_order_overall[0],
                  'rank_corr_topk_0_vs_1': rank_order_0_vs_1[0],
                  'rank_corr_topk_1_vs_0': rank_order_1_vs_0[0],
                  'kl_0_vs_1': kl_0_vs_1,
                  'kl_1_vs_0': kl_1_vs_0
                  }

        list_results.append(result)

    return list_results


def get_feature_list_comparision(conn, model_ids, buckets, top_k):
    df_imp = result_schema_helper.get_features_importances(
        conn=conn,
        model_ids=model_ids)

    df_imp = df_imp[['model_id', 'feature', 'feature_importance', 'rank_abs']]

    list_results = []

    for model_pair in itertools.combinations(model_ids, 2):
        # create the ranks to join after on feature name
        df_model_0 = df_imp[df_imp['model_id'] == model_pair[0]][['feature', 'rank_abs', 'feature_importance']]
        df_model_0.columns = ['feature', 'rank_abs_0', 'feature_importance_0']

        df_model_1 = df_imp[df_imp['model_id'] == model_pair[1]][['feature', 'rank_abs', 'feature_importance']]
        df_model_1.columns = ['feature', 'rank_abs_1', 'feature_importance_1']

        # merge both lists based on the feature
        table_df = df_model_0.merge(df_model_1, how='inner', on=['feature'])

        rank_order_overall = stats.spearmanr(table_df.rank_abs_0, table_df.rank_abs_1)

        # compare each topk against each other (not all entities are overlapping therefore it is not symmetric)
        table_df_0_vs_1 = table_df.sort_values(['rank_abs_0'], axis=0)
        rank_order_0_vs_1 = stats.spearmanr(table_df_0_vs_1.rank_abs_0[:top_k], table_df_0_vs_1.rank_abs_1[:top_k])

        table_df_1_vs_0 = table_df.sort_values(['rank_abs_1'], axis=0)
        rank_order_1_vs_0 = stats.spearmanr(table_df_1_vs_0.rank_abs_0[:top_k], table_df_1_vs_0.rank_abs_1[:top_k])

        jaccard_topk = jaccard_similarity(table_df.sort_values(['rank_abs_0'], axis=0)['feature'][:top_k],
                                          table_df.sort_values(['rank_abs_1'], axis=0)['feature'][:top_k])

        # calculate the KL divergence for the feature distribution, outer join to handle the case if
        # different features were used
        table_df_full = df_model_0.merge(df_model_1, how='outer', on=['feature'], )

        # Zero imputation
        table_df_full.fillna(0, inplace=True)

        # normalize the sum to 1
        feature_dist_0 = _probnorm(table_df_full.feature_importance_0)
        feature_dist_1 = _probnorm(table_df_full.feature_importance_1)


        kl_0_vs_1 = _KL(feature_dist_0, feature_dist_1)
        kl_1_vs_0 = _KL(feature_dist_1, feature_dist_0)

        result = {'model_0_id': model_pair[0],
                  'model_1_id': model_pair[1],
                  'jaccard': jaccard_topk,
                  'overall_rank_corr': rank_order_overall[0],
                  'rank_corr_topk_0_vs_1': rank_order_0_vs_1[0],
                  'rank_corr_topk_1_vs_0': rank_order_1_vs_0[0],
                  'kl_0_vs_1': kl_0_vs_1,
                  'kl_1_vs_0': kl_1_vs_0
                  }

        list_results.append(result)

    return list_results


def get_probability_by_decile(conn, model_id, test_time, plot=False, filename=None):
    """
    This methods retrieves the score for given model_id and test_time. It returns a list
    and saves the distribution if requested. When no filename is specified the format used
    is: score_model_id_(value)_test_time_(value).png
    """
    # default setting for decile
    no_bins=10

    df_score_label = result_schema_helper.get_prediction_score(
        conn=conn,
        model_id=model_id,
        as_of_date=test_time)

    df_score = df_score_label[['score', 'label_value']]

    [y_score_bin_mean, empirical_prob_pos] = error_analysis.reliability_curve(df_score_label.label_value,
                                                                              df_score_label.score, bins=no_bins)
    bins = [x for x in range(0, 10, 1)]

    df_prob_dec = pd.DataFrame([bins, list(empirical_prob_pos)]).T
    df_prob_dec.columns = ['bin', 'empirical_prob']
    df_prob_dec['bin'] = df_prob_dec['bin'].astype(int)

    if plot:
        plt.style.use('ggplot')

        if filename:
            error_analysis.plot_reliability_curve(df_score_label.score,
                                                  y_score_bin_mean,
                                                  empirical_prob_pos,
                                                  no_bins,
                                                  savefig=filename)

        else:
            error_analysis.plot_reliability_curve(df_score_label.score,
                                                  y_score_bin_mean,
                                                  empirical_prob_pos,
                                                  no_bins,
                                                  savefig='probability_calibration_model_id_' + str(
                                                      model_id) + '_test_time_' + str(test_time) + '.png')


    return df_prob_dec



def _probnorm(array):
    divisor = sum(array)

    return np.asarray([item / float(divisor) for item in array])


def _KL(p, q):
    p = p + 0.000001  # bc KLD undefined if any bin==0
    q = q + 0.000001

    return sum(p * np.log(p / q))
