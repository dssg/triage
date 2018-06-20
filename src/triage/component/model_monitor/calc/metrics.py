import numpy as np
import scipy.stats as stats


# weighted statistics
def weighted_var(x, weights):
    weighted_mean = np.average(x, weights=weights)
    return np.average((x - weighted_mean)**2, weights=weights)


def weighted_covar(x1, x2, weights):
    weighted_x1_mean = np.average(x1, weights=weights)
    weighted_x2_mean = np.average(x2, weights=weights)
    return np.average((x1 - weighted_x1_mean) * (x2 - weighted_x2_mean), weights=weights)


def weighted_corr(x1, x2, weights):
    return weighted_covar(x1, x2, weights) / (weighted_var(x1, weights) * weighted_var(x2, weights))


# pointwise comparisons
def jaccard_similarity(x1, x2, weights=None):
    if not weights:
        # calculate based on cardinality
        intersect_cardinality = len(set.intersection(*[set(x1), set(x2)]))
        union_cardinality = len(set.union(*[set(x1), set(x2)]))

        if union_cardinality != 0:
            return intersect_cardinality / float(union_cardinality)
        else:
            return 0.
    else:
        # calculate ordered metrics based on weights
        intersect_vals = np.in1d(x2, x1).astype(float) * weights
        return np.sum(intersect_vals) / np.sum(weights)


def spearman_rank_corr(x1, x2, weights=None):
    if not weights:
        return stats.spearmanr(x1, x2)[0]
    else:
        rank1 = stats.mstats.rankdata(x1)
        rank2 = stats.mstats.rankdata(x2)
        return weighted_corr(rank1, rank2, weights)


def kendall_tau(x1, x2, weights=None):
    if not weights:
        return stats.kendalltau(x1, x2)
    else:
        raise NotImplementedError("Weighted Kendalls tau not implemented yet")

