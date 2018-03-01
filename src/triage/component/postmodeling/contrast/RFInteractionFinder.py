from sklearn.tree._tree import TREE_LEAF
import itertools as it
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def _feature_pair_report(pair_and_values,
                         description='pairs',
                         measurement='value',
                         note=None,
                         n=10,
                         feature_list=None):
    print('-' * 80)
    print(description)
    print('-' * 80)
    print('feature pair : {}'.format(measurement))
    for pair, value in it.islice(pair_and_values, n):
        if feature_list is not None:
            print('({}*{}): {}'.format(feature_list[pair[0]],feature_list[pair[1]], value))    
        else: 
            print('{} : {}'.format(pair, value))
    if note is not None:
        print('* {}'.format(note))
    print()


def feature_pairs_in_tree(dt):
    """Lists subsequent features sorted by importance
    Parameters
    ----------
    dt : sklearn.tree.DecisionTreeClassifer
    Returns
    -------
    list of list of tuple of int :
        Going from inside to out:
        1. Each int is a feature that a node split on

        2. If two ints appear in the same tuple, then there was a node
           that split on the second feature immediately below a node
           that split on the first feature
        3. Tuples appearing in the same inner list appear at the same
           depth in the tree
        4. The outer list describes the entire tree
    """
    if not isinstance(dt, DecisionTreeClassifier):
        raise ValueError('dt must be an sklearn.tree.DecisionTreeClassifier')
    t = dt.tree_
    feature = t.feature
    children_left = t.children_left
    children_right = t.children_right
    result = []
    if t.children_left[0] == TREE_LEAF:
        return result
    next_queue = [0]
    while next_queue:
        this_queue = next_queue
        next_queue = []
        results_this_depth = []
        while this_queue:
            node = this_queue.pop()
            left_child = children_left[node]
            right_child = children_right[node]
            if children_left[left_child] != TREE_LEAF:
                results_this_depth.append(tuple(sorted(
                    (feature[node],
                     feature[left_child]))))
                next_queue.append(left_child)
            if children_left[right_child] != TREE_LEAF:
                results_this_depth.append(tuple(sorted(
                    (feature[node],
                     feature[right_child]))))
                next_queue.append(right_child)
        result.append(results_this_depth)
    result.pop()  # The last results are always empty
    return result


def feature_pairs_in_rf(rf, feature_list=None, weight_by_depth=None, verbose=True, n=20):
    """Describes the frequency of features appearing subsequently in each tree
    in a random forest

    Parameters
    ----------
    rf : sklearn.ensemble.RandomForestClassifier
        Fitted random forest
    weight_by_depth : iterable or None
        Weights to give to each depth in the "occurences weighted by depth
        metric"
        weight_by_depth is a vector. The 0th entry is the weight of being at
        depth 0; the 1st entry is the weight of being at depth 1, etc.
        If not provided, wdiogenes are linear with negative depth. If
        the provided vector is not as long as the number of depths, then
        remaining depths are weighted with 0
    verbose : boolean
        iff True, prints metrics to console
    n : int
        Prints the top-n-scoring feature pairs to console if verbose==True
    Returns
    -------
    (collections.Counter, list of collections.Counter, dict, dict)
        A tuple with a number of metrics
        1. A Counter of cooccuring feature pairs at all depths
        2. A list of Counters of feature pairs. Element 0 corresponds to
           depth 0, element 1 corresponds to depth 1 etc.
        3. A dict where keys are feature pairs and values are the average
           depth of those feature pairs
        4. A dict where keys are feature pairs and values are the number
           of occurences of those feature pairs weighted by depth

    """
    #if not isinstance(rf, RandomForestClassifier):
    #    raise ValueError(
    #        'rf must be an sklearn.Ensemble.RandomForestClassifier')

    pairs_by_est = [feature_pairs_in_tree(est) for est in rf.estimators_]
    pairs_by_depth = [list(it.chain(*pair_list)) for pair_list in
                      list(it.zip_longest(*pairs_by_est, fillvalue=[]))]
    pairs_flat = list(it.chain(*pairs_by_depth))
    depths_by_pair = {}
    for depth, pairs in enumerate(pairs_by_depth):
        for pair in pairs:
            try:
                depths_by_pair[pair] += [depth]
            except KeyError:
                depths_by_pair[pair] = [depth]
    counts_by_pair = Counter(pairs_flat)
    count_pairs_by_depth = [Counter(pairs) for pairs in pairs_by_depth]

    depth_len = len(pairs_by_depth)
    if weight_by_depth is None:
        weight_by_depth = [(depth_len - float(depth)) / depth_len for depth in
                           range(depth_len)]
    weight_filler = it.repeat(0.0, depth_len - len(weight_by_depth))
    wdiogenes = list(it.chain(weight_by_depth, weight_filler))

    average_depth_by_pair = {pair: float(sum(depths)) / len(depths) for
                             pair, depths in depths_by_pair.items()}

    weighted = {pair: sum([wdiogenes[depth] for depth in depths])
                for pair, depths in depths_by_pair.items()}

    if verbose:
        print('=' * 80)
        print('RF Subsequent Pair Analysis')
        print('=' * 80)
        print()
        _feature_pair_report(
            counts_by_pair.most_common(),
            'Overall Occurrences',
            'occurrences',
            n=n,
            feature_list=feature_list)
        _feature_pair_report(
            sorted([item for item in average_depth_by_pair.items()],
                   key=lambda item: item[1]),
            'Average depth',
            'average depth',
            'Max depth was {}'.format(depth_len - 1),
            n=n,
            feature_list=feature_list)
        _feature_pair_report(
            sorted([item for item in weighted.items()],
                   key=lambda item: item[1]),
            'Occurrences weighted by depth',
            'sum weight',
            'Wdiogenes for depth 0, 1, 2, ... were: {}'.format(wdiogenes),
            n=n,
            feature_list=feature_list)

        for depth, pairs in enumerate(count_pairs_by_depth):
            _feature_pair_report(
                pairs.most_common(),
                'Occurrences at depth {}'.format(depth),
                'occurrences',
                n=n,
                feature_list=feature_list)

    return (counts_by_pair, count_pairs_by_depth, average_depth_by_pair, weighted)
