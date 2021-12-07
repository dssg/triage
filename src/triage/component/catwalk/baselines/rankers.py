import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)
from scipy import stats
import numpy as np
import pandas as pd
from triage.component.catwalk.exceptions import BaselineFeatureNotInMatrix

REQUIRED_KEYS = frozenset(["feature", "low_value_high_score"])


class PercentileRankOneFeature:
    def __init__(self, feature, low_value_high_score=None, descend=None):
        logger.warning("DEPRECATION WARNING: PercentileRankOneFeature is being replaced by "
            "BaselineRankMultiFeature. Note, however, that the scores returned by the new "
            "ranker cannot be interpreted as percentiles."
        )
        if descend is not None:
            # If the deprecated `descend` parameter has been specified, raise a
            # warning, then use this value for low_value_high_score, which has
            # the same behavior
            logger.warning("DEPRECATION WARNING: parameter `descend` is deprecated for "
                "PercentileRankOneFeature. Use `low_value_high_score` instead."
            )
            if low_value_high_score is not None:
                raise ValueError("Only one of `descend` or `low_value_high_score` can be "
                    "specified for PercentileRankOneFeature."
                    )
            low_value_high_score = descend

        # set default this way so we can check if both have been specified above
        if low_value_high_score is None:
            low_value_high_score = False

        self.feature = feature  # which feature to rank on
        self.low_value_high_score = (
            low_value_high_score
        )  # should feature be ranked so lower values -> higher scores
        self.feature_importances_ = None

    def _set_feature_importances_(self, x):
        """ Assigns feature importances following the rule: 1 for the feature we
        are ranking on, 0 for all other features.
        """
        feature_importances = [0] * len(x.columns)
        try:
            position = x.columns.get_loc(self.feature)
        except KeyError:
            raise BaselineFeatureNotInMatrix(
                (
                    "Trying to rank on a feature ({feature_name})"
                    " not included in the training matrix!".format(
                        feature_name=self.feature
                    )
                )
            )
        feature_importances[position] = 1
        self.feature_importances_ = np.array(feature_importances)

    def fit(self, x, y):
        """ Set feature importances and return self.
        """
        self._set_feature_importances_(x)
        return self

    def predict_proba(self, x):
        """ Generate the rank percentile scores and return these.
        """
        # reduce x to the selected feature, raise error if not found
        x = x[self.feature]

        # we need different behavior depending on rank ordering. percentiles
        # should be able to be interpreted as "proportion of entities ranking
        # BELOW this entity's value". scipy will assign lower ranks to lower
        # values of the feature. so if the entities have values [0, 0, 1, 2, 2],
        # the first two entities will have the lowest ranks (and therefore the
        # lowest risk scores) and the last two will have the highest ranks (and
        # highest risk scores). for the "low_value_high_score" method, we need to reverse
        # this, and for both sorting directions, we need to convert the ranks to
        # percentiles.

        # when ascending: tied entities should get the *lowest* rank, so for
        # [0, 0, 1, 2, 2] the ranks should be [1, 1, 3, 4, 4]. these can be
        # converted to the number of entities below each value by subtracting 1
        # from each rank, yielding [0, 0, 2, 3, 3]. from here, we can calculate
        # the proportions by dividing by the length of each list.
        method = "min"
        subtract = 1

        # when `low_value_high_score=True`: tied entities should get the *highest* rank, so for
        # [0, 0, 1, 2, 2] the ranks should be [2, 2, 3, 5, 5]. if we reverse
        # these ranks by substracting all items from the maximum rank (5), we
        # end up with the correct ranks for calculating percentiles:
        # [3, 3, 2, 0, 0]. to simplify the code, we first divide by the length
        # of the list then subtract the result from the maxmimum percentile (1).
        # it produces the same result as subtracting from 5 then dividing:
        #   ([5, 5, 5, 5, 5] -  [2, 2, 3, 5, 5]) / 5  = [0.6, 0.6, 0.4, 0, 0]
        # and
        #    [1, 1, 1, 1, 1] - ([2, 2, 3, 5, 5]  / 5) = [0.6, 0.6, 0.4, 0, 0]
        if self.low_value_high_score:
            method = "max"
            subtract = 0

        # get the ranks and convert to percentiles
        ranks = stats.rankdata(x, method)
        ranks = [(rank - subtract) / len(x) for rank in ranks]
        if self.low_value_high_score:
            ranks = [1 - rank for rank in ranks]

        # format it like sklearn output and return
        return np.array([np.zeros(len(x)), ranks]).transpose()


class BaselineRankMultiFeature:
    def __init__(self, rules):
        if not isinstance(rules, list):
            rules = [rules]

        # validate rules: must have feature and sort order
        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError('Rules for BaselineRankMultiFeature must be of type dict')
            if not rule.keys() >= REQUIRED_KEYS:
                raise ValueError(f'BaselineRankMultiFeature rule "{rule}" missing one or more required keys ({REQUIRED_KEYS})')

        self.rules = rules
        self.feature_importances_ = None

    @property
    def all_feature_names(self):
        return [rule["feature"] for rule in self.rules]

    @property
    def all_sort_directions(self):
        # note that ascending=True sort will mean low values get low scores,
        # so negate the parameter direction to get the right relationship
        return [not rule['low_value_high_score'] for rule in self.rules]

    def _set_feature_importances_(self, x):
        """ Assigns feature importances following the rule: 1 for the features
        we are thresholding on, 0 for all other features.
        """
        feature_importances = [0] * len(x.columns)
        for feature_name in self.all_feature_names:
            try:
                position = x.columns.get_loc(feature_name)
            except KeyError:
                raise BaselineFeatureNotInMatrix(
                    (
                        "Rules refer to a feature ({feature_name}) not included in "
                        "the training matrix!".format(feature_name=feature_name)
                    )
                )
            feature_importances[position] = 1
        self.feature_importances_ = np.array(feature_importances)

    def fit(self, x, y):
        """ Set feature importances and return self.
        """
        self._set_feature_importances_(x)
        return self

    def predict_proba(self, x):
        """ Generate the rank scores and return these.
        """
        # reduce x to the selected set of features
        x = x[self.all_feature_names].reset_index(drop=True)

        x = x.sort_values(self.all_feature_names, ascending=self.all_sort_directions)

        # initialize curr_rank to -1 so the first record will have rank 0 (hence "score"
        # will range from 0 to 1)
        ranks = []
        curr_rank = -1
        prev = []

        # calculate ranks over sorted records, giving ties the same rank
        for rec in x.values:
            if not np.array_equal(prev, rec):
                curr_rank += 1
            ranks.append(curr_rank)
            prev = rec

        # normalize to 0 to 1 range
        x['score'] = [r/max(ranks) for r in ranks]

        # reset back to original sort order, calculate "score" for "0 class"
        scores_1 = x.sort_index()['score'].values
        scores_0 = np.array([1-s for s in scores_1])

        return np.array([scores_0, scores_1]).transpose()
