import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

class CutOff(BaseEstimator, TransformerMixin):
    """Transform features cutting values out of established range

    Args:
       feature_range: Range of allowed values, default=`(0,1)`

    Usage:
       The recommended way of using this is::

           from sklearn.pipeline import Pipeline

           minmax_scaler = preprocessing.MinMaxScaler()
           dsapp_cutoff = CutOff()
           lr  = linear_model.LogisticRegression()

           pipeline =Pipeline([
                 ('minmax_scaler',minmax_scaler),
                 ('dsapp_cutoff', dsapp_cutoff),
                 ('lr', lr)
           ])

           pipeline.fit(X_train, y_train)
           pipeline.predict(X_test)

    """

    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        feature_range = self.feature_range

        X = check_array(X, copy=self.copy, ensure_2d=True)

        if np.any(X > feature_range[1]) or np.any(X < feature_range[0]):
            logger.notice(
                f"You got feature values that are out of the range: {feature_range}. "
                f"The feature values will cutoff to fit in the range {feature_range}."
            )

        X[X > feature_range[1]] = feature_range[1]
        X[X < feature_range[0]] = feature_range[0]

        return X
