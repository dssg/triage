# coding: utf-8
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES


DEPRECATION_MSG_1D = (
    "Passing 1d arrays as data is deprecated in 0.17 and will "
    "raise ValueError in 0.19. Reshape your data either using "
    "X.reshape(-1, 1) if your data has a single feature or "
    "X.reshape(1, -1) if it contains a single sample."
)


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

        X = check_array(X, copy=self.copy, ensure_2d=False, dtype=FLOAT_DTYPES)

        if X.ndim == 1:
            warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)

        if np.any(X > feature_range[1]) or np.any(X < feature_range[0]):
            warnings.warn(
                "You got data that are out of the range: {}".format(feature_range)
            )

        X[X > feature_range[1]] = feature_range[1]
        X[X < feature_range[0]] = feature_range[0]

        return X
