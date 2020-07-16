# coding: utf-8

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from .transformers import CutOff


class ScaledLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    An in-place replacement for the scikit-learn's LogisticRegression.

    It incorporates the MaxMinScaler, and the CutOff as preparations
    for the  logistic regression.
    """

    def __init__(
        self,
        penalty="l2",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="saga",
        max_iter=100,
        multi_class="ovr",
        verbose=0,
        warm_start=False,
        n_jobs=1,
    ):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

        self.minmax_scaler = MinMaxScaler()
        self.dsapp_cutoff = CutOff()
        self.lr = LogisticRegression(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
        )

        self.pipeline = Pipeline(
            [
                ("minmax_scaler", self.minmax_scaler),
                ("dsapp_cutoff", self.dsapp_cutoff),
                ("lr", self.lr),
            ]
        )

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)

        self.min_ = self.pipeline.named_steps["minmax_scaler"].min_
        self.scale_ = self.pipeline.named_steps["minmax_scaler"].scale_
        self.data_min_ = self.pipeline.named_steps["minmax_scaler"].data_min_
        self.data_max_ = self.pipeline.named_steps["minmax_scaler"].data_max_
        self.data_range_ = self.pipeline.named_steps["minmax_scaler"].data_range_

        self.coef_ = self.pipeline.named_steps["lr"].coef_
        self.intercept_ = self.pipeline.named_steps["lr"].intercept_

        self.classes_ = self.pipeline.named_steps["lr"].classes_

        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict_log_proba(self, X):
        return self.pipeline.predict_log_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)
