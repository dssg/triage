import warnings

import numpy as np
import sklearn.linear_model
from sklearn.svm import SVC


def _ad_hoc_feature_importances(model):
    """
    Get the "ad-hoc feature importances" for scikit-learn's models
    lacking the `feature_importances_` attribute

    Args:
        model: A trained model that has not a `feature_importances_` attribute

    Returns:
        At this moment, this method only returns the odds ratio of both the
        intercept and the coefficients given by sklearn's implementation of
        the LogisticRegression.
        The order of the odds ratio list is the standard
        of the statistical packages (like R, SAS, etc) i.e. (intercept, coefficients)
    """
    feature_importances = None

    if isinstance(model, (sklearn.linear_model.logistic.LogisticRegression)):
        coef_odds_ratio = np.exp(model.coef_)
        # intercept_odds_ratio = np.exp(model.intercept_[:,np.newaxis])
        # We are ignoring the intercept

        # NOTE: We need to squeeze this array so it has the correct dimensions
        feature_importances = coef_odds_ratio.squeeze()

    return feature_importances


def get_feature_importances(model):
    """
    Get feature importances (from scikit-learn) of a trained model.

    Args:
        model: Trained model

    Returns:
        Feature importances, or failing that, None
    """
    feature_importances = None

    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_

    elif isinstance(model, (SVC)) and (model.get_params()["kernel"] == "linear"):
        feature_importances = model.coef_.squeeze()

    else:
        warnings.warn(
            "\nThe selected algorithm, doesn't support a standard way"
            "\nof calculate the importance of each feature used."
            "\nFalling back to ad-hoc methods"
            "\n(e.g. in LogisticRegression we will return Odd Ratios instead coefficients)"
        )

        feature_importances = _ad_hoc_feature_importances(model)

    return feature_importances
