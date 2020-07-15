import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)


import numpy as np
import sklearn.linear_model
from sklearn.svm import SVC
from triage.component.catwalk.estimators.classifiers import ScaledLogisticRegression


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

    if (isinstance(model, (sklearn.linear_model.LogisticRegression)) or
        isinstance(model, (ScaledLogisticRegression))):
        coef_odds_ratio = np.exp(model.coef_)
        # intercept_odds_ratio = np.exp(model.intercept_[:,np.newaxis])
        # We are ignoring the intercept

        # NOTE: We need to squeeze this array so it has the correct dimensions
        feature_importances = coef_odds_ratio.squeeze()

    elif isinstance(model, (SVC)) and (model.get_params()["kernel"] == "linear"):
        feature_importances = model.coef_.squeeze()

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

    else:
        logger.warning(
            "The selected algorithm, doesn't support a standard way "
            "of calculate the importance of each feature used. "
            "Falling back to ad-hoc methods "
            "(e.g. in LogisticRegression we will return Odd Ratios instead coefficients)"
        )

        feature_importances = _ad_hoc_feature_importances(model)

    # if we just ended up with a scalar (e.g., single feature logit), ensure we return an array
    if isinstance(feature_importances, np.ndarray) and feature_importances.shape == ():
        feature_importances = feature_importances.reshape((1,))

    return feature_importances
