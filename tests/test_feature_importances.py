import numpy

import warnings

import pytest

from triage.feature_importances import _ad_hoc_feature_importances, get_feature_importances

from sklearn import datasets
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

@pytest.fixture
def trained_models():
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=12345)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    return {'RF':rf, 'LR':lr}

def test_throwing_warning_if_lr(trained_models):
    with pytest.warns(UserWarning):
        get_feature_importances(trained_models['LR'])


def test_correct_feature_importances_for_lr(trained_models):
    feature_importances = get_feature_importances(trained_models['LR'])

    ## It returns the intercept, too
    assert feature_importances.shape == (30,)

def test_correct_feature_importances_for_rf(trained_models):
    feature_importances = get_feature_importances(trained_models['RF'])

    assert feature_importances.shape == (30,)
