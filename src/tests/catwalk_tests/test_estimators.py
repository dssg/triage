import numpy as np

import pytest

from triage.component.catwalk.estimators.transformers import CutOff
from triage.component.catwalk.estimators.classifiers import ScaledLogisticRegression

from sklearn import linear_model

from sklearn import datasets
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


@pytest.fixture
def data():
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12345
    )

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def test_cutoff_warning():
    X_data = [1, 2, 0.5, 0.7, 100, -1, -23, 0]

    cutoff = CutOff()

    with pytest.raises(ValueError):
        cutoff.fit_transform(X_data)


def test_cutoff_transformer():
    cutoff = CutOff()

    X_data = np.array([1, 2, 0.5, 0.7, 100, -1, -23, 0]).reshape(-1,1)

    assert np.all(cutoff.fit_transform(X_data) == np.array([1, 1, 0.5, 0.7, 1, 0, 0, 0]).reshape(-1,1))


def test_cutoff_inside_a_pipeline(data):
    minmax_scaler = preprocessing.MinMaxScaler()
    dsapp_cutoff = CutOff()

    pipeline = Pipeline(
        [("minmax_scaler", minmax_scaler), ("dsapp_cutoff", dsapp_cutoff)]
    )

    pipeline.fit(data["X_train"], data["y_train"])

    X_fake_new_data = data["X_test"][-1, :].reshape(1, -1) + 0.5

    mms = preprocessing.MinMaxScaler().fit(data["X_train"])

    assert np.all(
        (mms.transform(X_fake_new_data) > 1)
        == (pipeline.transform(X_fake_new_data) == 1)
    )


def test_dsapp_lr(data):
    dsapp_lr = ScaledLogisticRegression()
    dsapp_lr.fit(data["X_train"], data["y_train"])

    minmax_scaler = preprocessing.MinMaxScaler()
    dsapp_cutoff = CutOff()
    lr = linear_model.LogisticRegression(solver='lbfgs')

    pipeline = Pipeline(
        [("minmax_scaler", minmax_scaler), ("dsapp_cutoff", dsapp_cutoff), ("lr", lr)]
    )

    pipeline.fit(data["X_train"], data["y_train"])

    assert np.all(dsapp_lr.predict(data["X_test"]) == pipeline.predict(data["X_test"]))
