import os
import tempfile
from collections import OrderedDict

import pandas as pd
import yaml
from moto import mock_s3
import boto3
from numpy.testing import assert_almost_equal
from unittest import mock
import pytest

from triage.component.catwalk.storage import (
    CSVMatrixStore,
    FSStore,
    HDFMatrixStore,
    S3Store,
    ProjectStorage,
    ModelStorageEngine,
)


class SomeClass(object):
    def __init__(self, val):
        self.val = val


def test_S3Store():
    with mock_s3():
        client = boto3.client("s3")
        client.create_bucket(Bucket="test_bucket", ACL="public-read-write")
        store = S3Store(f"s3://test_bucket/a_path")
        assert not store.exists()
        store.write("val".encode("utf-8"))
        assert store.exists()
        newVal = store.load()
        assert newVal.decode("utf-8") == "val"
        store.delete()
        assert not store.exists()


def test_FSStore():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "tmpfile")
        store = FSStore(tmpfile)
        assert not store.exists()
        store.write("val".encode("utf-8"))
        assert store.exists()
        newVal = store.load()
        assert newVal.decode("utf-8") == "val"
        store.delete()
        assert not store.exists()


def test_ModelStorageEngine_nocaching(project_storage):
    mse = ModelStorageEngine(project_storage)
    mse.write('testobject', 'myhash')
    assert mse.exists('myhash')
    assert mse.load('myhash') == 'testobject'
    assert 'myhash' not in mse.cache


def test_ModelStorageEngine_caching(project_storage):
    mse = ModelStorageEngine(project_storage)
    with mse.cache_models():
        mse.write('testobject', 'myhash')
        with mock.patch.object(mse, "_get_store") as get_store_mock:
            assert mse.load('myhash') == 'testobject'
            assert not get_store_mock.called
        assert 'myhash' in mse.cache
    # when cache_models goes out of scope the cache should be empty
    assert 'myhash' not in mse.cache


DATA_DICT = OrderedDict(
    [
        ("entity_id", [1, 2]),
        ("k_feature", [0.5, 0.4]),
        ("m_feature", [0.4, 0.5]),
        ("label", [0, 1]),
    ]
)

METADATA = {"label_name": "label", "indices": ["entity_id"]}


def matrix_stores():
    df = pd.DataFrame.from_dict(DATA_DICT).set_index(["entity_id"])

    with tempfile.TemporaryDirectory() as tmpdir:
        project_storage = ProjectStorage(tmpdir)
        tmpcsv = os.path.join(tmpdir, "df.csv")
        tmpyaml = os.path.join(tmpdir, "df.yaml")
        tmphdf = os.path.join(tmpdir, "df.h5")
        with open(tmpyaml, "w") as outfile:
            yaml.dump(METADATA, outfile, default_flow_style=False)
            df.to_csv(tmpcsv)
            df.to_hdf(tmphdf, "matrix")
            csv = CSVMatrixStore(project_storage, [], "df")
            hdf = HDFMatrixStore(project_storage, [], "df")
            assert csv.matrix.equals(hdf.matrix)
            yield from [csv, hdf]


def test_MatrixStore_empty():
    for matrix_store in matrix_stores():
        assert not matrix_store.empty


def test_MatrixStore_metadata():
    for matrix_store in matrix_stores():
        assert matrix_store.metadata == METADATA


def test_MatrixStore_head_of_matrix():
    for matrix_store in matrix_stores():
        assert matrix_store.head_of_matrix.to_dict() == {
            "k_feature": {1: 0.5},
            "m_feature": {1: 0.4},
            "label": {1: 0},
        }


def test_MatrixStore_columns():
    for matrix_store in matrix_stores():
        assert matrix_store.columns() == ["k_feature", "m_feature"]


def test_MatrixStore_resort_columns():
    for matrix_store in matrix_stores():
        result = matrix_store.matrix_with_sorted_columns(
            ["m_feature", "k_feature"]
        ).values.tolist()
        expected = [[0.4, 0.5], [0.5, 0.4]]
        assert_almost_equal(expected, result)


def test_MatrixStore_already_sorted_columns():
    for matrix_store in matrix_stores():
        result = matrix_store.matrix_with_sorted_columns(
            ["k_feature", "m_feature"]
        ).values.tolist()
        expected = [[0.5, 0.4], [0.4, 0.5]]
        assert_almost_equal(expected, result)


def test_MatrixStore_sorted_columns_subset():
    with pytest.raises(ValueError):
        for matrix_store in matrix_stores():
            matrix_store.matrix_with_sorted_columns(["m_feature"]).values.tolist()


def test_MatrixStore_sorted_columns_superset():
    with pytest.raises(ValueError):
        for matrix_store in matrix_stores():
            matrix_store.matrix_with_sorted_columns(
                ["k_feature", "l_feature", "m_feature"]
            ).values.tolist()


def test_MatrixStore_sorted_columns_mismatch():
    with pytest.raises(ValueError):
        for matrix_store in matrix_stores():
            matrix_store.matrix_with_sorted_columns(
                ["k_feature", "l_feature"]
            ).values.tolist()


def test_MatrixStore_save():
    for matrix_store in matrix_stores():
        original_dict = matrix_store.matrix.to_dict()
        matrix_store.save()
        # nuke the cache to force reload
        matrix_store.matrix = None
        assert matrix_store.matrix.to_dict() == original_dict


def test_as_of_dates_entity_index(project_storage):
    data = {
        "entity_id": [1, 2],
        "feature_one": [0.5, 0.6],
        "feature_two": [0.5, 0.6],
    }
    matrix_store = CSVMatrixStore(project_storage, [], "test")
    matrix_store.matrix = pd.DataFrame.from_dict(data)
    matrix_store.metadata = {"end_time": "2016-01-01", "indices": ["entity_id"]}

    assert matrix_store.as_of_dates == ["2016-01-01"]


def test_as_of_dates_entity_date_index(project_storage):
    data = {
        "entity_id": [1, 2, 1, 2],
        "feature_one": [0.5, 0.6, 0.5, 0.6],
        "feature_two": [0.5, 0.6, 0.5, 0.6],
        "as_of_date": ["2016-01-01", "2016-01-01", "2017-01-01", "2017-01-01"],
    }
    matrix_store = CSVMatrixStore(project_storage, [], "test")
    matrix_store.matrix = pd.DataFrame.from_dict(data).set_index(
        ["entity_id", "as_of_date"]
    )
    matrix_store.metadata = {"indices": ["entity_id", "as_of_date"]}

    assert matrix_store.as_of_dates == ["2016-01-01", "2017-01-01"]


def test_s3_save():
    with mock_s3():

        client = boto3.client("s3")
        client.create_bucket(Bucket="fake-matrix-bucket", ACL="public-read-write")
        example = next(matrix_stores())
        project_storage = ProjectStorage("s3://fake-matrix-bucket")

        tosave = CSVMatrixStore(project_storage, [], "test")
        tosave.matrix = example.matrix
        tosave.metadata = example.metadata
        tosave.save()

        tocheck = CSVMatrixStore(project_storage, [], "test")
        assert tocheck.metadata == example.metadata
        assert tocheck.matrix.to_dict() == example.matrix.to_dict()
