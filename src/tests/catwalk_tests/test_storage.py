import datetime
import os
import tempfile
from collections import OrderedDict

import boto3
import pandas as pd
import pytest
import yaml
from moto import mock_s3
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal
from unittest import mock

from triage.component.catwalk.storage import (
    MatrixStore,
    CSVMatrixStore,
    FSStore,
    S3Store,
    ProjectStorage,
    ModelStorageEngine,
)

from tests.utils import CallSpy


class SomeClass:
    def __init__(self, val):
        self.val = val


@mock_s3
def test_S3Store():
    client = boto3.client("s3")
    client.create_bucket(
        Bucket="test_bucket",
        ACL="public-read-write",
        CreateBucketConfiguration={"LocationConstraint": "us-east-2"},
    )
    store = S3Store(f"s3://test_bucket/a_path")
    assert not store.exists()
    store.write("val".encode("utf-8"))
    assert store.exists()
    newVal = store.load()
    assert newVal.decode("utf-8") == "val"
    store.delete()
    assert not store.exists()


@mock_s3
def test_S3Store_large():
    client = boto3.client("s3")
    client.create_bucket(
        Bucket="test_bucket",
        ACL="public-read-write",
        CreateBucketConfiguration={"LocationConstraint": "us-east-2"},
    )

    store = S3Store("s3://test_bucket/a_path")
    assert not store.exists()

    # NOTE: The issue under test (currently) arises when too large a "part"
    # NOTE: is sent to S3 for upload -- greater than its 5 GiB limit on any
    # NOTE: single upload request.
    #
    # NOTE: Though s3fs uploads file parts as soon as its buffer reaches
    # NOTE: 5+ MiB, it does not ensure that its buffer -- and resulting
    # NOTE: upload "parts" -- remain under this limit (as the result of a
    # NOTE: single "write()").
    #
    # NOTE: Therefore, until s3fs adds handling to ensure it never attempts
    # NOTE: to upload such large payloads, we'll handle this in S3Store,
    # NOTE: by chunking out writes to s3fs.
    #
    # NOTE: This is all not only to explain the raison d'etre of this test,
    # NOTE: but also as context for the following warning: The
    # NOTE: payload we'll attempt to write, below, is far less than 5 GiB!!
    # NOTE: (Attempting to provision a 5 GiB string in RAM just for this
    # NOTE: test would be an ENORMOUS drag on test runs, and a conceivable
    # NOTE: disruption, depending on the test environment's resources.)
    #
    # NOTE: As such, this test *may* fall out of sync with either the code
    # NOTE: that it means to test or with the reality of the S3 API -- even
    # NOTE: to the point of self-invalidation. (But, this should do the
    # NOTE: trick; and, we can always increase the payload size here, or
    # NOTE: otherwise tweak configuration, as necessary.)
    one_mb = 2**20
    payload = b"0" * (10 * one_mb)  # 10MiB text of all zeros

    with CallSpy("botocore.client.BaseClient._make_api_call") as spy:
        store.write(payload)

    call_args = [call[0] for call in spy.calls]
    call_methods = [args[1] for args in call_args]

    assert call_methods == [
        "CreateMultipartUpload",
        "UploadPart",
        "UploadPart",
        "CompleteMultipartUpload",
    ]

    upload_args = call_args[1]
    upload_body = upload_args[2]["Body"]

    # NOTE: Why is this a BufferIO rather than the underlying buffer?!
    # NOTE: (Would have expected the result of BufferIO.read() -- str.)
    body_length = len(upload_body.getvalue())
    assert body_length == 5 * one_mb

    assert store.exists()
    assert store.load() == payload

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
    mse.write("testobject", "myhash")
    assert mse.exists("myhash")
    assert mse.load("myhash") == "testobject"
    assert "myhash" not in mse.cache


def test_ModelStorageEngine_caching(project_storage):
    mse = ModelStorageEngine(project_storage)
    with mse.cache_models():
        mse.write("testobject", "myhash")
        with mock.patch.object(mse, "_get_store") as get_store_mock:
            assert mse.load("myhash") == "testobject"
            assert not get_store_mock.called
        assert "myhash" in mse.cache
    # when cache_models goes out of scope the cache should be empty
    assert "myhash" not in mse.cache


DATA_DICT = OrderedDict(
    [
        ("entity_id", [1, 2]),
        ("as_of_date", [datetime.date(2017, 1, 1), datetime.date(2017, 1, 1)]),
        ("k_feature", [0.5, 0.4]),
        ("m_feature", [0.4, 0.5]),
        ("label", [0, 1]),
    ]
)

METADATA = {"label_name": "label"}


def matrix_stores():
    df = pd.DataFrame.from_dict(DATA_DICT).set_index(MatrixStore.indices)

    with tempfile.TemporaryDirectory() as tmpdir:
        project_storage = ProjectStorage(tmpdir)
        tmpcsv = os.path.join(tmpdir, "df.csv.gz")
        tmpyaml = os.path.join(tmpdir, "df.yaml")
        with open(tmpyaml, "w") as outfile:
            yaml.dump(METADATA, outfile, default_flow_style=False)
        df.to_csv(tmpcsv, compression="gzip")
        csv = CSVMatrixStore(project_storage, [], "df")
        # first test with caching
        with csv.cache():
            yield csv
        # with the caching out of scope they will be nuked
        # and this last version will not have any cache
        yield csv


def test_MatrixStore_empty():
    for matrix_store in matrix_stores():
        assert not matrix_store.empty


def test_MatrixStore_metadata():
    for matrix_store in matrix_stores():
        assert matrix_store.metadata == METADATA


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


def test_MatrixStore_labels_idempotency():
    for matrix_store in matrix_stores():
        assert matrix_store.labels.tolist() == [0, 1]
        assert matrix_store.labels.tolist() == [0, 1]


def test_MatrixStore_save():
    data = {
        "entity_id": [1, 2],
        "as_of_date": [pd.Timestamp(2017, 1, 1), pd.Timestamp(2017, 1, 1)],
        "feature_one": [0.5, 0.6],
        "feature_two": [0.5, 0.6],
        "label": [1, 0],
    }
    df = pd.DataFrame.from_dict(data)
    labels = df.pop("label")

    for matrix_store in matrix_stores():
        matrix_store.metadata = METADATA

        matrix_store.matrix_label_tuple = df, labels
        matrix_store.save()
        assert_frame_equal(matrix_store.design_matrix, df)


def test_MatrixStore_caching():
    for matrix_store in matrix_stores():
        with matrix_store.cache():
            matrix = matrix_store.design_matrix
            with mock.patch.object(matrix_store, "_load") as load_mock:
                assert_frame_equal(matrix_store.design_matrix, matrix)
                assert not load_mock.called


def test_as_of_dates(project_storage):
    data = {
        "entity_id": [1, 2, 1, 2],
        "feature_one": [0.5, 0.6, 0.5, 0.6],
        "feature_two": [0.5, 0.6, 0.5, 0.6],
        "as_of_date": [
            pd.Timestamp(2016, 1, 1),
            pd.Timestamp(2016, 1, 1),
            pd.Timestamp(2017, 1, 1),
            pd.Timestamp(2017, 1, 1),
        ],
        "label": [1, 0, 1, 0],
    }
    df = pd.DataFrame.from_dict(data)
    matrix_store = CSVMatrixStore(
        project_storage,
        [],
        "test",
        matrix=df,
        metadata={"indices": ["entity_id", "as_of_date"], "label_name": "label"},
    )
    assert matrix_store.as_of_dates == [
        datetime.date(2016, 1, 1),
        datetime.date(2017, 1, 1),
    ]


@mock_s3
def test_s3_save():
    client = boto3.client("s3")
    client.create_bucket(
        Bucket="fake-matrix-bucket",
        ACL="public-read-write",
        CreateBucketConfiguration={"LocationConstraint": "us-east-2"},
    )
    for example in matrix_stores():
        if not isinstance(example, CSVMatrixStore):
            continue
        project_storage = ProjectStorage("s3://fake-matrix-bucket")

        tosave = CSVMatrixStore(project_storage, [], "test")
        tosave.metadata = example.metadata
        tosave.matrix_label_tuple = example.matrix_label_tuple
        tosave.save()

        tocheck = CSVMatrixStore(project_storage, [], "test")
        assert tocheck.metadata == example.metadata
        assert tocheck.design_matrix.to_dict() == example.design_matrix.to_dict()
