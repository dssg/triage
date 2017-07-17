from catwalk.storage import S3Store, FSStore, MemoryStore, InMemoryMatrixStore
from moto import mock_s3
import tempfile
import boto3
import os
import pandas
from collections import OrderedDict
import unittest


class SomeClass(object):
    def __init__(self, val):
        self.val = val


def test_S3Store():
    with mock_s3():
        s3_conn = boto3.resource('s3')
        s3_conn.create_bucket(Bucket='a-bucket')
        store = S3Store(s3_conn.Object('a-bucket', 'a-path'))
        assert not store.exists()
        store.write(SomeClass('val'))
        assert store.exists()
        newVal = store.load()
        assert newVal.val == 'val'
        store.delete()
        assert not store.exists()


def test_FSStore():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, 'tmpfile')
        store = FSStore(tmpfile)
        assert not store.exists()
        store.write(SomeClass('val'))
        assert store.exists()
        newVal = store.load()
        assert newVal.val == 'val'
        store.delete()
        assert not store.exists()


def test_MemoryStore():
    store = MemoryStore(None)
    assert not store.exists()
    store.write(SomeClass('val'))
    assert store.exists()
    newVal = store.load()
    assert newVal.val == 'val'
    store.delete()
    assert not store.exists()


class MatrixStoreTest(unittest.TestCase):
    def matrix_store(self):
        data_dict = OrderedDict([
            ('entity_id', [1, 2]),
            ('k_feature', [0.5, 0.4]),
            ('m_feature', [0.4, 0.5]),
            ('label', [0, 1])
        ])
        df = pandas.DataFrame.from_dict(data_dict)
        metadata = {
            'label_name': 'label',
            'indices': ['entity_id'],
        }
        matrix_store = InMemoryMatrixStore(matrix=df, metadata=metadata)
        return matrix_store

    def test_MatrixStore_resort_columns(self):
        result = self.matrix_store().\
            matrix_with_sorted_columns(
                ['entity_id', 'm_feature', 'k_feature']
            )\
            .values\
            .tolist()
        expected = [
            [1, 0.4, 0.5],
            [2, 0.5, 0.4]
        ]
        self.assertEqual(expected, result)

    def test_MatrixStore_already_sorted_columns(self):
        result = self.matrix_store().\
            matrix_with_sorted_columns(
                ['entity_id', 'k_feature', 'm_feature']
            )\
            .values\
            .tolist()
        expected = [
            [1, 0.5, 0.4],
            [2, 0.4, 0.5]
        ]
        self.assertEqual(expected, result)

    def test_MatrixStore_sorted_columns_subset(self):
        with self.assertRaises(ValueError):
            self.matrix_store().\
                matrix_with_sorted_columns(['entity_id', 'm_feature'])\
                .values\
                .tolist()

    def test_MatrixStore_sorted_columns_superset(self):
        with self.assertRaises(ValueError):
            self.matrix_store().\
                matrix_with_sorted_columns(
                    ['entity_id', 'k_feature', 'l_feature', 'm_feature']
                )\
                .values\
                .tolist()

    def test_MatrixStore_sorted_columns_mismatch(self):
        with self.assertRaises(ValueError):
            self.matrix_store().\
                matrix_with_sorted_columns(
                    ['entity_id', 'k_feature', 'l_feature']
                )\
                .values\
                .tolist()
