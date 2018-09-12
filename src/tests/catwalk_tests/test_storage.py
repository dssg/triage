import os
import tempfile
import unittest
import yaml
from collections import OrderedDict

import pandas as pd
from moto import mock_s3

from triage.component.catwalk.storage import (
    CSVMatrixStore,
    FSStore,
    HDFMatrixStore,
    MemoryStore,
    S3Store,
)


class SomeClass(object):
    def __init__(self, val):
        self.val = val


def test_S3Store():
    with mock_s3():
        import boto3
        client = boto3.client('s3')
        client.create_bucket(Bucket='test_bucket', ACL='public-read-write')
        store = S3Store(path=f"s3://test_bucket/a_path")
        assert not store.exists()
        store.write('val'.encode('utf-8'))
        assert store.exists()
        newVal = store.load()
        assert newVal.decode('utf-8') == 'val'
        store.delete()
        assert not store.exists()


def test_FSStore():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, 'tmpfile')
        store = FSStore(tmpfile)
        assert not store.exists()
        store.write('val'.encode('utf-8'))
        assert store.exists()
        newVal = store.load()
        assert newVal.decode('utf-8') == 'val'
        store.delete()
        assert not store.exists()


def test_MemoryStore():
    store = MemoryStore(None)
    assert not store.exists()
    store.write('val'.encode('utf-8'))
    assert store.exists()
    newVal = store.load()
    assert newVal.decode('utf-8') == 'val'
    store.delete()
    assert not store.exists()


class MatrixStoreTest(unittest.TestCase):
    data_dict = OrderedDict([
        ('entity_id', [1, 2]),
        ('k_feature', [0.5, 0.4]),
        ('m_feature', [0.4, 0.5]),
        ('label', [0, 1])
    ])

    metadata = {
        'label_name': 'label',
        'indices': ['entity_id'],
    }

    def matrix_stores(self):
        df = pd.DataFrame.from_dict(self.data_dict).set_index(['entity_id'])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpcsv = os.path.join(tmpdir, 'df.csv')
            tmpyaml = os.path.join(tmpdir, 'metadata.yaml')
            tmphdf = os.path.join(tmpdir, 'df.h5')
            with open(tmpyaml, 'w') as outfile:
                yaml.dump(self.metadata, outfile, default_flow_style=False)
                df.to_csv(tmpcsv)
                df.to_hdf(tmphdf, 'matrix')
                csv = CSVMatrixStore(matrix_path=tmpcsv, metadata_path=tmpyaml)
                hdf = HDFMatrixStore(matrix_path=tmphdf, metadata_path=tmpyaml)
                assert csv.matrix.equals(hdf.matrix)
                yield from [csv, hdf]

    def test_MatrixStore_empty(self):
        for matrix_store in self.matrix_stores():
            assert not matrix_store.empty

    def test_MatrixStore_metadata(self):
        for matrix_store in self.matrix_stores():
            assert matrix_store.metadata == self.metadata

    def test_MatrixStore_head_of_matrix(self):
        for matrix_store in self.matrix_stores():
            assert matrix_store.head_of_matrix.to_dict() == {
                'k_feature': {1: 0.5},
                'm_feature': {1: 0.4},
                'label': {1: 0}
            }

    def test_MatrixStore_columns(self):
        for matrix_store in self.matrix_stores():
            assert matrix_store.columns() == ['k_feature', 'm_feature']

    def test_MatrixStore_resort_columns(self):
        for matrix_store in self.matrix_stores():
            result = matrix_store.\
                matrix_with_sorted_columns(
                    ['m_feature', 'k_feature']
                )\
                .values\
                .tolist()
            expected = [
                [0.4, 0.5],
                [0.5, 0.4]
            ]
            self.assertEqual(expected, result)

    def test_MatrixStore_already_sorted_columns(self):
        for matrix_store in self.matrix_stores():
            result = matrix_store.\
                matrix_with_sorted_columns(
                    ['k_feature', 'm_feature']
                )\
                .values\
                .tolist()
            expected = [
                [0.5, 0.4],
                [0.4, 0.5]
            ]
            self.assertEqual(expected, result)

    def test_MatrixStore_sorted_columns_subset(self):
        with self.assertRaises(ValueError):
            for matrix_store in self.matrix_stores():
                matrix_store.\
                    matrix_with_sorted_columns(['m_feature'])\
                    .values\
                    .tolist()

    def test_MatrixStore_sorted_columns_superset(self):
        with self.assertRaises(ValueError):
            for matrix_store in self.matrix_stores():
                matrix_store.\
                    matrix_with_sorted_columns(
                        ['k_feature', 'l_feature', 'm_feature']
                    )\
                    .values\
                    .tolist()

    def test_MatrixStore_sorted_columns_mismatch(self):
        with self.assertRaises(ValueError):
            for matrix_store in self.matrix_stores():
                matrix_store.\
                    matrix_with_sorted_columns(
                        ['k_feature', 'l_feature']
                    )\
                    .values\
                    .tolist()

    def test_as_of_dates_entity_index(self):
        data = {
            'entity_id': [1, 2],
            'feature_one': [0.5, 0.6],
            'feature_two': [0.5, 0.6],
        }
        inmemory = CSVMatrixStore(matrix_path='memory://', metadata_path='memory://')
        inmemory.matrix = pd.DataFrame.from_dict(data)
        inmemory.metadata = {'end_time': '2016-01-01', 'indices': ['entity_id']}

        self.assertEqual(inmemory.as_of_dates, ['2016-01-01'])

    def test_as_of_dates_entity_date_index(self):
        data = {
            'entity_id': [1, 2, 1, 2],
            'feature_one': [0.5, 0.6, 0.5, 0.6],
            'feature_two': [0.5, 0.6, 0.5, 0.6],
            'as_of_date': ['2016-01-01', '2016-01-01', '2017-01-01', '2017-01-01']
        }
        inmemory = CSVMatrixStore(matrix_path='memory://', metadata_path='memory://')
        inmemory.matrix = pd.DataFrame.from_dict(data).set_index(['entity_id', 'as_of_date'])
        inmemory.metadata = {'indices': ['entity_id', 'as_of_date']}

        self.assertEqual(inmemory.as_of_dates, ['2016-01-01', '2017-01-01'])

    def test_s3_save(self):
        with mock_s3():
            import boto3
            client = boto3.client('s3')
            client.create_bucket(Bucket='fake-matrix-bucket', ACL='public-read-write')
            example = next(self.matrix_stores())

            tosave = CSVMatrixStore(
                matrix_path='s3://fake-matrix-bucket/test.csv',
                metadata_path='s3://fake-matrix-bucket/test.yaml'
            )
            tosave.matrix = example.matrix
            tosave.metadata = example.metadata
            tosave.save()

            tocheck = CSVMatrixStore(
                matrix_path='s3://fake-matrix-bucket/test.csv',
                metadata_path='s3://fake-matrix-bucket/test.yaml'
            )
            assert tocheck.metadata == example.metadata
            assert tocheck.matrix.to_dict() == example.matrix.to_dict()
