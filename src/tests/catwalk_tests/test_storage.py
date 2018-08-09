import os
import tempfile
import unittest
import yaml
from collections import OrderedDict

import pandas as pd
from moto import mock_s3, mock_s3_deprecated

from triage.component.catwalk.storage import (
    CSVMatrixStore,
    FSStore,
    HDFMatrixStore,
    ModelStore,
    MatrixStore,
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

    def matrix_store(self):
        data_dict = OrderedDict([
            ('entity_id', [1, 2]),
            ('k_feature', [0.5, 0.4]),
            ('m_feature', [0.4, 0.5]),
            ('label', [0, 1])
        ])
        df = pd.DataFrame.from_dict(data_dict).set_index(['entity_id'])
        metadata = {
            'label_name': 'label',
            'indices': ['entity_id'],
        }

        inmemory = MatrixStore(matrix_path='memory://', metadata_path='memory://', matrix=df, metadata=metadata)
        inmemory.save()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpcsv = os.path.join(tmpdir, 'df.csv')
            tmpyaml = os.path.join(tmpdir, 'metadata.yaml')
            tmphdf = os.path.join(tmpdir, 'df.h5')
            with open(tmpyaml, 'w') as outfile:
                yaml.dump(metadata, outfile, default_flow_style=False)
                df.to_csv(tmpcsv)
                df.to_hdf(tmphdf, 'matrix')
                csv = CSVMatrixStore(matrix_path=tmpcsv, metadata_path=tmpyaml)
                hdf = HDFMatrixStore(matrix_path=tmphdf, metadata_path=tmpyaml)

                assert csv.matrix.to_dict() == inmemory.matrix.to_dict()
                assert hdf.matrix.to_dict() == inmemory.matrix.to_dict()

                assert csv.metadata == inmemory.metadata
                assert hdf.metadata == inmemory.metadata

                assert csv.head_of_matrix.to_dict() == inmemory.head_of_matrix.to_dict()
                assert hdf.head_of_matrix.to_dict() == inmemory.head_of_matrix.to_dict()

                assert csv.empty == inmemory.empty
                assert hdf.empty == inmemory.empty

                assert csv.labels().to_dict() == inmemory.labels().to_dict()
                assert hdf.labels().to_dict() == inmemory.labels().to_dict()

        matrix_store = [inmemory, csv, hdf]

        return matrix_store

    def test_MatrixStore_resort_columns(self):
        matrix_store_list = self.matrix_store()
        for matrix_store in matrix_store_list:
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
        matrix_store_list = self.matrix_store()
        for matrix_store in matrix_store_list:
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
            matrix_store_list = self.matrix_store()
            for matrix_store in matrix_store_list:
                matrix_store.\
                    matrix_with_sorted_columns(['m_feature'])\
                    .values\
                    .tolist()

    def test_MatrixStore_sorted_columns_superset(self):
        with self.assertRaises(ValueError):
            matrix_store_list = self.matrix_store()
            for matrix_store in matrix_store_list:
                matrix_store.\
                    matrix_with_sorted_columns(
                        ['k_feature', 'l_feature', 'm_feature']
                    )\
                    .values\
                    .tolist()

    def test_MatrixStore_sorted_columns_mismatch(self):
        with self.assertRaises(ValueError):
            matrix_store_list = self.matrix_store()
            for matrix_store in matrix_store_list:
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
        inmemory = MatrixStore(matrix_path='memory://', metadata_path='memory://')
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
        inmemory = MatrixStore(matrix_path='memory://', metadata_path='memory://')
        inmemory.matrix = pd.DataFrame.from_dict(data).set_index(['entity_id', 'as_of_date'])
        inmemory.metadata = {'indices': ['entity_id', 'as_of_date']}

        self.assertEqual(inmemory.as_of_dates, ['2016-01-01', '2017-01-01'])

    def test_s3_save(self):
        with mock_s3():
            import boto3
            client = boto3.client('s3')
            client.create_bucket(Bucket='fake-matrix-bucket', ACL='public-read-write')
            example = self.matrix_store()[0]
            
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
