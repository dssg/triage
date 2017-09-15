from catwalk.storage import S3Store, FSStore, MemoryStore, InMemoryMatrixStore, HDFMatrixStore, CSVMatrixStore
from moto import mock_s3
import tempfile
import boto3
import os
import pandas
from collections import OrderedDict
import unittest
import yaml

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

        inmemory = InMemoryMatrixStore(matrix=df, metadata=metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpcsv = os.path.join(tmpdir, 'df.csv')
            tmpyaml = os.path.join(tmpdir, 'metadata.yaml')
            tmphdf = os.path.join(tmpdir, 'df.h5')
            with open(tmpyaml, 'w') as outfile:
                yaml.dump(metadata, outfile, default_flow_style=False)
                df.to_csv(tmpcsv, index=False)
                df.to_hdf(tmphdf, 'test')
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


        matrix_store = [inmemory, hdf, csv]
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
        matrix = InMemoryMatrixStore(
            matrix=pandas.DataFrame.from_dict(data),
            metadata={'end_time': '2016-01-01', 'indices': ['entity_id']}
        )

        self.assertEqual(matrix.as_of_dates, ['2016-01-01'])

    def test_as_of_dates_entity_date_index(self):
        data = {
            'entity_id': [1, 2, 1, 2],
            'feature_one': [0.5, 0.6, 0.5, 0.6],
            'feature_two': [0.5, 0.6, 0.5, 0.6],
            'as_of_date': ['2016-01-01', '2016-01-01', '2017-01-01', '2017-01-01']
        }
        matrix = InMemoryMatrixStore(
            matrix=pandas.DataFrame.from_dict(data),
            metadata={'indices': ['entity_id', 'as_of_date']}
        )

        self.assertEqual(matrix.as_of_dates, ['2016-01-01', '2017-01-01'])
