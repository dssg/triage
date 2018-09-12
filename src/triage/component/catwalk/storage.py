# coding: utf-8

from io import BytesIO
import os
from os.path import dirname
import pathlib
import logging
from sklearn.externals import joblib
from urllib.parse import urlparse
from triage.component.results_schema import TestEvaluation, TrainEvaluation, \
    TestPrediction, TrainPrediction

import pandas as pd
import s3fs
import yaml


class Store(object):
    def __init__(self, path):
        self.path = path

    @classmethod
    def factory(self, path):
        path_parsed = urlparse(path)
        scheme = path_parsed.scheme

        if scheme in ('', 'file'):
            return FSStore(path)
        elif scheme == 's3':
            return S3Store(path)
        elif scheme == 'memory':
            return MemoryStore(path)
        else:
            raise ValueError('Unable to infer correct Store from project path')

    def __str__(self):
        return f"{self.__class__.__name__}(path={self.path})"

    def __repr__(self):
        return str(self)

    def exists(self):
        raise NotImplementedError

    def load(self):
        with self.open('rb') as fd:
            return fd.read()

    def write(self, bytestream):
        with self.open('wb') as fd:
            fd.write(bytestream)

    def open(self, *args, **kwargs):
        raise NotImplementedError


class S3Store(Store):

    def exists(self):
        s3 = s3fs.S3FileSystem()
        return s3.exists(self.path)

    def delete(self):
        s3 = s3fs.S3FileSystem()
        s3.rm(self.path)

    def open(self, *args, **kwargs):
        s3 = s3fs.S3FileSystem()
        return s3.open(self.path, *args, **kwargs)


class FSStore(Store):
    def exists(self):
        return os.path.isfile(self.path)

    def delete(self):
        os.remove(self.path)

    def open(self, *args, **kwargs):
        os.makedirs(dirname(self.path), exist_ok=True)
        return open(self.path, *args, **kwargs)


class MemoryStore(Store):
    store = None

    def exists(self):
        return self.store is not None

    def delete(self):
        self.store = None

    def write(self, bytestream):
        self.store = bytestream

    def load(self):
        return self.store

    def open(self, *args, **kwargs):
        raise ValueError('MemoryStore objects cannot be opened and closed like files'
                         'Use write/load methods instead.')


class ModelStore():
    def __init__(self, store):
        self.store = store

    def write(self, obj):
        with self.store.open('wb') as fd:
            joblib.dump(obj, fd, compress=True)

    def load(self):
        with self.store.open('rb') as fd:
            return joblib.load(fd)

    def exists(self):
        return self.store.exists()

    def delete(self):
        return self.store.delete()


class ModelStorageEngine(object):
    def __init__(self, project_path):
        self.project_path = project_path
        self.model_dir = 'trained_models'

    @classmethod
    def factory(self, path):
        path_parsed = urlparse(path)
        scheme = path_parsed.scheme

        if scheme in ('', 'file'):
            return FSModelStorageEngine(path)
        elif scheme == 's3':
            return S3ModelStorageEngine(path)
        elif scheme == 'memory':
            return InMemoryModelStorageEngine(path)
        else:
            raise ValueError('Unable to infer correct ModelStorageEngine from path')


    def get_store(self, model_hash):
        pass


class S3ModelStorageEngine(ModelStorageEngine):
    def get_store(self, model_hash):
        return ModelStore(S3Store(str(pathlib.PurePosixPath(self.project_path, self.model_dir, model_hash))))


class FSModelStorageEngine(ModelStorageEngine):
    def get_store(self, model_hash):
        return ModelStore(FSStore(pathlib.Path(self.project_path, self.model_dir, model_hash)))


class InMemoryModelStorageEngine(ModelStorageEngine):
    stores = {}

    def get_store(self, model_hash):
        if model_hash not in self.stores:
            self.stores[model_hash] = MemoryStore(model_hash)
        return self.stores[model_hash]


class MatrixStore(object):
    _labels = None

    def __init__(self, matrix_path='memory://', metadata_path='memory://', matrix=None, metadata=None):
        self.matrix_path = matrix_path
        self.metadata_path = metadata_path

        self.matrix_base_store = Store.factory(self.matrix_path)
        self.metadata_base_store = Store.factory(self.metadata_path)

        self.matrix = matrix
        self.metadata = metadata

    @property
    def matrix(self):
        if self.__matrix is None:
            self.__matrix = self._load()
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix):
        self.__matrix = matrix

    @property
    def metadata(self):
        if self.__metadata is None:
            self.__metadata = self.load_metadata()
        return self.__metadata

    @metadata.setter
    def metadata(self, metadata):
        self.__metadata = metadata

    @property
    def head_of_matrix(self):
        return self.matrix.head(1)

    @property
    def empty(self):
        if not self.matrix_base_store.exists():
            return True
        else:
            head_of_matrix = self.head_of_matrix
            return head_of_matrix.empty

    def columns(self, include_label=False):
        head_of_matrix = self.head_of_matrix
        columns = head_of_matrix.columns.tolist()
        if include_label:
            return columns
        else:
            return [
                col for col in columns
                if col != self.metadata['label_name']
            ]

    def labels(self):
        if self._labels is not None:
            logging.debug('using stored labels')
            return self._labels
        else:
            logging.debug('popping labels from matrix')
            self._labels = self.matrix.pop(self.metadata['label_name'])
            return self._labels

    @property
    def uuid(self):
        return self.metadata['metta-uuid']

    @property
    def as_of_dates(self):
        if 'as_of_date' in self.matrix.index.names:
            return sorted(list(set([as_of_date for entity_id, as_of_date in self.matrix.index])))
        else:
            return [self.metadata['end_time']]

    @property
    def num_entities(self):
        if self.matrix.index.names == ['entity_id']:
            return len(self.matrix.index.values)
        elif 'entity_id' in self.matrix.index.names:
            return len(self.matrix.index.levels[self.matrix.index.names.index('entity_id')])

    @property
    def matrix_type(self):
        if self.metadata['matrix_type'] == 'train':
            return TrainMatrixType
        elif self.metadata['matrix_type'] == 'test':
            return TestMatrixType
        else:
            raise Exception('''matrix metadata for matrix {} must contain 'matrix_type'
             = "train" or "test" '''.format(self.uuid))

    def matrix_with_sorted_columns(self, columns):
        columnset = set(self.columns())
        desired_columnset = set(columns)
        if columnset == desired_columnset:
            if self.columns() != columns:
                logging.warning('Column orders not the same, re-ordering')
            return self.matrix[columns]
        else:
            if columnset.issuperset(desired_columnset):
                raise ValueError('''
                    Columnset is superset of desired columnset. Extra items: %s
                ''', columnset - desired_columnset)
            elif columnset.issubset(desired_columnset):
                raise ValueError('''
                    Columnset is subset of desired columnset. Extra items: %s
                ''', desired_columnset - columnset)
            else:
                raise ValueError('''
                    Columnset and desired columnset mismatch. Unique items: %s
                ''', columnset ^ desired_columnset)

    def load_metadata(self):
        with self.metadata_base_store.open('rb') as fd:
            return yaml.load(fd)

    def save(self):
        raise NotImplementedError

    def __getstate__(self):
        # when we serialize (say, for multiprocessing),
        # we don't want the cached members to show up
        # as they can be enormous
        self.matrix = None
        self._labels = None
        self.metadata = None
        return self.__dict__.copy()


class HDFMatrixStore(MatrixStore):

    def __init__(self, matrix_path=None, metadata_path=None):
        super().__init__(matrix_path, metadata_path)
        if isinstance(self.matrix_base_store, S3Store):
            raise ValueError('HDFMatrixStore cannot be used with S3')
        self.metadata = self.load_metadata()

    @property
    def head_of_matrix(self):
        try:
            head_of_matrix = pd.read_hdf(self.matrix_path, start=0, stop=1)
            # Is the index already in place?
            if head_of_matrix.index.name not in self.metadata['indices']:
                head_of_matrix.set_index(self.metadata['indices'], inplace=True)
        except pd.errors.EmptyDataError:
            head_of_matrix = None

        return head_of_matrix

    def _load(self):
        matrix = pd.read_hdf(self.matrix_path)

        # Is the index already in place?
        if matrix.index.name not in self.metadata['indices']:
            matrix.set_index(self.metadata['indices'], inplace=True)

        return matrix

    def save(self):
        with self.matrix_base_store.open('wb') as fd:
            self.matrix.to_hdf(fd, format='table', mode='wb')

        with self.metadata_base_store.open('wb') as fd:
            yaml.dump(self.metadata, fd, encoding='utf-8')


class CSVMatrixStore(MatrixStore):

    def __init__(self, matrix_path=None, metadata_path=None):
        super().__init__(matrix_path, metadata_path)

    @property
    def head_of_matrix(self):
        try:
            with self.matrix_base_store.open('rb') as fd:
                head_of_matrix = pd.read_csv(fd, nrows=1)
                head_of_matrix.set_index(self.metadata['indices'], inplace=True)
        except FileNotFoundError as fnfe:
            logging.exception(f"Matrix isn't there: {fnfe}")
            logging.exception("Returning Empty data frame")
            head_of_matrix = pd.DataFrame()

        return head_of_matrix

    def _load(self):
        with self.matrix_base_store.open('rb') as fd:
            matrix = pd.read_csv(fd)

        matrix.set_index(self.metadata['indices'], inplace=True)

        return matrix

    def save(self):
        self.matrix_base_store.write(self.matrix.to_csv(None).encode('utf-8'))
        with self.metadata_base_store.open('wb') as fd:
            yaml.dump(self.metadata, fd, encoding='utf-8')


class TestMatrixType(object):
    string_name = 'test'
    evaluation_obj = TestEvaluation
    prediction_obj = TestPrediction
    is_test = True


class TrainMatrixType(object):
    string_name = 'train'
    evaluation_obj = TrainEvaluation
    prediction_obj = TrainPrediction
    is_test = False
