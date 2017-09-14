from .utils import upload_object_to_key, key_exists, model_cache_key, download_object
import os
import pickle
import pandas
import yaml
import logging


class Store(object):
    def __init__(self, path):
        self.path = path

    def exists(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def write(self, obj):
        raise NotImplementedError


class S3Store(Store):
    def exists(self):
        return key_exists(self.path)

    def write(self, obj):
        upload_object_to_key(obj, self.path)

    def load(self):
        return download_object(self.path)

    def delete(self):
        self.path.delete()


class FSStore(Store):
    def exists(self):
        return os.path.isfile(self.path)

    def write(self, obj):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w+b') as f:
            pickle.dump(obj, f)

    def load(self):
        with open(self.path, 'rb') as f:
            return pickle.load(f)

    def delete(self):
        os.remove(self.path)


class MemoryStore(Store):
    store = None

    def exists(self):
        return self.store is not None

    def write(self, obj):
        self.store = obj

    def load(self):
        return self.store

    def delete(self):
        self.store = None


class ModelStorageEngine(object):
    def __init__(self, project_path):
        self.project_path = project_path

    def get_store(self, model_hash):
        pass


class S3ModelStorageEngine(ModelStorageEngine):
    def __init__(self, s3_conn, *args, **kwargs):
        super(S3ModelStorageEngine, self).__init__(*args, **kwargs)
        self.s3_conn = s3_conn

    def get_store(self, model_hash):
        return S3Store(model_cache_key(
            self.project_path,
            model_hash,
            self.s3_conn
        ))


class FSModelStorageEngine(ModelStorageEngine):
    def __init__(self, *args, **kwargs):
        super(FSModelStorageEngine, self).__init__(*args, **kwargs)
        os.makedirs(os.path.join(self.project_path, 'trained_models'), exist_ok=True)

    def get_store(self, model_hash):
        return FSStore('/'.join([
            self.project_path,
            'trained_models',
            model_hash
        ]))


class InMemoryModelStorageEngine(ModelStorageEngine):
    stores = {}

    def get_store(self, model_hash):
        if model_hash not in self.stores:
            self.stores[model_hash] = MemoryStore(model_hash)
        return self.stores[model_hash]


class MatrixStore(object):
    _labels = None

    def __init__(self, matrix_path=None, metadata_path=None):
        self.matrix_path = matrix_path
        self.metadata_path = metadata_path
        self._matrix = None
        self._metadata = None
        self._head_of_matrix = None

    @property
    def matrix(self):
        if self._matrix is None:
            self._load()
        return self._matrix

    @property
    def metadata(self):
        if self._metadata is None:
            self._load()
        return self._metadata

    @property
    def head_of_matrix(self):
        if self._head_of_matrix is None:
            self._get_head_of_matrix()
        return self._head_of_matrix

    @property
    def empty(self):
        if not os.path.isfile(self.matrix_path):
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


class HDFMatrixStore(MatrixStore):
    def _get_head_of_matrix(self):
        try:
            hdf = pandas.HDFStore(self.matrix_path)
            key = hdf.keys()[0]
            head_of_matrix = hdf.select(key, start=0, stop=1)
            head_of_matrix.set_index(self.metadata['indices'], inplace=True)
            self._head_of_matrix = head_of_matrix
        except pandas.error.EmptyDataError:
            self._head_of_matrix = None


    def _load(self):
        self._matrix = pandas.read_hdf(self.matrix_path, mode='r+')
        with open(self.metadata_path) as f:
            self._metadata = yaml.load(f)
        self._matrix.set_index(self.metadata['indices'], inplace=True)


class CSVMatrixStore(MatrixStore):
    def _get_head_of_matrix(self):
        try:
            head_of_matrix = pandas.read_csv(self.matrix_path, nrows=1)
            head_of_matrix.set_index(self.metadata['indices'], inplace=True)
            self._head_of_matrix = head_of_matrix
        except pandas.error.EmptyDataError:
            self._head_of_matrix = None

    def _load(self):
        self._matrix = pandas.read_csv(self.matrix_path)
        with open(self.metadata_path) as f:
            self._metadata = yaml.load(f)
        self._matrix.set_index(self.metadata['indices'], inplace=True)


class InMemoryMatrixStore(MatrixStore):
    def __init__(self, matrix, metadata, labels=None):
        self._matrix = matrix
        self._metadata = metadata
        self._labels = labels
        self._head_of_matrix = None

    def _get_head_of_matrix(self):
        self._head_of_matrix = self.matrix.head(n=1)

    @property
    def empty(self):
        head_of_matrix = self.head_of_matrix
        return head_of_matrix.empty

    @property
    def matrix(self):
        if self._metadata['indices'][0] in self._matrix.columns:
            self._matrix.set_index(self._metadata['indices'], inplace=True)
        return self._matrix
