# coding: utf-8

import os
import logging
from sklearn.externals import joblib
from urllib.parse import urlparse
from triage.component.results_schema import TestEvaluation, TrainEvaluation, \
    TestPrediction, TrainPrediction

import pandas as pd
import s3fs
import sys
import yaml


class Store(object):
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return f"{self.__class__.__name__}(path={self.path})"

    def __repr__(self):
        return str(self)

    def exists(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def write(self, obj):
        raise NotImplementedError


class S3Store(Store):

    def exists(self):
        s3 = s3fs.S3FileSystem()
        return s3.exists(self.path)

    def write(self, obj):
        s3 = s3fs.S3FileSystem()
        with s3.open(self.path, 'wb') as f:
            joblib.dump(obj, f, compress=True)

    def load(self):
        s3 = s3fs.S3FileSystem()
        with s3.open(self.path, 'rb') as f:
            return joblib.load(f)

    def delete(self):
        s3 = s3fs.S3FileSystem()
        s3.rm(self.path)


class FSStore(Store):
    def exists(self):
        return os.path.isfile(self.path)

    def write(self, obj):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w+b') as f:
            joblib.dump(obj, f, compress=True)

    def load(self):
        with open(self.path, 'rb') as f:
            return joblib.load(f)

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
    def __init__(self, *args, **kwargs):
        super(S3ModelStorageEngine, self).__init__(*args, **kwargs)

    def get_store(self, model_hash):
        full_path = os.path.join(self.project_path, 'trained_models', model_hash)
        return S3Store(path=full_path)


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
        self.matrix = None
        self.metadata = None
        self.head_of_matrix = None

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
        if self.__head_of_matrix is None:
            self.__head_of_matrix = self._get_head_of_matrix()
        return self.__head_of_matrix

    @head_of_matrix.setter
    def head_of_matrix(self, head_of_matrix):
        self.__head_of_matrix = head_of_matrix

    @property
    def empty(self):
        path_parsed = urlparse(self.matrix_path)
        scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'

        if scheme in ('', 'file') and (not os.path.exists(self.matrix_path)):
            return True
        elif scheme == 's3' and (not s3fs.S3FileSystem().exists(self.matrix_path)):
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

    def save_metadata(self, df, project_path, name):
        path_parsed = urlparse(project_path)
        scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'
        if not scheme or scheme == 'file':  # Local file
            with open(os.path.join(project_path, name + ".yaml"), "wb") as f:
                yaml.dump(df, f, encoding='utf-8')
        elif scheme == 's3':
            s3 = s3fs.S3FileSystem()
            with s3.open(os.path.join(project_path, name + ".yaml"), "wb") as f:
                yaml.dump(df, f, encoding='utf-8')
        else:
            raise ValueError(f"""
                  URL scheme not supported:
                  {scheme} (from {os.path.join(project_path, name + '.yaml')})
            """)

    def load_metadata(self):
        path_parsed = urlparse(self.metadata_path)
        scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'
        if not scheme or scheme == 'file':  # Local file
            with open(self.metadata_path, "r", encoding='utf-8') as f:
                metadata = yaml.load(f.read())
        elif scheme == 's3':
            s3 = s3fs.S3FileSystem()
            with s3.open(self.metadata_path, 'rb', encoding='utf-8') as f:
                metadata = yaml.load(f.read())
        else:
            raise ValueError(f"""
                  URL scheme not supported:
                  {scheme} (from {self.metadata_path})"
            """)

        return metadata

    def __getstate__(self):
        # when we serialize (say, for multiprocessing),
        # we don't want the cached members to show up
        # as they can be enormous
        self.matrix = None
        self._labels = None
        self.metadata = None
        self.head_of_matrix = None
        return self.__dict__.copy()


class HDFMatrixStore(MatrixStore):

    def __init__(self, matrix_path=None, metadata_path=None):
        super().__init__(matrix_path, metadata_path)
        self.metadata = self.load_metadata()

    def _get_head_of_matrix(self):
        try:
            head_of_matrix = pd.read_hdf(self.matrix_path, start=0, stop=1)
            # Is the index already in place?
            if head_of_matrix.index.name not in self.metadata['indices']:
                head_of_matrix.set_index(self.metadata['indices'], inplace=True)
        except pd.errors.EmptyDataError:
            head_of_matrix = None

        return head_of_matrix

    def _load(self):
        path_parsed = urlparse(self.matrix_path)
        scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'

        if scheme in ('', 'file'):  # Local file
            matrix = pd.read_hdf(self.matrix_path)
        else:
            raise ValueError(f"""
                  URL scheme not supported:
                  {scheme} (from {self.matrix_path})
            """)
        # Is the index already in place?
        if matrix.index.name not in self.metadata['indices']:
            matrix.set_index(self.metadata['indices'], inplace=True)

        return matrix

    def save(self, project_path, name):
        path_parsed = urlparse(project_path)
        scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'

        if self.scheme in ('', 'file'):  # Local file
            with open(os.path.join(project_path, name + ".h5"), "w") as f:
                self.matrix.to_hdf(f, format='table', mode='w')
        else:
            raise ValueError(f"""
                  URL scheme not supported:
                  {scheme} (from {os.path.join(project_path, name + '.h5')})
            """)

        self.save_metadata(self.metadata, project_path, name)


class CSVMatrixStore(MatrixStore):

    def __init__(self, matrix_path=None, metadata_path=None):
        super().__init__(matrix_path, metadata_path)

    def _get_head_of_matrix(self):
        path_parsed = urlparse(self.matrix_path)
        scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'

        try:
            if scheme in ('', 'file'):  # Local file
                with open(self.matrix_path, "r") as f:
                    head_of_matrix = pd.read_csv(f, nrows=1)
            elif scheme == 's3':
                s3 = s3fs.S3FileSystem()
                with s3.open(self.matrix_path, 'rb') as f:
                    head_of_matrix = pd.read_csv(f, nrows=1)
            else:
                raise ValueError(f"URL scheme not supported: {scheme} (from {self.matrix_path})")

            head_of_matrix.set_index(self.metadata['indices'], inplace=True)

        except FileNotFoundError as fnfe:
            logging.exception(f"Matrix isn't there: {fnfe}")
            logging.exception("Returning Empty data frame")
            head_of_matrix = pd.DataFrame()

        return head_of_matrix

    def _load(self):
        path_parsed = urlparse(self.matrix_path)
        scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'

        if scheme in ('', 'file'):  # Local file
            with open(self.matrix_path, "r") as f:
                matrix = pd.read_csv(f)
        elif scheme == 's3':
            s3 = s3fs.S3FileSystem()
            with s3.open(self.matrix_path, 'rb') as f:
                matrix = pd.read_csv(f)

        matrix.set_index(self.metadata['indices'], inplace=True)

        return matrix

    def save(self, project_path, name):
        path_parsed = urlparse(project_path)
        scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'

        if not scheme or scheme == 'file':  # Local file
            with open(os.path.join(project_path, name + ".csv"), "w") as f:
                self.matrix.to_csv(f)
        elif scheme == 's3':
            bytes_to_write = self.matrix.to_csv(None).encode()
            s3 = s3fs.S3FileSystem()
            with s3.open(os.path.join(project_path, name + ".csv"), "wb") as f:
                f.write(bytes_to_write)
        else:
            raise ValueError(f"URL scheme not supported: {scheme} "
                             "(from {os.path.join(project_path, name + '.csv')})")

        self.save_metadata(self.metadata, project_path, name)


class InMemoryMatrixStore(MatrixStore):
    def __init__(self, matrix, metadata, labels=None):
        super().__init__()
        self.matrix = matrix
        self.metadata = metadata
        if self.matrix.index.names != self.metadata['indices']:
            self.matrix.set_index(self.metadata['indices'], inplace=True)
        self._labels = labels
        self.head_of_matrix = None

    def _get_head_of_matrix(self):
        return self.matrix.head(n=1)

    @property
    def empty(self):
        head_of_matrix = self.head_of_matrix
        return head_of_matrix.empty

    def save(self, project_path, name):
        return None


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
