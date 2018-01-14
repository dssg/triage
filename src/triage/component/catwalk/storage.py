import logging
import os
import pickle

import pandas as pd
import smart_open
import yaml

from .utils import (
    upload_object_to_key,
    key_exists,
    model_cache_key,
    download_object,
)

import s3fs

try:
    from urllib.parse import urlparse
except:
    from urlparse import urlparse

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
            pickle.dump(obj, f)

    def load(self):
        s3 = s3fs.S3FileSystem()
        with s3.open(self.path, 'rb') as f:
            return pickle.load(f)

    def delete(self):
        s3 = s3fs.S3FileSystem()
        s3.rm(self.path)


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
    def __init__(self, *args, **kwargs):
        super(S3ModelStorageEngine, self).__init__(*args, **kwargs)
        self.s3_conn = s3_conn

    def get_store(self, model_hash):
        full_path=os.path.join(self.project_path, 'trained_models', model_hash)
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

    def save_yaml(self, df, project_path, name):
        path_parsed = urlparse(project_path)
        scheme = path_parsed.scheme # If '' of 'file' is a regular file or 's3'
        if not scheme or scheme == 'file': # Local file
            with open(os.path.join(project_path, name + ".yaml"), "wb") as f:
                yaml.dump(df, f, encoding='utf-8')
        elif scheme == 's3':
            s3 = s3fs.S3FileSystem()
            with s3.open(os.path.join(project_path, name + ".yaml"), "wb") as f:
                yaml.dump(df, f, encoding='utf-8')
        else:
            raise ValueError(f"URL scheme not supported: {scheme} (from {os.path.join(project_path, name + '.yaml')})")

    def load_yaml(self, metadata_path):
        path_parsed = urlparse(metadata_path)
        scheme = path_parsed.scheme # If '' of 'file' is a regular file or 's3'
        if not scheme or scheme == 'file': # Local file
            with open(metadata_path, "r", encoding='utf-8') as f:
                metadata = yaml.load(f.read())
        elif scheme == 's3':
            s3 = s3fs.S3FileSystem()
            with s3.open(metadata_path, 'rb', encoding='utf-8') as f:
                metadata = yaml.load(f.read())
        else:
            raise ValueError(f"URL scheme not supported: {scheme} (from {metadata_path})")

    def __getstate__(self):
        # when we serialize (say, for multiprocessing),
        # we don't want the cached members to show up
        # as they can be enormous
        self._matrix = None
        self._labels = None
        self._metadata = None
        self._head_of_matrix = None
        return self.__dict__.copy()


class HDFMatrixStore(MatrixStore):

    def _get_head_of_matrix(self):
        try:
            hdf = pd.HDFStore(self.matrix_path)
            key = hdf.keys()[0]
            head_of_matrix = hdf.select(key, start=0, stop=1)
            head_of_matrix.set_index(self.metadata['indices'], inplace=True)
            self._head_of_matrix = head_of_matrix
        except pd.errors.EmptyDataError:
            self._head_of_matrix = None

    def _load(self):
        with smart_open.smart_open(self.matrix_path, "rb") as f:
            self._matrix = self._read_hdf_from_buffer(f)
        self._metadata = self.load_yaml(self.metadata_path)
        try:
            self._matrix.set_index(self._metadata['indices'], inplace=True)
        except Exception:
            pass

    def _read_hdf_from_buffer(self, buffer):
        with pd.HDFStore(
                "data.h5",
                mode="r",
                driver="H5FD_CORE",
                driver_core_backing_store=0,
                driver_core_image=buffer.read()) as store:

            if len(store.keys()) > 1:
                raise Exception('Ambiguous matrix store. More than one dataframe in the hdf file.')

            try:
                return store["matrix"]

            except KeyError:
                print("The hdf file should contain one and only key, matrix.")
                return store[store.keys()[0]]

    def _write_hdf_to_buffer(self, df):
        with pd.HDFStore(
                "data.h5",
                mode="w",
                driver="H5FD_CORE",
                driver_core_backing_store=0) as out:
            out["matrix"] = df
            return out._handle.get_file_image()

    def save(self, project_path, name):
        with smart_open.smart_open(os.path.join(project_path, name + ".h5"), "wb") as f:
            f.write(self._write_hdf_to_buffer(self.matrix))
        self.save_yaml(self.metadata, project_path, name)


class CSVMatrixStore(MatrixStore):
    def _get_head_of_matrix(self):
        try:
            head_of_matrix = pd.read_csv(self.matrix_path, nrows=1)
            head_of_matrix.set_index(self.metadata['indices'], inplace=True)
            self._head_of_matrix = head_of_matrix
        except pd.errors.EmptyDataError:
            self._head_of_matrix = None

    def _load(self):
        path_parsed = urlparse(self.matrix_path)
        scheme = path_parsed.scheme # If '' of 'file' is a regular file or 's3'
        if not scheme or scheme == 'file': # Local file
            with open(self.matrix_path, "r") as f:
                self._matrix = pd.read_csv(f)
        elif scheme == 's3':
            s3 = s3fs.S3FileSystem()
            with s3.open(self.matrix_path, 'rb') as f:
                self_matrix = pd.read_csv(f)
        else:
            raise ValueError(f"URL scheme not supported: {scheme} (from {self.matrix_path})")

        self._metadata = self.load_yaml(self.metadata_path)
        self._matrix.set_index(self.metadata['indices'], inplace=True)

    def save(self, project_path, name):
        path_parsed = urlparse(self.matrix_path)
        scheme = path_parsed.scheme # If '' of 'file' is a regular file or 's3'
        if not scheme or scheme == 'file': # Local file
            with open(os.path.join(project_path, name + ".csv"), "w") as f:
                self.matrix.to_csv(f)
        elif scheme == 's3':
            s3 = s3fs.S3FileSystem()
            with s3.open(os.path.join(project_path, name + ".csv"), "wb") as f:
                self.matrix.to_csv(f)
        else:
            raise ValueError(f"URL scheme not supported: {scheme} (from {os.path.join(project_path, name + '.csv')})")


        self.save_yaml(self.metadata, project_path, name)


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

    def save(self, project_path, name):
        return None
