# coding: utf-8

import os
from os.path import dirname
import pathlib
import logging
from sklearn.externals import joblib
from urllib.parse import urlparse
from triage.component.results_schema import (
    TestEvaluation,
    TrainEvaluation,
    TestPrediction,
    TrainPrediction,
)
from triage.util.pandas import downcast_matrix

import pandas as pd
import s3fs
import yaml


class Store(object):
    """Base class for classes which know how to access a file in a preset medium.

    Used to hold references to persisted objects with knowledge about how they can be accessed.
    without loading them into memory. In this way, they can be easily and quickly serialized
    across processes but centralize the reading/writing code.

    Each subclass be scoped to a specific storage medium (e.g. Filesystem, S3)
        and implement the access methods for that medium.

    Implements write/load methods for interacting directly using bytestreams,
        plus an open method that works as an open filehandle.
    """

    def __init__(self, *pathparts):
        self.pathparts = pathparts

    @classmethod
    def factory(self, *pathparts):
        path_parsed = urlparse(pathparts[0])
        scheme = path_parsed.scheme

        if scheme in ("", "file"):
            return FSStore(*pathparts)
        elif scheme == "s3":
            return S3Store(*pathparts)
        else:
            raise ValueError("Unable to infer correct Store from project path")

    def __str__(self):
        return f"{self.__class__.__name__}(path={self.path})"

    def __repr__(self):
        return str(self)

    def exists(self):
        raise NotImplementedError

    def load(self):
        with self.open("rb") as fd:
            return fd.read()

    def write(self, bytestream):
        with self.open("wb") as fd:
            fd.write(bytestream)

    def open(self, *args, **kwargs):
        raise NotImplementedError


class S3Store(Store):
    """Store an object in S3.

    Example:
    ```
    store = S3Store('s3://my-bucket', 'models', 'model.pkl')
    return store.load()
    ```

    Args:
        *pathparts: A variable length list of components of the path, to be processed in order.
            All components will be joined using PurePosixPath to create the final path.
    """

    def __init__(self, *pathparts):
        self.path = str(
            pathlib.PurePosixPath(pathparts[0].replace("s3://", ""), *pathparts[1:])
        )

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
    """Store an object on the local filesystem.

    Example:
    ```
    store = FSStore('/mnt', 'models', 'model.pkl')
    return store.load()
    ```

    Args:
        *pathparts: A variable length list of components of the path, to be processed in order.
            All components will be joined using pathlib.Path to create the final path
                using the correct separator for the operating system. However, if you pass
                components that already contain a separator, those separators won't be modified
    """

    def __init__(self, *pathparts):
        self.path = pathlib.Path(*pathparts)
        os.makedirs(dirname(self.path), exist_ok=True)

    def exists(self):
        return os.path.isfile(self.path)

    def delete(self):
        os.remove(self.path)

    def open(self, *args, **kwargs):
        return open(self.path, *args, **kwargs)


class ProjectStorage(object):
    """Store and access files associated with a project.

    Args:
        project_path (string): The base path for all files in the project.
            The scheme prefix of the path will determine the storage medium.
    """

    def __init__(self, project_path):
        self.project_path = project_path
        self.storage_class = Store.factory(self.project_path).__class__

    def get_store(self, directories, leaf_filename):
        """Return a storage object for one filename

        Args:
        directories (list): A list of subdirectories
        leaf_filename (string): The filename without any directory information

        Returns:
            triage.component.catwalk.storage.Store object
        """
        return self.storage_class(self.project_path, *directories, leaf_filename)

    def matrix_storage_engine(self, matrix_storage_class=None, matrix_directory=None):
        """Return a matrix storage engine bound to this project's storage

        Args:
            matrix_storage_class (class) A subclass of MatrixStore
            matrix_directory (string, optional) A directory to store matrices.
                If not passed will allow the MatrixStorageEngine to decide
        Returns: triage.component.catwalk.storage.MatrixStorageEngine
        """
        return MatrixStorageEngine(self, matrix_storage_class, matrix_directory)

    def model_storage_engine(self, model_directory=None):
        """Return a model storage engine bound to this project's storage

        Args:
            model_directory (string, optional) A directory to store models
                If not passed will allow the ModelStorageEngine to decide
        Returns: triage.component.catwalk.storage.ModelStorageEngine
        """
        return ModelStorageEngine(self, model_directory)


class ModelStorageEngine(object):
    """Store arbitrary models in a given project storage using joblib

    Args:
        project_storage (triage.component.catwalk.storage.ProjectStorage)
            A project file storage engine
        model_directory (string, optional) A directory name for models. Defaults to 'trained_models'
    """

    def __init__(self, project_storage, model_directory=None):
        self.project_storage = project_storage
        self.directories = [model_directory or "trained_models"]

    def write(self, obj, model_hash):
        """Persist a model object using joblib. Also performs compression

        Args:
            obj (object) A picklable model object
            model_hash (string) An identifier, unique within this project, for the model
        """
        with self._get_store(model_hash).open("wb") as fd:
            joblib.dump(obj, fd, compress=True)

    def load(self, model_hash):
        """Load a model object using joblib

        Args:
            model_hash (string) An identifier, unique within this project, for the model

        Returns: (object) A model object
        """
        with self._get_store(model_hash).open("rb") as fd:
            return joblib.load(fd)

    def exists(self, model_hash):
        """Check whether the model is persisted

        Args:
            model_hash (string) An identifier, unique within this project, for the model

        Returns: (bool) Whether or not a model by that identifier exists in project storage
        """
        return self._get_store(model_hash).exists()

    def delete(self, model_hash):
        """Delete the model identified by this hash from project storage

        Args:
            model_hash (string) An identifier, unique within this project, for the model
        """
        return self._get_store(model_hash).delete()

    def _get_store(self, model_hash):
        return self.project_storage.get_store(self.directories, model_hash)


class MatrixStorageEngine(object):
    """Store matrices in a given project storage

    Args:
        project_storage (triage.component.catwalk.storage.ProjectStorage)
            A project file storage engine
        matrix_storage_class (class) A subclass of MatrixStore
        matrix_directory (string, optional) A directory to store matrices. Defaults to 'matrices'
    """

    def __init__(
        self, project_storage, matrix_storage_class=None, matrix_directory=None
    ):
        self.project_storage = project_storage
        self.matrix_storage_class = matrix_storage_class or CSVMatrixStore
        self.directories = [matrix_directory or "matrices"]

    def get_store(self, matrix_uuid):
        """Return a storage object for a given matrix uuid.

        Args:
            matrix_uuid (string) A unique identifier within the project for a matrix.

        Returns: (MatrixStore) a reference to the matrix and its companion metadata
        """
        return self.matrix_storage_class(
            self.project_storage, self.directories, matrix_uuid
        )


class MatrixStore(object):
    """Base class for classes that allow access of a matrix and its metadata.

    Subclasses should be scoped to a storage format (e.g. CSV, HDF)
        and implement the _load, save, and head_of_matrix methods for that storage format

    Args:
        project_storage (triage.component.catwalk.storage.ProjectStorage)
            A project file storage engine
        directories (list): A list of subdirectories
        matrix_uuid (string): A unique identifier within the project for a matrix.
        matrix (pandas.DataFrame, optional): The raw matrix.
            Defaults to None, which means it will be loaded from storage on demand
        metadata (dict, optional). The matrix' metadata.
            Defaults to None, which means it will be loaded from storage on demand.
    """

    _labels = None

    def __init__(
        self, project_storage, directories, matrix_uuid, matrix=None, metadata=None
    ):
        self.matrix_uuid = matrix_uuid
        self.matrix_base_store = project_storage.get_store(
            directories, f"{matrix_uuid}.{self.suffix}"
        )
        self.metadata_base_store = project_storage.get_store(
            directories, f"{matrix_uuid}.yaml"
        )

        self.matrix = matrix
        self.metadata = metadata

    @property
    def matrix(self):
        """The raw matrix. Will load from storage into memory if not already loaded"""
        if self.__matrix is None:
            self.__matrix = self._load()
            # Is the index already in place?
            if self.__matrix.index.names != self.metadata['indices']:
                self.__matrix.set_index(self.metadata['indices'], inplace=True)

            self.__matrix = downcast_matrix(self.__matrix)
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix):
        self.__matrix = matrix

    @property
    def metadata(self):
        """The raw metadata. Will load from storage into memory if not already loaded"""
        if self.__metadata is None:
            self.__metadata = self.load_metadata()
        return self.__metadata

    @metadata.setter
    def metadata(self, metadata):
        self.__metadata = metadata

    @property
    def head_of_matrix(self):
        """The first line of the matrix"""
        return self.matrix.head(1)

    @property
    def exists(self):
        """Whether or not the matrix and metadata exist in storage"""
        return self.matrix_base_store.exists() and self.metadata_base_store.exists()

    @property
    def empty(self):
        """Whether or not the matrix has at least one row"""
        if not self.matrix_base_store.exists():
            return True
        else:
            head_of_matrix = self.head_of_matrix
            return head_of_matrix.empty

    def columns(self, include_label=False):
        """The matrix's column list"""
        head_of_matrix = self.head_of_matrix
        columns = head_of_matrix.columns.tolist()
        if include_label:
            return columns
        else:
            return [col for col in columns if col != self.metadata["label_name"]]

    def labels(self):
        """The matrix's label column."""
        if self._labels is not None:
            logging.debug("using stored labels")
            return self._labels
        else:
            logging.debug("popping labels from matrix")
            self._labels = self.matrix.pop(self.metadata["label_name"])
            return self._labels

    @property
    def uuid(self):
        """The matrix's unique id within the project"""
        return self.matrix_uuid

    @property
    def as_of_dates(self):
        """The list of as-of-dates in the matrix"""
        if "as_of_date" in self.matrix.index.names:
            return sorted(
                list(set([as_of_date for entity_id, as_of_date in self.matrix.index]))
            )
        else:
            return [self.metadata["end_time"]]

    @property
    def num_entities(self):
        """The number of entities in the matrix"""
        if self.matrix.index.names == ["entity_id"]:
            return len(self.matrix.index.values)
        elif "entity_id" in self.matrix.index.names:
            return len(
                self.matrix.index.levels[self.matrix.index.names.index("entity_id")]
            )

    @property
    def matrix_type(self):
        """The MatrixType (train or test). Returns an object with:
            a string name,
            evaluation ORM class
            prediction ORM class
            a boolean `is_test`
        """
        if self.metadata["matrix_type"] == "train":
            return TrainMatrixType
        elif self.metadata["matrix_type"] == "test":
            return TestMatrixType
        else:
            raise Exception(
                """matrix metadata for matrix {} must contain 'matrix_type'
             = "train" or "test" """.format(
                    self.uuid
                )
            )

    def matrix_with_sorted_columns(self, columns):
        """Return the matrix with columns sorted in the given column order

        Args:
            columns (list) The order of column names to return.
                Will error if this list does not contain the same elements as the matrix's columns
        """
        columnset = set(self.columns())
        desired_columnset = set(columns)
        if columnset == desired_columnset:
            if self.columns() != columns:
                logging.warning("Column orders not the same, re-ordering")
            return self.matrix[columns]
        else:
            if columnset.issuperset(desired_columnset):
                raise ValueError(
                    """
                    Columnset is superset of desired columnset. Extra items: %s
                """,
                    columnset - desired_columnset,
                )
            elif columnset.issubset(desired_columnset):
                raise ValueError(
                    """
                    Columnset is subset of desired columnset. Extra items: %s
                """,
                    desired_columnset - columnset,
                )
            else:
                raise ValueError(
                    """
                    Columnset and desired columnset mismatch. Unique items: %s
                """,
                    columnset ^ desired_columnset,
                )

    def load_metadata(self):
        """Load metadata from storage"""
        with self.metadata_base_store.open("rb") as fd:
            return yaml.load(fd)

    def save(self):
        raise NotImplementedError

    def __getstate__(self):
        """Remove object of a large size upon serialization.

        This helps in a multiprocessing context.
        """
        self.matrix = None
        self._labels = None
        self.metadata = None
        return self.__dict__.copy()


class HDFMatrixStore(MatrixStore):
    """Store and access matrices using HDF"""

    suffix = "h5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.matrix_base_store, S3Store):
            raise ValueError("HDFMatrixStore cannot be used with S3")

    @property
    def head_of_matrix(self):
        try:
            head_of_matrix = pd.read_hdf(self.matrix_base_store.path, start=0, stop=1)
            # Is the index already in place?
            if head_of_matrix.index.names != self.metadata["indices"]:
                head_of_matrix.set_index(self.metadata["indices"], inplace=True)
        except pd.errors.EmptyDataError:
            head_of_matrix = None

        return head_of_matrix

    def _load(self):
        return pd.read_hdf(self.matrix_base_store.path)

    def save(self):
        hdf = pd.HDFStore(
            self.matrix_base_store.path,
            mode="w",
            complevel=4,
            complib="zlib",
            format="table",
        )
        hdf.put(self.matrix_uuid, self.matrix.apply(pd.to_numeric), data_columns=True)
        hdf.close()

        with self.metadata_base_store.open("wb") as fd:
            yaml.dump(self.metadata, fd, encoding="utf-8")


class CSVMatrixStore(MatrixStore):
    """Store and access matrices using CSV"""

    suffix = "csv"

    @property
    def head_of_matrix(self):
        try:
            with self.matrix_base_store.open("rb") as fd:
                head_of_matrix = pd.read_csv(fd, nrows=1)
                head_of_matrix.set_index(self.metadata["indices"], inplace=True)
        except FileNotFoundError as fnfe:
            logging.exception(f"Matrix isn't there: {fnfe}")
            logging.exception("Returning Empty data frame")
            head_of_matrix = pd.DataFrame()

        return head_of_matrix

    def _load(self):
        parse_dates_argument = (
            ["as_of_date"] if "as_of_date" in self.metadata["indices"] else False
        )
        with self.matrix_base_store.open("rb") as fd:
            return pd.read_csv(fd, parse_dates=parse_dates_argument)

    def save(self):
        self.matrix_base_store.write(self.matrix.to_csv(None).encode("utf-8"))
        with self.metadata_base_store.open("wb") as fd:
            yaml.dump(self.metadata, fd, encoding="utf-8")


class TestMatrixType(object):
    string_name = "test"
    evaluation_obj = TestEvaluation
    prediction_obj = TestPrediction
    is_test = True


class TrainMatrixType(object):
    string_name = "train"
    evaluation_obj = TrainEvaluation
    prediction_obj = TrainPrediction
    is_test = False
