# coding: utf-8

import itertools

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import os
import pathlib
from contextlib import contextmanager
from os.path import dirname
from urllib.parse import urlparse

import gzip
import pandas as pd
import s3fs
import wrapt
import yaml
import joblib

from triage.component.results_schema import (
    TestEvaluation,
    TrainEvaluation,
    TestPrediction,
    TrainPrediction,
    TestPredictionMetadata,
    TrainPredictionMetadata,
    TestAequitas,
    TrainAequitas
)
from triage.util.pandas import downcast_matrix


class Store:
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
        path_head, *path_parts: one or more path components,
            (to be joined by PurePosixPath to create the final path).

        **config: arguments to be passed to the S3Fs client constructor.

    """
    class S3FileWrapper(wrapt.ObjectProxy):

        # don't allow wrapped object to take wrapper's place
        # upon __enter__
        def __enter__(self):
            return self

        def write(self, data, block_size=(5 * 2 ** 20)):
            out = 0

            for offset in itertools.count(0, block_size):
                chunk = data[offset:(offset + block_size)]

                if not chunk:
                    return out

                out += self.__wrapped__.write(chunk)

    def __init__(self, path_head, *path_parts, **config):
        self.path = str(
            pathlib.PurePosixPath(path_head.replace('s3://', ''),
                                  *path_parts)
        )
        self.config = config

    @property
    def client(self):
        return s3fs.S3FileSystem(**self.config)

    def exists(self):
        return self.client.exists(self.path)

    def delete(self):
        self.client.rm(self.path)

    def open(self, *args, **kwargs):
        # NOTE: remove S3FileWrapper as soon as s3fs properly
        # NOTE: chunks out too-large writes
        # NOTE: see also: tests.catwalk_tests.test_storage.test_S3Store_large
        s3file = self.client.open(self.path, *args, **kwargs)
        return self.S3FileWrapper(s3file)


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


class ProjectStorage:
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


class ModelStorageEngine:
    """Store arbitrary models in a given project storage using joblib

    Args:
        project_storage (triage.component.catwalk.storage.ProjectStorage)
            A project file storage engine
        model_directory (string, optional) A directory name for models.
            Defaults to 'trained_models'
    """
    def __init__(self, project_storage, model_directory=None):
        self.project_storage = project_storage
        self.directories = [model_directory or "trained_models"]
        self.should_cache = False
        self.reset_cache()

    def reset_cache(self):
        self.cache = {}

    @contextmanager
    def cache_models(self):
        """Caches each model in memory as it is written.

        Must be used as a context manager.
        The cache is cleared when the context manager goes out of scope
        """
        self.should_cache = True
        try:
            yield
        finally:
            self.reset_cache()
            self.should_cache = False

    def write(self, obj, model_hash):
        """Persist a model object using joblib. Also performs compression

        Args:
            obj  (object) A picklable model object
            model_hash (string) An identifier, unique within this project, for the model
        """
        if self.should_cache:
            logger.spam(f"Caching model {model_hash}")
            self.cache[model_hash] = obj
        with self._get_store(model_hash).open("wb") as fd:
            joblib.dump(obj, fd, compress=True)

    def load(self, model_hash):
        """Load a model object using joblib

        Args:
            model_hash (string) An identifier, unique within this project, for the model

        Returns: (object) A model object
        """
        if self.should_cache and model_hash in self.cache:
            logger.spam(f"Returning model {model_hash} from cache")
            return self.cache[model_hash]
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


class MatrixStorageEngine:
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


class MatrixStore:
    """Base class for classes that allow access of a matrix and its metadata.

    Subclasses should be scoped to a storage format (e.g. CSV)
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
    _matrix_label_tuple = None
    indices = ['entity_id', 'as_of_date']

    def __init__(
        self, project_storage, directories, matrix_uuid, matrix=None, metadata=None
    ):
        self.should_cache = False
        self.matrix_uuid = matrix_uuid
        self.matrix_base_store = project_storage.get_store(
            directories, f"{matrix_uuid}.{self.suffix}"
        )
        self.metadata_base_store = project_storage.get_store(
            directories, f"{matrix_uuid}.yaml"
        )

        self.metadata = metadata
        if matrix is not None:
            self._matrix_label_tuple = self._preprocess_and_split_matrix(matrix)

    @contextmanager
    def cache(self):
        """Enable caching

        Must be used as a context manager.
        The cache is cleared when the context manager goes out of scope
        """
        self.should_cache = True
        try:
            yield
        finally:
            self.clear_cache()
            self.should_cache = False

    def _preprocess_and_split_matrix(self, matrix_with_labels):
        """Perform desired preprocessing that we generally want to do after loading a matrix

        This includes setting the index (depending on the storage, may not be serializable)
        and downcasting.
        """
        if matrix_with_labels.index.names != self.indices:
            matrix_with_labels.set_index(self.indices, inplace=True)
        index_of_date = matrix_with_labels.index.names.index('as_of_date')
        if matrix_with_labels.index.levels[index_of_date].dtype != "datetime64[ns]":
            raise ValueError(f"Woah is {matrix_with_labels.index.levels[index_of_date].dtype}")
        matrix_with_labels = downcast_matrix(matrix_with_labels)
        labels = matrix_with_labels.pop(self.label_column_name)
        design_matrix = matrix_with_labels
        return design_matrix, labels

    @property
    def matrix_label_tuple(self):
        if self._matrix_label_tuple:
            return self._matrix_label_tuple
        design_matrix, labels = self._preprocess_and_split_matrix(self._load())
        if self.should_cache:
            self._matrix_label_tuple = design_matrix, labels
        return design_matrix, labels

    @matrix_label_tuple.setter
    def matrix_label_tuple(self, matrix_label_tuple):
        self._matrix_label_tuple = matrix_label_tuple

    @property
    def design_matrix(self):
        """The matrix without the label vector, only the index and features"""
        return self.matrix_label_tuple[0]

    @property
    def labels(self):
        if type(self.matrix_label_tuple[1]) != pd.Series:
            raise TypeError("Label stored as something other than pandas Series")
        return self.matrix_label_tuple[1]

    @property
    def metadata(self):
        """The raw metadata. Will load from storage into memory if not already loaded"""
        if self.__metadata is not None:
            return self.__metadata
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

    @property
    def label_column_name(self):
        return self.metadata["label_name"]

    @property
    def index(self):
        if self.metadata['indices'] != self.indices:
            raise ValueError(f"Indices must be {self.indices}")
        return self.design_matrix.index

    @property
    def uuid(self):
        """The matrix's unique id within the project"""
        return self.matrix_uuid

    @property
    def as_of_dates(self):
        """All as-of-dates in the matrix. Will be converted to datetime.date"""
        return sorted(set(
            as_of_date.date() if hasattr(as_of_date, 'date') else as_of_date
            for entity_id, as_of_date in self.design_matrix.index
        ))

    @property
    def num_entities(self):
        """The number of entities in the matrix"""
        return len(
            self.design_matrix.index.levels[self.design_matrix.index.names.index("entity_id")]
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
                logger.debug("Column orders not the same, re-ordering")
            return self.design_matrix[columns]
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

    @property
    def full_matrix_for_saving(self):
        return self.design_matrix.assign(**{self.label_column_name: self.labels})

    def load_metadata(self):
        """Load metadata from storage"""
        with self.metadata_base_store.open("rb") as fd:
            return yaml.full_load(fd)

    def save(self):
        raise NotImplementedError

    def clear_cache(self):
        self._matrix_label_tuple = None

    def __getstate__(self):
        """Remove object of a large size upon serialization.

        This helps in a multiprocessing context.
        """
        state = self.__dict__.copy()
        state['_matrix_label_tuple'] = None
        return state


class CSVMatrixStore(MatrixStore):
    """Store and access compressed matrices using CSV"""

    suffix = "csv.gz"

    @property
    def head_of_matrix(self):
        try:
            with self.matrix_base_store.open("rb") as fd:
                head_of_matrix = pd.read_csv(fd, compression="gzip", nrows=1)
                head_of_matrix.set_index(self.indices, inplace=True)
        except FileNotFoundError as fnfe:
            logger.exception(f"Matrix {self.uuid} not found Returning Empty data frame")
            head_of_matrix = pd.DataFrame()

        return head_of_matrix

    def _load(self):
        with self.matrix_base_store.open("rb") as fd:
            return pd.read_csv(fd, compression="gzip", parse_dates=["as_of_date"])

    def save(self):
        self.matrix_base_store.write(gzip.compress(self.full_matrix_for_saving.to_csv(None).encode("utf-8")))
        with self.metadata_base_store.open("wb") as fd:
            yaml.dump(self.metadata, fd, encoding="utf-8")


class TestMatrixType:
    string_name = "test"
    evaluation_obj = TestEvaluation
    prediction_obj = TestPrediction
    aequitas_obj = TestAequitas
    prediction_metadata_obj = TestPredictionMetadata
    is_test = True


class TrainMatrixType:
    string_name = "train"
    evaluation_obj = TrainEvaluation
    prediction_obj = TrainPrediction
    aequitas_obj = TrainAequitas
    prediction_metadata_obj = TrainPredictionMetadata
    is_test = False
