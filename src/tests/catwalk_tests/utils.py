import datetime
import random
import tempfile
from contextlib import contextmanager
import pytest

import numpy
import pandas
import yaml

from triage.component.catwalk.storage import (
    ProjectStorage,
)
from triage.util.structs import FeatureNameList


@contextmanager
def fake_metta(matrix_dict, metadata):
    """Stores matrix and metadata in a metta-data-like form

    Args:
    matrix_dict (dict) of form { columns: values }.
        Expects an entity_id to be present which it will use as the index
    metadata (dict). Any metadata that should be set

    Yields:
        tuple of filenames for matrix and metadata
    """
    matrix = pandas.DataFrame.from_dict(matrix_dict).set_index("entity_id")
    with tempfile.NamedTemporaryFile() as matrix_file:
        with tempfile.NamedTemporaryFile("w") as metadata_file:
            hdf = pandas.HDFStore(matrix_file.name)
            hdf.put("title", matrix, data_columns=True)
            matrix_file.seek(0)

            yaml.dump(metadata, metadata_file)
            metadata_file.seek(0)
            yield (matrix_file.name, metadata_file.name)


def fake_labels(length):
    return numpy.array([random.choice([True, False]) for i in range(0, length)])


@pytest.fixture
def sample_metadata():
    return {
        "feature_start_time": datetime.date(2012, 12, 20),
        "end_time": datetime.date(2016, 12, 20),
        "label_name": "label",
        "as_of_date_frequency": "1w",
        "max_training_history": "5y",
        "state": "default",
        "cohort_name": "default",
        "label_timespan": "1y",
        "metta-uuid": "1234",
        "feature_names": FeatureNameList(["ft1", "ft2"]),
        "feature_groups": ["all: True"],
        "indices": ["entity_id"],
    }


@pytest.fixture
def sample_df():
    return pandas.DataFrame.from_dict(
        {
            "entity_id": [1, 2],
            "feature_one": [3, 4],
            "feature_two": [5, 6],
            "label": ["good", "bad"],
        }
    ).set_index("entity_id")


@pytest.fixture
def sample_matrix_store():
    with tempfile.TemporaryDirectory() as tempdir:
        project_storage = ProjectStorage(tempdir)
        store = project_storage.matrix_storage_engine().get_store("1234")
        store.matrix = sample_df()
        store.metadata = sample_metadata()
        return store
