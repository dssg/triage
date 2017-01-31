from contextlib import contextmanager
import pandas
import tempfile
import yaml
import numpy
import random
from triage.db import ensure_db, Model, metadata
from sqlalchemy.orm import sessionmaker, clear_mappers
import testing.postgresql
from sqlalchemy import create_engine
from functools import wraps
import logging


@contextmanager
def fake_db():
    metadata.clear()
    clear_mappers()
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        yield engine


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
    matrix = pandas.DataFrame.from_dict(matrix_dict).set_index('entity_id')
    with tempfile.NamedTemporaryFile() as matrix_file:
        with tempfile.NamedTemporaryFile('w') as metadata_file:
            hdf = pandas.HDFStore(matrix_file.name)
            hdf.put('title', matrix, data_columns=True)
            matrix_file.seek(0)

            yaml.dump(metadata, metadata_file)
            metadata_file.seek(0)
            yield (matrix_file.name, metadata_file.name)


class MockTrainedModel(object):
    def predict(self, dataset):
        return numpy.array([random.random() for i in range(0, len(dataset))])


def fake_trained_model(project_path, storage_engine, db_engine):
    """Creates and stores a trivial trained model

    Args:
        project_path (string) a desired fs/s3 project path
        storage_engine (triage.storage.StorageEngine)
        db_engine (sqlalchemy.engine)

    Returns:
        (int) model id for database retrieval
    """
    trained_model = MockTrainedModel()
    storage_engine.get_store('abcd').write(trained_model)
    session = sessionmaker(db_engine)()
    db_model = Model(model_hash='abcd')
    session.add(db_model)
    session.commit()
    return db_model.model_id
