from contextlib import contextmanager
import pandas
import tempfile
import yaml
import numpy
import random
from results_schema import Model
from sqlalchemy.orm import sessionmaker
import datetime
import os
from catwalk.storage import CSVMatrixStore, HDFMatrixStore
from metta import metta_io as metta
from collections import OrderedDict


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


def fake_labels(length):
    return numpy.array([random.choice([True, False]) for i in range(0, length)])


class MockTrainedModel(object):
    def predict_proba(self, dataset):
        return numpy.random.rand(len(dataset), len(dataset))


def fake_trained_model(project_path, model_storage_engine, db_engine, train_matrix_uuid='efgh'):
    """Creates and stores a trivial trained model

    Args:
        project_path (string) a desired fs/s3 project path
        model_storage_engine (catwalk.storage.ModelStorageEngine)
        db_engine (sqlalchemy.engine)

    Returns:
        (int) model id for database retrieval
    """
    trained_model = MockTrainedModel()
    model_storage_engine.get_store('abcd').write(trained_model)
    session = sessionmaker(db_engine)()
    db_model = Model(model_hash='abcd', train_matrix_uuid=train_matrix_uuid)
    session.add(db_model)
    session.commit()
    return trained_model, db_model.model_id


def sample_metta_csv_diff_order(directory):
    """Stores matrix and metadata in a metta-data-like form

    The train and test matrices will have different column orders

    Args:
        directory (str)
    """
    train_dict = OrderedDict([
        ('entity_id', [1, 2]),
        ('k_feature', [0.5, 0.4]),
        ('m_feature', [0.4, 0.5]),
        ('label', [0, 1])
    ])
    train_matrix = pandas.DataFrame.from_dict(train_dict)
    train_metadata = {
        'beginning_of_time': datetime.date(2014, 1, 1),
        'end_time': datetime.date(2015, 1, 1),
        'matrix_id': 'train_matrix',
        'label_name': 'label',
        'label_window': '3month',
        'indices': ['entity_id'],
    }

    test_dict = OrderedDict([
        ('entity_id', [3, 4]),
        ('m_feature', [0.4, 0.5]),
        ('k_feature', [0.5, 0.4]),
        ('label', [0, 1])
    ])

    test_matrix = pandas.DataFrame.from_dict(test_dict)
    test_metadata = {
        'beginning_of_time': datetime.date(2015, 1, 1),
        'end_time': datetime.date(2016, 1, 1),
        'matrix_id': 'test_matrix',
        'label_name': 'label',
        'label_window': '3month',
        'indices': ['entity_id'],
    }

    train_uuid, test_uuid = metta.archive_train_test(
        train_config=train_metadata,
        df_train=train_matrix,
        test_config=test_metadata,
        df_test=test_matrix,
        directory=directory,
        format='csv'
    )

    train_store = CSVMatrixStore(
        matrix_path=os.path.join(directory, '{}.csv'.format(train_uuid)),
        metadata_path=os.path.join(directory, '{}.yaml'.format(train_uuid))
    )
    test_store = CSVMatrixStore(
        matrix_path=os.path.join(directory, '{}.csv'.format(test_uuid)),
        metadata_path=os.path.join(directory, '{}.yaml'.format(test_uuid))
    )
    return train_store, test_store


def sample_metta_hdf_diff_order(directory):
    """Stores matrix and metadata in a metta-data-like form

    The train and test matrices will have different column orders

    Args:
        directory (str)
    """
    train_dict = OrderedDict([
        ('entity_id', [1, 2]),
        ('k_feature', [0.5, 0.4]),
        ('m_feature', [0.4, 0.5]),
        ('label', [0, 1])
    ])
    train_matrix = pandas.DataFrame.from_dict(train_dict)
    train_metadata = {
        'beginning_of_time': datetime.date(2014, 1, 1),
        'end_time': datetime.date(2015, 1, 1),
        'matrix_id': 'train_matrix',
        'label_name': 'label',
        'label_window': '3month',
        'indices': ['entity_id'],
    }

    test_dict = OrderedDict([
        ('entity_id', [3, 4]),
        ('m_feature', [0.4, 0.5]),
        ('k_feature', [0.5, 0.4]),
        ('label', [0, 1])
    ])

    test_matrix = pandas.DataFrame.from_dict(test_dict)
    test_metadata = {
        'beginning_of_time': datetime.date(2015, 1, 1),
        'end_time': datetime.date(2016, 1, 1),
        'matrix_id': 'test_matrix',
        'label_name': 'label',
        'label_window': '3month',
        'indices': ['entity_id'],
    }

    train_uuid, test_uuid = metta.archive_train_test(
        train_config=train_metadata,
        df_train=train_matrix,
        test_config=test_metadata,
        df_test=test_matrix,
        directory=directory,
        format='hdf'
    )

    train_store = HDFMatrixStore(
        matrix_path=os.path.join(directory, '{}.h5'.format(train_uuid)),
        metadata_path=os.path.join(directory, '{}.yaml'.format(train_uuid))
    )
    test_store = HDFMatrixStore(
        matrix_path=os.path.join(directory, '{}.h5'.format(test_uuid)),
        metadata_path=os.path.join(directory, '{}.yaml'.format(test_uuid))
    )
    return train_store, test_store
