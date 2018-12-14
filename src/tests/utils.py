import datetime
import random
import tempfile
from contextlib import contextmanager

import numpy
import pandas
import yaml
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import testing.postgresql
from triage.component.catwalk.db import ensure_db
from triage.component.catwalk.storage import MatrixStore, ProjectStorage
from triage.component.results_schema import Model, Matrix
from triage.experiments import CONFIG_VERSION
from tests.results_tests.factories import init_engine, session, MatrixFactory
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


class MockTrainedModel(object):
    def predict_proba(self, dataset):
        return numpy.random.rand(len(dataset), len(dataset))


class MockMatrixStore(MatrixStore):
    def __init__(
        self,
        matrix_type,
        matrix_uuid,
        label_count,
        db_engine,
        init_labels=None,
        metadata_overrides=None,
        matrix=None,
    ):
        base_metadata = {
            "feature_start_time": datetime.date(2014, 1, 1),
            "end_time": datetime.date(2015, 1, 1),
            "as_of_date_frequency": "1y",
            "matrix_id": "some_matrix",
            "label_name": "label",
            "label_timespan": "3month",
            "indices": ["entity_id"],
            "matrix_type": matrix_type,
        }
        metadata_overrides = metadata_overrides or {}
        base_metadata.update(metadata_overrides)
        if matrix is None:
            matrix = pandas.DataFrame.from_dict(
                {
                    "entity_id": [1, 2],
                    "feature_one": [3, 4],
                    "feature_two": [5, 6],
                    "label": [7, 8],
                }
            ).set_index("entity_id")
        if init_labels is None:
            init_labels = []
        self.matrix = matrix
        self.metadata = base_metadata
        self.label_count = label_count
        self.init_labels = init_labels
        self.matrix_uuid = matrix_uuid

        session = sessionmaker(db_engine)()
        session.add(Matrix(matrix_uuid=matrix_uuid))

    def labels(self):
        if self.init_labels == []:
            return fake_labels(self.label_count)
        else:
            return self.init_labels


def fake_trained_model(db_engine, train_matrix_uuid="efgh"):
    """Creates and stores a trivial trained model and training matrix

    Args:
        db_engine (sqlalchemy.engine)

    Returns:
        (int) model id for database retrieval
    """
    session = sessionmaker(db_engine)()
    session.merge(Matrix(matrix_uuid=train_matrix_uuid))

    # Create the fake trained model and store in db
    trained_model = MockTrainedModel()
    db_model = Model(model_hash="abcd", train_matrix_uuid=train_matrix_uuid)
    session.add(db_model)
    session.commit()
    return trained_model, db_model.model_id


def matrix_metadata_creator(**override_kwargs):
    """Create a sample valid matrix metadata with optional overrides

    Args:
    **override_kwargs: Keys and values to override in the metadata

    Returns: (dict)
    """
    base_metadata = {
        "feature_start_time": datetime.date(2012, 12, 20),
        "end_time": datetime.date(2016, 12, 20),
        "label_name": "label",
        "as_of_date_frequency": "1w",
        "max_training_history": "5y",
        "matrix_id": "tester-1",
        "state": "active",
        "cohort_name": "default",
        "label_timespan": "1y",
        "metta-uuid": "1234",
        "matrix_type": "test",
        "feature_names": FeatureNameList(["ft1", "ft2"]),
        "feature_groups": ["all: True"],
        "indices": ["entity_id", "as_of_date"],
    }
    for override_key, override_value in override_kwargs.items():
        base_metadata[override_key] = override_value

    return base_metadata


def matrix_creator(index=None):
    """Return a sample matrix.

    Args:
        index (list, optional): The matrix index column names. Defaults to ['entity_id', 'date']
    """
    if not index:
        index = ["entity_id", "as_of_date"]
    source_dict = {
        "entity_id": [1, 2],
        "feature_one": [3, 4],
        "feature_two": [5, 6],
        "label": [0, 1],
    }
    if "as_of_date" in index:
        source_dict["as_of_date"] = [
            datetime.datetime(2016, 1, 1),
            datetime.datetime(2016, 1, 1),
        ]
    return pandas.DataFrame.from_dict(source_dict).set_index(index)


def get_matrix_store(project_storage, matrix=None, metadata=None):
    """Return a matrix store associated with the given project storage.
    Also adds an entry in the matrices table if it doesn't exist already

    Args:
        project_storage (triage.component.catwalk.storage.ProjectStorage) A project's storage
        matrix (dataframe, optional): A matrix to store. Defaults to the output of matrix_creator()
        metadata (dict, optional): matrix metadata.
            defaults to the output of matrix_metadata_creator()
    """
    if matrix is None:
        matrix = matrix_creator()
    if not metadata:
        metadata = matrix_metadata_creator()
    matrix_store = project_storage.matrix_storage_engine().get_store(
        metadata["metta-uuid"]
    )
    matrix_store.matrix = matrix
    matrix_store.metadata = metadata
    matrix_store.save()
    if (
        session.query(Matrix).filter(Matrix.matrix_uuid == matrix_store.uuid).count()
        == 0
    ):
        MatrixFactory(matrix_uuid=matrix_store.uuid)
        session.commit()
    return matrix_store


@contextmanager
def rig_engines():
    """Set up a db engine and project storage engine

    Yields (tuple) (database engine, project storage engine)
    """
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        init_engine(db_engine)
        with tempfile.TemporaryDirectory() as temp_dir:
            project_storage = ProjectStorage(temp_dir)
            yield db_engine, project_storage


def populate_source_data(db_engine):
    complaints = [
        (1, "2010-10-01", 5),
        (1, "2011-10-01", 4),
        (1, "2011-11-01", 4),
        (1, "2011-12-01", 4),
        (1, "2012-02-01", 5),
        (1, "2012-10-01", 4),
        (1, "2013-10-01", 5),
        (2, "2010-10-01", 5),
        (2, "2011-10-01", 5),
        (2, "2011-11-01", 4),
        (2, "2011-12-01", 4),
        (2, "2012-02-01", 6),
        (2, "2012-10-01", 5),
        (2, "2013-10-01", 6),
        (3, "2010-10-01", 5),
        (3, "2011-10-01", 3),
        (3, "2011-11-01", 4),
        (3, "2011-12-01", 4),
        (3, "2012-02-01", 4),
        (3, "2012-10-01", 3),
        (3, "2013-10-01", 4),
    ]

    entity_zip_codes = [(1, "60120"), (2, "60123"), (3, "60123")]

    zip_code_events = [("60120", "2012-10-01", 1), ("60123", "2012-10-01", 10)]

    events = [
        (1, 1, "2011-01-01"),
        (1, 1, "2011-06-01"),
        (1, 1, "2011-09-01"),
        (1, 1, "2012-01-01"),
        (1, 1, "2012-01-10"),
        (1, 1, "2012-06-01"),
        (1, 1, "2013-01-01"),
        (1, 0, "2014-01-01"),
        (1, 1, "2015-01-01"),
        (2, 1, "2011-01-01"),
        (2, 1, "2011-06-01"),
        (2, 1, "2011-09-01"),
        (2, 1, "2012-01-01"),
        (2, 1, "2013-01-01"),
        (2, 1, "2014-01-01"),
        (2, 1, "2015-01-01"),
        (3, 0, "2011-01-01"),
        (3, 0, "2011-06-01"),
        (3, 0, "2011-09-01"),
        (3, 0, "2012-01-01"),
        (3, 0, "2013-01-01"),
        (3, 1, "2014-01-01"),
        (3, 0, "2015-01-01"),
    ]

    db_engine.execute(
        """create table cat_complaints (
        entity_id int,
        as_of_date date,
        cat_sightings int
        )"""
    )

    db_engine.execute(
        """create table entity_zip_codes (
        entity_id int,
        zip_code text
        )"""
    )

    for entity_zip_code in entity_zip_codes:
        db_engine.execute(
            "insert into entity_zip_codes values (%s, %s)", entity_zip_code
        )

    db_engine.execute(
        """create table zip_code_events (
        zip_code text,
        as_of_date date,
        num_events int
    )"""
    )
    for zip_code_event in zip_code_events:
        db_engine.execute(
            "insert into zip_code_events values (%s, %s, %s)", zip_code_event
        )

    for complaint in complaints:
        db_engine.execute("insert into cat_complaints values (%s, %s, %s)", complaint)

    db_engine.execute(
        """create table events (
        entity_id int,
        outcome int,
        outcome_date date
    )"""
    )

    for event in events:
        db_engine.execute("insert into events values (%s, %s, %s)", event)


def sample_config():
    temporal_config = {
        "feature_start_time": "2010-01-01",
        "feature_end_time": "2014-01-01",
        "label_start_time": "2011-01-01",
        "label_end_time": "2014-01-01",
        "model_update_frequency": "1year",
        "training_label_timespans": ["6months"],
        "test_label_timespans": ["6months"],
        "training_as_of_date_frequencies": "1day",
        "test_as_of_date_frequencies": "3months",
        "max_training_histories": ["6months"],
        "test_durations": ["1months"],
    }

    scoring_config = {
        "testing_metric_groups": [
            {"metrics": ["precision@"], "thresholds": {"top_n": [2]}}
        ],
        "training_metric_groups": [
            {"metrics": ["precision@"], "thresholds": {"top_n": [3]}}
        ],
    }

    grid_config = {
        "sklearn.tree.DecisionTreeClassifier": {
            "min_samples_split": [10, 100],
            "max_depth": [3, 5],
            "criterion": ["gini"],
        }
    }

    feature_config = [
        {
            "prefix": "entity_features",
            "from_obj": "cat_complaints",
            "knowledge_date_column": "as_of_date",
            "aggregates_imputation": {"all": {"type": "constant", "value": 0}},
            "aggregates": [{"quantity": "cat_sightings", "metrics": ["count", "avg"]}],
            "intervals": ["1year"],
            "groups": ["entity_id"],
        },
        {
            "prefix": "zip_code_features",
            "from_obj": "entity_zip_codes join zip_code_events using (zip_code)",
            "knowledge_date_column": "as_of_date",
            "aggregates_imputation": {"all": {"type": "constant", "value": 0}},
            "aggregates": [{"quantity": "num_events", "metrics": ["max", "min"]}],
            "intervals": ["1year"],
            "groups": ["entity_id", "zip_code"],
        },
    ]

    cohort_config = {
        "query": "select distinct(entity_id) from events where '{as_of_date}'::date < outcome_date",
        "name": "has_past_events",
    }

    label_config = {
        "query": """
            select
            events.entity_id,
            bool_or(outcome::bool)::integer as outcome
            from events
            where '{as_of_date}'::date <= outcome_date
                and outcome_date < '{as_of_date}'::date + interval '{label_timespan}'
                group by entity_id
        """,
        "name": "custom_label_name",
        "include_missing_labels_in_train_as": False,
    }

    return {
        "config_version": CONFIG_VERSION,
        "label_config": label_config,
        "entity_column_name": "entity_id",
        "model_comment": "test2-final-final",
        "model_group_keys": ["label_name", "label_type", "custom_key"],
        "feature_aggregations": feature_config,
        "cohort_config": cohort_config,
        "temporal_config": temporal_config,
        "grid_config": grid_config,
        "scoring": scoring_config,
        "user_metadata": {"custom_key": "custom_value"},
        "individual_importance": {"n_ranks": 2},
    }
