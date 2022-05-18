import datetime
import functools
import importlib
import os
import random
import tempfile
from contextlib import contextmanager
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import testing.postgresql
from descriptors import cachedproperty
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from triage.component.catwalk.db import ensure_db
from triage.component.catwalk.storage import MatrixStore, ProjectStorage
from triage.component.catwalk.utils import filename_friendly_hash
from triage.component.results_schema import Model, Matrix
from triage.experiments import CONFIG_VERSION
from triage.util.structs import FeatureNameList

from tests.results_tests.factories import init_engine, session, MatrixFactory

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa


CONFIG_QUERY_DATA = {
    "cohort": {
        "query": """
            select distinct(entity_id)
            from events
            where '{as_of_date}'::date >= outcome_date
        """,
        "filepath": "cohorts/file.sql",
    },
    "label": {
        "query": """
            select
                events.entity_id,
                bool_or(outcome::bool)::integer as outcome
            from events
            where '{as_of_date}'::date <= outcome_date
                and outcome_date < '{as_of_date}'::date + interval '{label_timespan}'
            group by entity_id
        """,
        "filepath": "labels/file.sql",
    },
}

MOCK_FILES = {
    os.path.join(
        os.path.abspath(os.getcwd()), f"{CONFIG_QUERY_DATA['label']['filepath']}"
    ): CONFIG_QUERY_DATA["label"]["query"],
    os.path.join(
        os.path.abspath(os.getcwd()), f"{CONFIG_QUERY_DATA['cohort']['filepath']}"
    ): CONFIG_QUERY_DATA["cohort"]["query"],
}


def open_side_effect(name):
    return mock.mock_open(read_data=MOCK_FILES[name]).return_value


def fake_labels(length):
    return np.array([random.choice([True, False]) for i in range(0, length)])


class MockTrainedModel:
    def predict_proba(self, dataset):
        return np.random.rand(len(dataset), len(dataset))


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
        init_as_of_dates=None,
    ):
        base_metadata = {
            "feature_start_time": datetime.date(2014, 1, 1),
            "end_time": datetime.date(2015, 1, 1),
            "as_of_date_frequency": "1y",
            "matrix_id": "some_matrix",
            "label_name": "label",
            "label_timespan": "3month",
            "indices": MatrixStore.indices,
            "matrix_type": matrix_type,
            "as_of_times": [datetime.date(2014, 10, 1), datetime.date(2014, 7, 1)],
        }
        metadata_overrides = metadata_overrides or {}
        base_metadata.update(metadata_overrides)
        if matrix is None:
            matrix = pd.DataFrame.from_dict(
                {
                    "entity_id": [1, 2],
                    "as_of_date": [pd.Timestamp(2014, 10, 1), pd.Timestamp(2014, 7, 1)],
                    "feature_one": [3, 4],
                    "feature_two": [5, 6],
                    "label": [7, 8],
                }
            ).set_index(MatrixStore.indices)
        if init_labels is None:
            init_labels = []
        labels = matrix.pop("label")
        self.matrix_label_tuple = matrix, labels
        self.metadata = base_metadata
        self.label_count = label_count
        self.init_labels = pd.Series(init_labels, dtype="float64")
        self.matrix_uuid = matrix_uuid
        self.init_as_of_dates = init_as_of_dates or []

        session = sessionmaker(db_engine)()
        session.add(Matrix(matrix_uuid=matrix_uuid))
        session.commit()

    @property
    def as_of_dates(self):
        """The list of as-of-dates in the matrix"""
        return self.init_as_of_dates or self.metadata["as_of_times"]

    @property
    def labels(self):
        if len(self.init_labels) > 0:
            return self.init_labels
        else:
            return fake_labels(self.label_count)


def fake_trained_model(
    db_engine, train_matrix_uuid="efgh", train_end_time=datetime.datetime(2016, 1, 1)
):
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
    db_model = Model(
        model_hash="abcd",
        train_matrix_uuid=train_matrix_uuid,
        train_end_time=train_end_time,
    )
    session.add(db_model)
    session.commit()
    model_id = db_model.model_id
    session.close()
    return trained_model, model_id


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
        "indices": MatrixStore.indices,
        "as_of_times": [datetime.date(2016, 12, 20)],
    }
    for override_key, override_value in override_kwargs.items():
        base_metadata[override_key] = override_value

    return base_metadata


def matrix_creator():
    """Return a sample matrix."""

    source_dict = {
        "entity_id": [1, 2],
        "as_of_date": [pd.Timestamp(2016, 1, 1), pd.Timestamp(2016, 1, 1)],
        "feature_one": [3, 4],
        "feature_two": [5, 6],
        "label": [0, 1],
    }
    return pd.DataFrame.from_dict(source_dict)


def get_matrix_store(project_storage, matrix=None, metadata=None, write_to_db=True):
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
    matrix["as_of_date"] = matrix["as_of_date"].apply(pd.Timestamp)
    matrix.set_index(MatrixStore.indices, inplace=True)
    matrix_store = project_storage.matrix_storage_engine().get_store(
        filename_friendly_hash(metadata)
    )
    matrix_store.metadata = metadata
    new_matrix = matrix.copy()
    labels = new_matrix.pop(matrix_store.label_column_name)
    matrix_store.matrix_label_tuple = new_matrix, labels
    matrix_store.save()
    matrix_store.clear_cache()
    if write_to_db:
        if (
            session.query(Matrix)
            .filter(Matrix.matrix_uuid == matrix_store.uuid)
            .count()
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

    zip_code_demographics = [
        ("60120", "hispanic", "2011-01-01"),
        ("60123", "white", "2011-01-01"),
    ]

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

    db_engine.execute(
        "create table zip_code_demographics (zip_code text, ethnicity text, as_of_date date)"
    )
    for demographic_row in zip_code_demographics:
        db_engine.execute(
            "insert into zip_code_demographics values (%s, %s, %s)", demographic_row
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


def sample_cohort_config(query_source="filepath"):
    return {
        "name": "has_past_events",
        query_source: CONFIG_QUERY_DATA["cohort"][query_source],
    }


def sample_config(query_source="filepath"):
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
        "subsets": [
            {
                "name": "evens",
                "query": """\
                    select distinct entity_id
                    from events
                    where entity_id %% 2 = 0
                    and outcome_date < '{as_of_date}'::date
                """,
            },
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
        },
        {
            "prefix": "zip_code_features",
            "from_obj": "entity_zip_codes join zip_code_events using (zip_code)",
            "knowledge_date_column": "as_of_date",
            "aggregates_imputation": {"all": {"type": "constant", "value": 0}},
            "aggregates": [{"quantity": "num_events", "metrics": ["max", "min"]}],
            "intervals": ["1year"],
        },
    ]

    cohort_config = sample_cohort_config(query_source)

    label_config = {
        query_source: CONFIG_QUERY_DATA["label"][query_source],
        "name": "custom_label_name",
        "include_missing_labels_in_train_as": False,
    }

    bias_audit_config = {
        "from_obj_query": "select * from zip_code_demographics join entity_zip_codes using (zip_code)",
        "attribute_columns": ["ethnicity"],
        "knowledge_date_column": "as_of_date",
        "entity_id_column": "entity_id",
        "ref_groups_method": "predefined",
        "ref_groups": {"ethnicity": "white"},
        "thresholds": {"percentiles": [], "top_n": [2]},
    }

    return {
        "config_version": CONFIG_VERSION,
        "random_seed": 1234,
        "label_config": label_config,
        "entity_column_name": "entity_id",
        "model_comment": "test2-final-final",
        "model_group_keys": [
            "label_name",
            "label_type",
            "custom_key",
            "class_path",
            "parameters",
        ],
        "feature_aggregations": feature_config,
        "cohort_config": cohort_config,
        "temporal_config": temporal_config,
        "grid_config": grid_config,
        "bias_audit_config": bias_audit_config,
        "prediction": {"rank_tiebreaker": "random"},
        "scoring": scoring_config,
        "user_metadata": {"custom_key": "custom_value"},
        "individual_importance": {"n_ranks": 2},
    }


@contextmanager
def assert_plot_figures_added():
    num_figures_before = plt.gcf().number
    yield
    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after


class CallSpy:
    """Callable-wrapper and -patcher to record invocations.

    ``CallSpy``, (unlike ``Mock``), makes it easy to wrap callables for
    the express purpose of recording how they're invoked – without
    modifying functionality. And ``CallSpy``, (unlike ``Mock``),
    reproduces the descriptor interface, such that methods can be
    patched and proxied for this purpose, as easily as functions.

    For example, as a context manager::

        with CallSpy('my_module.MyClass.my_method') as spy:
            ...

        assert (('arg0',), {'param0': 0}) in spy.calls

    """

    def __init__(self, signature):
        self.calls = []
        self.signature = signature

    @cachedproperty
    def target_path(self):
        return self.signature.split(".")

    @cachedproperty
    def target_name(self):
        return self.target_path[-1]

    @cachedproperty
    def target_base(self):
        # walk target path until can no longer import it as a module path
        for index in range(len(self.target_path)):
            path_parts = self.target_path[: (index + 1)]
            import_path = ".".join(path_parts)

            try:
                base = importlib.import_module(import_path)
            except ImportError:
                # we've imported all that we can import
                # walk the remainder by attribute access
                remainder = self.target_path[index:-1]
                for part in remainder:
                    base = getattr(base, part)

                return base

        raise ValueError(f"cannot patch signature {self.signature!r}")

    @cachedproperty
    def target_object(self):
        return getattr(self.target_base, self.target_name)

    @cachedproperty
    def patch(self):
        return mock.patch.object(self.target_base, self.target_name, new=self)

    def start(self):
        if not callable(self.target_object):
            # 1. ensure target_object set before patching
            # 2. check that it's sane (needn't be done here but reasonable)
            raise TypeError(f"signature target not callable {self.target_object!r}")

        self.patch.start()

    def stop(self):
        self.patch.stop()

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.target_object(*args, **kwargs)

    def __get__(self, instance, cls=None):
        if instance is None:
            return self

        return functools.partial(self, instance)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
