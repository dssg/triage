import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import datetime
import shutil
import sys
import random
from contextlib import contextmanager
import functools
import operator
import tempfile

import sqlalchemy

import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker

from triage.component.results_schema import Model
from triage.util.structs import FeatureNameList




def str_in_sql(values):
    return ",".join(map(lambda x: "'{}'".format(x), values))


def feature_list(feature_dictionary):
    """Convert a feature dictionary to a sorted list

    Args: feature_dictionary (dict)

    Returns: sorted list of feature names
    """
    if not feature_dictionary:
        return FeatureNameList()
    return FeatureNameList(sorted(
        functools.reduce(
            operator.concat,
            (feature_dictionary[key] for key in feature_dictionary.keys()),
        )
    ))


def convert_string_column_to_date(column):
    return [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in column]


def create_features_table(table_number, table, engine):
    engine.execute(
        """
            create table features.features{} (
                entity_id int, as_of_date date, f{} int, f{} int
            )
        """.format(
            table_number, (table_number * 2) + 1, (table_number * 2) + 2
        )
    )
    for row in table:
        engine.execute(
            """
                insert into features.features{} values (%s, %s, %s, %s)
            """.format(
                table_number
            ),
            row,
        )


def create_entity_date_df(
    labels,
    states,
    as_of_dates,
    state_one,
    state_two,
    label_name,
    label_type,
    label_timespan,
):
    """ This function makes a pandas DataFrame that mimics the entity-date table
    for testing against.
    """
    0, "2016-02-01", "1 month", "booking", "binary", 0
    labels_table = pd.DataFrame(
        labels,
        columns=[
            "entity_id",
            "as_of_date",
            "label_timespan",
            "label_name",
            "label_type",
            "label",
        ],
    )
    states_table = pd.DataFrame(
        states, columns=["entity_id", "as_of_date", "state_one", "state_two"]
    ).set_index(["entity_id", "as_of_date"])
    as_of_dates = [date.date() for date in as_of_dates]
    labels_table = labels_table[labels_table["label_name"] == label_name]
    labels_table = labels_table[labels_table["label_type"] == label_type]
    labels_table = labels_table[labels_table["label_timespan"] == label_timespan]
    labels_table = labels_table.join(other=states_table, on=("entity_id", "as_of_date"))
    labels_table = labels_table[labels_table["state_one"] & labels_table["state_two"]]
    ids_dates = labels_table[["entity_id", "as_of_date"]]
    ids_dates = ids_dates.sort_values(["entity_id", "as_of_date"])
    ids_dates["as_of_date"] = [
        datetime.datetime.strptime(date, "%Y-%m-%d").date()
        for date in ids_dates["as_of_date"]
    ]
    ids_dates = ids_dates[ids_dates["as_of_date"].isin(as_of_dates)]
    logger.spam(ids_dates)

    return ids_dates.reset_index(drop=True)


def NamedTempFile():
    if sys.version_info >= (3, 0, 0):
        return tempfile.NamedTemporaryFile(mode="w+", newline="")
    else:
        return tempfile.NamedTemporaryFile()


@contextmanager
def TemporaryDirectory():
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)


def fake_labels(length):
    return np.array([random.choice([True, False]) for i in range(0, length)])


class MockTrainedModel:
    def predict_proba(self, dataset):
        return np.random.rand(len(dataset), len(dataset))


def fake_trained_model(project_path, model_storage_engine, db_engine):
    """Creates and stores a trivial trained model

    Args:
        project_path (string) a desired fs/s3 project path
        model_storage_engine (triage.storage.ModelStorageEngine)
        db_engine (sqlalchemy.engine)

    Returns:
        (int) model id for database retrieval
    """
    trained_model = MockTrainedModel()
    model_storage_engine.write(trained_model, "abcd")
    session = sessionmaker(db_engine)()
    db_model = Model(model_hash="abcd")
    session.add(db_model)
    session.commit()
    return trained_model, db_model.model_id


def assert_index(engine, table, column):
    """Assert that a table has an index on a given column

    Does not care which position the column is in the index
    Modified from https://www.gab.lc/articles/index_on_id_with_postgresql

    Args:
        engine (sqlalchemy.engine) a database engine
        table (string) the name of a table
        column (string) the name of a column
    """
    query = """
        SELECT 1
        FROM pg_class t
             JOIN pg_index ix ON t.oid = ix.indrelid
             JOIN pg_class i ON i.oid = ix.indexrelid
             JOIN pg_attribute a ON a.attrelid = t.oid
        WHERE
             a.attnum = ANY(ix.indkey) AND
             t.relkind = 'r' AND
             t.relname = '{table_name}' AND
             a.attname = '{column_name}'
    """.format(
        table_name=table, column_name=column
    )
    num_results = len([row for row in engine.execute(query)])
    assert num_results >= 1


def create_dense_state_table(db_engine, table_name, data):
    db_engine.execute(
        """create table {} (
        entity_id int,
        state text,
        start_time timestamp,
        end_time timestamp
    )""".format(
            table_name
        )
    )

    for row in data:
        db_engine.execute(
            "insert into {} values (%s, %s, %s, %s)".format(table_name), row
        )


def create_binary_outcome_events(db_engine, table_name, events_data):
    db_engine.execute(
        "create table events (entity_id int, outcome_date date, outcome bool)"
    )
    for event in events_data:
        db_engine.execute(
            "insert into {} values (%s, %s, %s::bool)".format(table_name), event
        )


def retry_if_db_error(exception):
    return isinstance(exception, sqlalchemy.exc.OperationalError)
