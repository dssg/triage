import datetime
import shutil
import sys
import tempfile
import random
from contextlib import contextmanager

import pandas as pd
import yaml
import numpy as np
from sqlalchemy import text

from triage.logging import ic


def convert_string_column_to_date(column):
    return [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in column]


def create_schemas(engine, features_tables, labels, states):
    """ This function makes a features schema and populates it with the fake
    data from above.

    :param engine: a postgresql engine
    :type engine: sqlalchemy.engine
    """
    # create features schema and associated tables
    with engine.connect() as conn:
        conn.execute(text("drop schema if exists features cascade; create schema features;"))
        conn.commit()
    for table_number, table in enumerate(features_tables):
        create_features_table(table_number, table, engine)
    # create labels schema and table
    create_labels(engine, labels)

    # create cohort table
    with engine.connect() as conn:
        conn.execute(text("drop schema if exists staging cascade; create schema staging;"))
        conn.execute(text(
            """
                create table cohort (
                    entity_id int,
                    as_of_date date,
                    active bool
                )
            """
        ))
        for row in states:
            conn.execute(
                text("insert into cohort values (:entity_id, :as_of_date, :active)"),
                {"entity_id": row[0], "as_of_date": row[1], "active": row[2]}
            )
        conn.commit()


def create_labels(engine, labels):
    with engine.connect() as conn:
        conn.execute(text("drop schema if exists labels cascade; create schema labels;"))
        conn.execute(text(
            """
                create table labels.labels (
                    entity_id int,
                    as_of_date date,
                    label_timespan interval,
                    label_name char(30),
                    label_type char(30),
                    label int
                )
            """
        ))
        for row in labels:
            conn.execute(
                text("insert into labels.labels values (:entity_id, :as_of_date, :label_timespan, :label_name, :label_type, :label)"),
                {"entity_id": row[0], "as_of_date": row[1], "label_timespan": row[2], "label_name": row[3], "label_type": row[4], "label": row[5]}
            )
        conn.commit()
    return 'labels.labels'

def create_features_table(table_number, table, engine):
    with engine.connect() as conn:
        conn.execute(text(
            """
                create table features.features{} (
                    entity_id int, as_of_date date, f{} int, f{} int
                )
            """.format(
                table_number, (table_number * 2) + 1, (table_number * 2) + 2
            )
        ))
        for row in table:
            conn.execute(
                text("""
                    insert into features.features{} values (:entity_id, :as_of_date, :f1, :f2)
                """.format(table_number)),
                {"entity_id": row[0], "as_of_date": row[1], "f1": row[2], "f2": row[3]}
            )
        conn.commit()


def create_entity_date_df(
    labels,
    states,
    as_of_dates,
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
        states, columns=["entity_id", "as_of_date", "active"]
    ).set_index(["entity_id", "as_of_date"])
    as_of_dates = [date.date() for date in as_of_dates]
    labels_table = labels_table[labels_table["label_name"] == label_name]
    labels_table = labels_table[labels_table["label_type"] == label_type]
    labels_table = labels_table[labels_table["label_timespan"] == label_timespan]
    labels_table = labels_table.join(other=states_table, on=("entity_id", "as_of_date"))
    labels_table = labels_table[labels_table["active"]]
    ids_dates = labels_table[["entity_id", "as_of_date"]]
    ids_dates = ids_dates.sort_values(["entity_id", "as_of_date"])
    ids_dates["as_of_date"] = [
        datetime.datetime.strptime(date, "%Y-%m-%d").date()
        for date in ids_dates["as_of_date"]
    ]
    ids_dates = ids_dates[ids_dates["as_of_date"].isin(as_of_dates)]
    ic(ids_dates)

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
             t.relname = :table_name AND
             a.attname = :column_name
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"table_name": table, "column_name": column})
        num_results = len([row for row in result])
    assert num_results >= 1


def create_binary_outcome_events(db_engine, table_name, events_data):
    with db_engine.connect() as conn:
        conn.execute(text(
            "create table events (entity_id int, outcome_date date, outcome bool)"
        ))
        for event in events_data:
            conn.execute(
                text("insert into {} values (:entity_id, :outcome_date, CAST(:outcome AS bool))".format(table_name)),
                {"entity_id": event[0], "outcome_date": event[1], "outcome": event[2]}
            )
        conn.commit()
