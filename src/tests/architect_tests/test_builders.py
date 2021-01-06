import datetime
from unittest import TestCase

import pandas as pd
import testing.postgresql
from mock import Mock
from triage import create_engine
from contextlib import contextmanager

from triage.component.catwalk.utils import filename_friendly_hash
from triage.component.architect.feature_group_creator import FeatureGroup
from triage.component.architect.builders import MatrixBuilder
from triage.component.catwalk.db import ensure_db
from triage.component.catwalk.storage import ProjectStorage
from triage.component.results_schema.schema import Matrix

from .utils import (
    create_schemas,
    create_entity_date_df,
    convert_string_column_to_date,
    TemporaryDirectory,
)

from tests.utils import matrix_metadata_creator


# make some fake features data

states = [
    [0, "2016-01-01", False],
    [0, "2016-02-01", False],
    [0, "2016-03-01", False],
    [0, "2016-04-01", False],
    [0, "2016-05-01", False],
    [0, "2016-06-01", True],
    [1, "2016-01-01", False],
    [1, "2016-02-01", False],
    [1, "2016-03-01", False],
    [1, "2016-04-01", False],
    [1, "2016-05-01", False],
    [2, "2016-01-01", False],
    [2, "2016-02-01", True],
    [2, "2016-03-01", False],
    [2, "2016-04-01", True],
    [2, "2016-05-01", False],
    [3, "2016-01-01", False],
    [3, "2016-02-01", True],
    [3, "2016-03-01", False],
    [3, "2016-04-01", True],
    [3, "2016-05-01", False],
    [4, "2016-01-01", True],
    [4, "2016-02-01", True],
    [4, "2016-03-01", True],
    [4, "2016-04-01", True],
    [4, "2016-05-01", True],
    [5, "2016-01-01", False],
    [5, "2016-02-01", False],
    [5, "2016-03-01", False],
    [5, "2016-04-01", False],
    [5, "2016-05-01", False],
]

features0_pre = [
    [0, "2016-01-01", 2, 1],
    [1, "2016-01-01", 1, 2],
    [0, "2016-02-01", 2, 3],
    [1, "2016-02-01", 2, 4],
    [0, "2016-03-01", 3, 3],
    [1, "2016-03-01", 3, 4],
    [0, "2016-04-01", 4, 3],
    [1, "2016-05-01", 5, 4],
]

features1_pre = [
    [2, "2016-01-01", 1, 1],
    [3, "2016-01-01", 1, 2],
    [2, "2016-02-01", 2, 3],
    [3, "2016-02-01", 2, 2],
    [0, "2016-03-01", 3, 3],
    [1, "2016-03-01", 3, 4],
    [2, "2016-03-01", 3, 3],
    [3, "2016-03-01", 3, 4],
    [3, "2016-03-01", 3, 4],
    [0, "2016-03-01", 3, 3],
    [4, "2016-03-01", 1, 4],
    [5, "2016-03-01", 2, 4],
]

# collate will ensure every entity/date combination in the state
# table have an imputed value in the features table, so ensure
# this is true for our test (filling with 9's):
f0_dict = {(r[0], r[1]): r for r in features0_pre}
f1_dict = {(r[0], r[1]): r for r in features1_pre}

for rec in states:
    ent_dt = (rec[0], rec[1])
    f0_dict[ent_dt] = f0_dict.get(ent_dt, list(ent_dt + (9, 9)))
    f1_dict[ent_dt] = f1_dict.get(ent_dt, list(ent_dt + (9, 9)))

features0 = sorted(f0_dict.values(), key=lambda x: (x[1], x[0]))
features1 = sorted(f1_dict.values(), key=lambda x: (x[1], x[0]))

features_tables = [features0, features1]

# make some fake labels data

labels = [
    [0, "2016-02-01", "1 month", "booking", "binary", 0],
    [0, "2016-03-01", "1 month", "booking", "binary", 0],
    [0, "2016-04-01", "1 month", "booking", "binary", 0],
    [0, "2016-05-01", "1 month", "booking", "binary", 1],
    [0, "2016-01-01", "1 month", "ems", "binary", 0],
    [0, "2016-02-01", "1 month", "ems", "binary", 0],
    [0, "2016-03-01", "1 month", "ems", "binary", 0],
    [0, "2016-04-01", "1 month", "ems", "binary", 0],
    [0, "2016-05-01", "1 month", "ems", "binary", 0],
    [1, "2016-01-01", "1 month", "booking", "binary", 0],
    [1, "2016-02-01", "1 month", "booking", "binary", 0],
    [1, "2016-03-01", "1 month", "booking", "binary", 0],
    [1, "2016-04-01", "1 month", "booking", "binary", 0],
    [1, "2016-05-01", "1 month", "booking", "binary", 1],
    [1, "2016-01-01", "1 month", "ems", "binary", 0],
    [1, "2016-02-01", "1 month", "ems", "binary", 0],
    [1, "2016-03-01", "1 month", "ems", "binary", 0],
    [1, "2016-04-01", "1 month", "ems", "binary", 0],
    [1, "2016-05-01", "1 month", "ems", "binary", 0],
    [2, "2016-01-01", "1 month", "booking", "binary", 0],
    [2, "2016-02-01", "1 month", "booking", "binary", 0],
    [2, "2016-03-01", "1 month", "booking", "binary", 1],
    [2, "2016-04-01", "1 month", "booking", "binary", 0],
    [2, "2016-05-01", "1 month", "booking", "binary", 1],
    [2, "2016-01-01", "1 month", "ems", "binary", 0],
    [2, "2016-02-01", "1 month", "ems", "binary", 0],
    [2, "2016-03-01", "1 month", "ems", "binary", 0],
    [2, "2016-04-01", "1 month", "ems", "binary", 0],
    [2, "2016-05-01", "1 month", "ems", "binary", 1],
    [3, "2016-01-01", "1 month", "booking", "binary", 0],
    [3, "2016-02-01", "1 month", "booking", "binary", 0],
    [3, "2016-03-01", "1 month", "booking", "binary", 1],
    [3, "2016-04-01", "1 month", "booking", "binary", 0],
    [3, "2016-05-01", "1 month", "booking", "binary", 1],
    [3, "2016-01-01", "1 month", "ems", "binary", 0],
    [3, "2016-02-01", "1 month", "ems", "binary", 0],
    [3, "2016-03-01", "1 month", "ems", "binary", 0],
    [3, "2016-04-01", "1 month", "ems", "binary", 1],
    [3, "2016-05-01", "1 month", "ems", "binary", 0],
    [4, "2016-01-01", "1 month", "booking", "binary", 1],
    [4, "2016-02-01", "1 month", "booking", "binary", 0],
    [4, "2016-03-01", "1 month", "booking", "binary", 0],
    [4, "2016-04-01", "1 month", "booking", "binary", 0],
    [4, "2016-05-01", "1 month", "booking", "binary", 0],
    [4, "2016-01-01", "1 month", "ems", "binary", 0],
    [4, "2016-02-01", "1 month", "ems", "binary", 1],
    [4, "2016-03-01", "1 month", "ems", "binary", 0],
    [4, "2016-04-01", "1 month", "ems", "binary", 1],
    [4, "2016-05-01", "1 month", "ems", "binary", 1],
    [5, "2016-01-01", "1 month", "booking", "binary", 1],
    [5, "2016-02-01", "1 month", "booking", "binary", 0],
    [5, "2016-03-01", "1 month", "booking", "binary", 0],
    [5, "2016-04-01", "1 month", "booking", "binary", 0],
    [5, "2016-05-01", "1 month", "booking", "binary", 0],
    [5, "2016-01-01", "1 month", "ems", "binary", 0],
    [5, "2016-02-01", "1 month", "ems", "binary", 1],
    [5, "2016-03-01", "1 month", "ems", "binary", 0],
    [5, "2016-04-01", "1 month", "ems", "binary", 0],
    [5, "2016-05-01", "1 month", "ems", "binary", 0],
    [0, "2016-02-01", "3 month", "booking", "binary", 0],
    [0, "2016-03-01", "3 month", "booking", "binary", 0],
    [0, "2016-04-01", "3 month", "booking", "binary", 0],
    [0, "2016-05-01", "3 month", "booking", "binary", 1],
    [0, "2016-01-01", "3 month", "ems", "binary", 0],
    [0, "2016-02-01", "3 month", "ems", "binary", 0],
    [0, "2016-03-01", "3 month", "ems", "binary", 0],
    [0, "2016-04-01", "3 month", "ems", "binary", 0],
    [0, "2016-05-01", "3 month", "ems", "binary", 0],
    [1, "2016-01-01", "3 month", "booking", "binary", 0],
    [1, "2016-02-01", "3 month", "booking", "binary", 0],
    [1, "2016-03-01", "3 month", "booking", "binary", 0],
    [1, "2016-04-01", "3 month", "booking", "binary", 0],
    [1, "2016-05-01", "3 month", "booking", "binary", 1],
    [1, "2016-01-01", "3 month", "ems", "binary", 0],
    [1, "2016-02-01", "3 month", "ems", "binary", 0],
    [1, "2016-03-01", "3 month", "ems", "binary", 0],
    [1, "2016-04-01", "3 month", "ems", "binary", 0],
    [1, "2016-05-01", "3 month", "ems", "binary", 0],
    [2, "2016-01-01", "3 month", "booking", "binary", 0],
    [2, "2016-02-01", "3 month", "booking", "binary", 0],
    [2, "2016-03-01", "3 month", "booking", "binary", 1],
    [2, "2016-04-01", "3 month", "booking", "binary", 0],
    [2, "2016-05-01", "3 month", "booking", "binary", 1],
    [2, "2016-01-01", "3 month", "ems", "binary", 0],
    [2, "2016-02-01", "3 month", "ems", "binary", 0],
    [2, "2016-03-01", "3 month", "ems", "binary", 0],
    [2, "2016-04-01", "3 month", "ems", "binary", 0],
    [2, "2016-05-01", "3 month", "ems", "binary", 1],
    [3, "2016-01-01", "3 month", "booking", "binary", 0],
    [3, "2016-02-01", "3 month", "booking", "binary", 0],
    [3, "2016-03-01", "3 month", "booking", "binary", 1],
    [3, "2016-04-01", "3 month", "booking", "binary", 0],
    [3, "2016-05-01", "3 month", "booking", "binary", 1],
    [3, "2016-01-01", "3 month", "ems", "binary", 0],
    [3, "2016-02-01", "3 month", "ems", "binary", 0],
    [3, "2016-03-01", "3 month", "ems", "binary", 0],
    [3, "2016-04-01", "3 month", "ems", "binary", 1],
    [3, "2016-05-01", "3 month", "ems", "binary", 0],
    [3, "2016-05-01", "3 month", "ems", "binary", 0],
    [4, "2016-01-01", "3 month", "booking", "binary", 0],
    [4, "2016-02-01", "3 month", "booking", "binary", 0],
    [4, "2016-03-01", "3 month", "booking", "binary", 1],
    [4, "2016-04-01", "3 month", "booking", "binary", 0],
    [4, "2016-05-01", "3 month", "booking", "binary", 1],
    [4, "2016-01-01", "3 month", "ems", "binary", 0],
    [4, "2016-02-01", "3 month", "ems", "binary", 0],
    [4, "2016-03-01", "3 month", "ems", "binary", 0],
    [4, "2016-04-01", "3 month", "ems", "binary", 0],
    [4, "2016-05-01", "3 month", "ems", "binary", 1],
    [5, "2016-01-01", "3 month", "booking", "binary", 0],
    [5, "2016-02-01", "3 month", "booking", "binary", 0],
    [5, "2016-03-01", "3 month", "booking", "binary", 1],
    [5, "2016-04-01", "3 month", "booking", "binary", 0],
    [5, "2016-05-01", "3 month", "booking", "binary", 1],
    [5, "2016-01-01", "3 month", "ems", "binary", 0],
    [5, "2016-02-01", "3 month", "ems", "binary", 0],
    [5, "2016-03-01", "3 month", "ems", "binary", 0],
    [5, "2016-04-01", "3 month", "ems", "binary", 1],
    [5, "2016-05-01", "3 month", "ems", "binary", 0],
]

label_name = "booking"
label_type = "binary"

db_config = {
    "features_schema_name": "features",
    "labels_schema_name": "labels",
    "labels_table_name": "labels",
    "cohort_table_name": "cohort",
}

experiment_hash = None


@contextmanager
def get_matrix_storage_engine():
    with TemporaryDirectory() as temp_dir:
        yield ProjectStorage(temp_dir).matrix_storage_engine()


def test_query_to_df():
    """ Test the write_to_csv function by checking whether the csv contains the
    correct number of lines.
    """
    with testing.postgresql.Postgresql() as postgresql:
        # create an engine and generate a table with fake feature data
        engine = create_engine(postgresql.url())
        create_schemas(
            engine=engine, features_tables=features_tables, labels=labels, states=states
        )

        with get_matrix_storage_engine() as matrix_storage_engine:
            builder = MatrixBuilder(
                db_config=db_config,
                matrix_storage_engine=matrix_storage_engine,
                experiment_hash=experiment_hash,
                engine=engine,
            )

            # for each table, check that corresponding csv has the correct # of rows
            for table in features_tables:
                df = builder.query_to_df(
                    """
                        select *
                        from features.features{}
                    """.format(
                        features_tables.index(table)
                    )
                )
                assert len(df) == len(table)


def test_make_entity_date_table():
    """ Test that the make_entity_date_table function contains the correct
    values.
    """
    dates = [
        datetime.datetime(2016, 1, 1, 0, 0),
        datetime.datetime(2016, 2, 1, 0, 0),
        datetime.datetime(2016, 3, 1, 0, 0),
    ]

    # make a dataframe of entity ids and dates to test against
    ids_dates = create_entity_date_df(
        labels=labels,
        states=states,
        as_of_dates=dates,
        label_name="booking",
        label_type="binary",
        label_timespan="1 month",
    )

    with testing.postgresql.Postgresql() as postgresql:
        # create an engine and generate a table with fake feature data
        engine = create_engine(postgresql.url())
        create_schemas(
            engine=engine, features_tables=features_tables, labels=labels, states=states
        )

        with get_matrix_storage_engine() as matrix_storage_engine:
            builder = MatrixBuilder(
                db_config=db_config,
                matrix_storage_engine=matrix_storage_engine,
                experiment_hash=experiment_hash,
                engine=engine,
            )
            engine.execute("CREATE TABLE features.tmp_entity_date (a int, b date);")
            # call the function to test the creation of the table
            entity_date_table_name = builder.make_entity_date_table(
                as_of_times=dates,
                label_type="binary",
                label_name="booking",
                state="active",
                matrix_uuid="my_uuid",
                matrix_type="train",
                label_timespan="1 month",
            )

            # read in the table
            result = pd.read_sql(
                "select * from features.{} order by entity_id, as_of_date".format(
                    entity_date_table_name
                ),
                engine,
            )
            # compare the table to the test dataframe
            test = result == ids_dates
            assert test.all().all()


def test_make_entity_date_table_include_missing_labels():
    """ Test that the make_entity_date_table function contains the correct
    values.
    """
    dates = [
        datetime.datetime(2016, 1, 1, 0, 0),
        datetime.datetime(2016, 2, 1, 0, 0),
        datetime.datetime(2016, 3, 1, 0, 0),
        datetime.datetime(2016, 6, 1, 0, 0),
    ]

    # same as the other make_entity_date_label test except there is an extra date, 2016-06-01
    # entity 0 is included in this date via the states table, but has no label

    # make a dataframe of entity ids and dates to test against
    ids_dates = create_entity_date_df(
        labels=labels,
        states=states,
        as_of_dates=dates,
        label_name="booking",
        label_type="binary",
        label_timespan="1 month",
    )
    # this line adds the new entity-date combo as an expected one
    ids_dates = ids_dates.append(
        {"entity_id": 0, "as_of_date": datetime.date(2016, 6, 1)}, ignore_index=True
    )

    with testing.postgresql.Postgresql() as postgresql:
        # create an engine and generate a table with fake feature data
        engine = create_engine(postgresql.url())
        create_schemas(
            engine=engine, features_tables=features_tables, labels=labels, states=states
        )

        with get_matrix_storage_engine() as matrix_storage_engine:
            builder = MatrixBuilder(
                db_config=db_config,
                matrix_storage_engine=matrix_storage_engine,
                experiment_hash=experiment_hash,
                include_missing_labels_in_train_as=False,
                engine=engine,
            )
            engine.execute("CREATE TABLE features.tmp_entity_date (a int, b date);")
            # call the function to test the creation of the table
            entity_date_table_name = builder.make_entity_date_table(
                as_of_times=dates,
                label_type="binary",
                label_name="booking",
                state="active",
                matrix_uuid="my_uuid",
                matrix_type="train",
                label_timespan="1 month",
            )

            # read in the table
            result = pd.read_sql(
                "select * from features.{} order by entity_id, as_of_date".format(
                    entity_date_table_name
                ),
                engine,
            )

            # compare the table to the test dataframe
            assert sorted(result.values.tolist()) == sorted(ids_dates.values.tolist())


def test_load_features_data():
    dates = [datetime.datetime(2016, 1, 1, 0, 0), datetime.datetime(2016, 2, 1, 0, 0)]

    # make dataframe for entity ids and dates
    ids_dates = create_entity_date_df(
        labels=labels,
        states=states,
        as_of_dates=dates,
        label_name="booking",
        label_type="binary",
        label_timespan="1 month",
    )

    features = [["f1", "f2"], ["f3", "f4"]]
    # make dataframes of features to test against
    features_dfs = []
    for i, table in enumerate(features_tables):
        cols = ["entity_id", "as_of_date"] + features[i]
        temp_df = pd.DataFrame(table, columns=cols)
        temp_df["as_of_date"] = convert_string_column_to_date(temp_df["as_of_date"])
        features_dfs.append(
            ids_dates.merge(
                right=temp_df, how="left", on=["entity_id", "as_of_date"]
            ).set_index(["entity_id", "as_of_date"])
        )

    # create an engine and generate a table with fake feature data
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_schemas(
            engine=engine, features_tables=features_tables, labels=labels, states=states
        )

        with get_matrix_storage_engine() as matrix_storage_engine:
            builder = MatrixBuilder(
                db_config=db_config,
                matrix_storage_engine=matrix_storage_engine,
                experiment_hash=experiment_hash,
                engine=engine,
            )

            # make the entity-date table
            entity_date_table_name = builder.make_entity_date_table(
                as_of_times=dates,
                label_type="binary",
                label_name="booking",
                state="active",
                matrix_type="train",
                matrix_uuid="my_uuid",
                label_timespan="1 month",
            )

            feature_dictionary = dict(
                ("features{}".format(i), feature_list)
                for i, feature_list in enumerate(features)
            )

            returned_features_dfs = builder.load_features_data(
                as_of_times=dates,
                feature_dictionary=feature_dictionary,
                entity_date_table_name=entity_date_table_name,
                matrix_uuid="my_uuid",
            )

            # get the queries and test them
            for result, df in zip(returned_features_dfs, features_dfs):
                test = result == df
                assert test.all().all()


def test_load_labels_data():
    """ Test the load_labels_data function by checking whether the query
    produces the correct labels
    """
    # set up labeling config variables
    dates = [datetime.datetime(2016, 1, 1, 0, 0), datetime.datetime(2016, 2, 1, 0, 0)]

    # make a dataframe of labels to test against
    labels_df = pd.DataFrame(
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

    labels_df["as_of_date"] = convert_string_column_to_date(labels_df["as_of_date"])
    labels_df.set_index(["entity_id", "as_of_date"])

    # create an engine and generate a table with fake feature data
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_schemas(engine, features_tables, labels, states)
        with get_matrix_storage_engine() as matrix_storage_engine:
            builder = MatrixBuilder(
                db_config=db_config,
                matrix_storage_engine=matrix_storage_engine,
                experiment_hash=experiment_hash,
                engine=engine,
            )

            # make the entity-date table
            entity_date_table_name = builder.make_entity_date_table(
                as_of_times=dates,
                label_type="binary",
                label_name="booking",
                state="active",
                matrix_type="train",
                matrix_uuid="my_uuid",
                label_timespan="1 month",
            )

            result = builder.load_labels_data(
                label_name=label_name,
                label_type=label_type,
                label_timespan="1 month",
                matrix_uuid="my_uuid",
                entity_date_table_name=entity_date_table_name,
            )
            df = pd.DataFrame.from_dict(
                {
                    "entity_id": [2, 3, 4, 4],
                    "as_of_date": [dates[1], dates[1], dates[0], dates[1]],
                    "booking": [0, 0, 1, 0],
                }
            ).set_index(["entity_id", "as_of_date"])

            test = result == df
            assert test.all().all()


def test_load_labels_data_include_missing_labels_as_false():
    """ Test the load_labels_data function by checking whether the query
    produces the correct labels
    """
    # set up labeling config variables
    dates = [
        datetime.datetime(2016, 1, 1, 0, 0),
        datetime.datetime(2016, 2, 1, 0, 0),
        datetime.datetime(2016, 6, 1, 0, 0),
    ]

    # same as the other load_labels_data test, except we include an extra date, 2016-06-01
    # this date does have entity 0 included via the states table, but no labels

    # make a dataframe of labels to test against
    labels_df = pd.DataFrame(
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

    labels_df["as_of_date"] = convert_string_column_to_date(labels_df["as_of_date"])
    labels_df.set_index(["entity_id", "as_of_date"])

    # create an engine and generate a table with fake feature data
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_schemas(engine, features_tables, labels, states)
        with get_matrix_storage_engine() as matrix_storage_engine:
            builder = MatrixBuilder(
                db_config=db_config,
                matrix_storage_engine=matrix_storage_engine,
                experiment_hash=experiment_hash,
                engine=engine,
                include_missing_labels_in_train_as=False,
            )

            # make the entity-date table
            entity_date_table_name = builder.make_entity_date_table(
                as_of_times=dates,
                label_type="binary",
                label_name="booking",
                state="active",
                matrix_type="train",
                matrix_uuid="my_uuid",
                label_timespan="1 month",
            )

            result = builder.load_labels_data(
                label_name=label_name,
                label_type=label_type,
                label_timespan="1 month",
                matrix_uuid="my_uuid",
                entity_date_table_name=entity_date_table_name,
            )
            df = pd.DataFrame.from_dict(
                {
                    "entity_id": [0, 2, 3, 4, 4],
                    "as_of_date": [dates[2], dates[1], dates[1], dates[0], dates[1]],
                    "booking": [0, 0, 0, 1, 0],
                }
            ).set_index(["entity_id", "as_of_date"])
            # the first row would not be here if we had not configured the Builder
            # to include missing labels as false

            test = result == df
            assert test.all().all()


class TestMergeFeatureCSVs(TestCase):
    def test_badinput(self):
        """We assert column names, so replacing 'date' with 'as_of_date'
        should result in an error"""
        with get_matrix_storage_engine() as matrix_storage_engine:
            builder = MatrixBuilder(
                db_config=db_config,
                matrix_storage_engine=matrix_storage_engine,
                experiment_hash=experiment_hash,
                engine=None,
            )
            dataframes = [
                pd.DataFrame.from_records(
                    [(1, 3, 3), (4, 5, 6), (7, 8, 9)],
                    columns=("entity_id", "date", "f1"),
                    index=["entity_id", "date"],
                ),
                pd.DataFrame.from_records(
                    [(1, 2, 3), (4, 5, 9), (7, 8, 15)],
                    columns=("entity_id", "date", "f3"),
                    index=["entity_id", "date"],
                ),
                pd.DataFrame.from_records(
                    [(1, 2, 2), (4, 5, 20), (7, 8, 56)],
                    columns=("entity_id", "date", "f3"),
                    index=["entity_id", "date"],
                ),
            ]

            with self.assertRaises(ValueError):
                builder.merge_feature_csvs(dataframes, matrix_uuid="1234")


class TestBuildMatrix(TestCase):
    @property
    def good_metadata(self):
        return {
            "matrix_id": "hi",
            "state": "active",
            "label_name": "booking",
            "end_time": datetime.datetime(2016, 3, 1, 0, 0),
            "feature_start_time": datetime.datetime(2016, 1, 1, 0, 0),
            "label_timespan": "1 month",
            "max_training_history": "1 month",
            "test_duration": "1 month",
            "indices": ["entity_id", "as_of_date"],
        }

    @property
    def good_feature_dictionary(self):
        return FeatureGroup(
            name="mygroup",
            features_by_table={"features0": ["f1", "f2"], "features1": ["f3", "f4"]},
        )

    @property
    def good_dates(self):
        return [
            datetime.datetime(2016, 1, 1, 0, 0),
            datetime.datetime(2016, 2, 1, 0, 0),
            datetime.datetime(2016, 3, 1, 0, 0),
        ]

    def test_train_matrix(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            ensure_db(engine)
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states,
            )

            with get_matrix_storage_engine() as matrix_storage_engine:
                builder = MatrixBuilder(
                    db_config=db_config,
                    matrix_storage_engine=matrix_storage_engine,
                    experiment_hash=experiment_hash,
                    engine=engine,
                )
                uuid = filename_friendly_hash(self.good_metadata)
                builder.build_matrix(
                    as_of_times=self.good_dates,
                    label_name="booking",
                    label_type="binary",
                    feature_dictionary=self.good_feature_dictionary,
                    matrix_metadata=self.good_metadata,
                    matrix_uuid=uuid,
                    matrix_type="train",
                )
                assert len(matrix_storage_engine.get_store(uuid).design_matrix) == 5
                assert builder.sessionmaker().query(Matrix).get(uuid).feature_dictionary ==self.good_feature_dictionary

    def test_test_matrix(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            ensure_db(engine)
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states,
            )

            with get_matrix_storage_engine() as matrix_storage_engine:
                builder = MatrixBuilder(
                    db_config=db_config,
                    matrix_storage_engine=matrix_storage_engine,
                    experiment_hash=experiment_hash,
                    engine=engine,
                )

                uuid = filename_friendly_hash(self.good_metadata)
                builder.build_matrix(
                    as_of_times=self.good_dates,
                    label_name="booking",
                    label_type="binary",
                    feature_dictionary=self.good_feature_dictionary,
                    matrix_metadata=self.good_metadata,
                    matrix_uuid=uuid,
                    matrix_type="test",
                )

                assert len(matrix_storage_engine.get_store(uuid).design_matrix) == 5

    def test_nullcheck(self):
        f0_dict = {(r[0], r[1]): r for r in features0_pre}
        f1_dict = {(r[0], r[1]): r for r in features1_pre}

        features0 = sorted(f0_dict.values(), key=lambda x: (x[1], x[0]))
        features1 = sorted(f1_dict.values(), key=lambda x: (x[1], x[0]))

        features_tables = [features0, features1]

        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states,
            )

            dates = [
                datetime.datetime(2016, 1, 1, 0, 0),
                datetime.datetime(2016, 2, 1, 0, 0),
                datetime.datetime(2016, 3, 1, 0, 0),
            ]

            with get_matrix_storage_engine() as matrix_storage_engine:
                builder = MatrixBuilder(
                    db_config=db_config,
                    matrix_storage_engine=matrix_storage_engine,
                    experiment_hash=experiment_hash,
                    engine=engine,
                )

                feature_dictionary = {
                    "features0": ["f1", "f2"],
                    "features1": ["f3", "f4"],
                }
                matrix_metadata = {
                    "matrix_id": "hi",
                    "state": "active",
                    "label_name": "booking",
                    "end_time": datetime.datetime(2016, 3, 1, 0, 0),
                    "feature_start_time": datetime.datetime(2016, 1, 1, 0, 0),
                    "label_timespan": "1 month",
                    "test_duration": "1 month",
                    "indices": ["entity_id", "as_of_date"],
                }
                uuid = filename_friendly_hash(matrix_metadata)
                with self.assertRaises(ValueError):
                    builder.build_matrix(
                        as_of_times=dates,
                        label_name="booking",
                        label_type="binary",
                        feature_dictionary=feature_dictionary,
                        matrix_metadata=matrix_metadata,
                        matrix_uuid=uuid,
                        matrix_type="test",
                    )

    def test_replace_false_rerun(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            ensure_db(engine)
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states,
            )

            dates = [
                datetime.datetime(2016, 1, 1, 0, 0),
                datetime.datetime(2016, 2, 1, 0, 0),
                datetime.datetime(2016, 3, 1, 0, 0),
            ]

            with get_matrix_storage_engine() as matrix_storage_engine:
                builder = MatrixBuilder(
                    db_config=db_config,
                    matrix_storage_engine=matrix_storage_engine,
                    experiment_hash=experiment_hash,
                    engine=engine,
                    replace=False,
                )

                feature_dictionary = {
                    "features0": ["f1", "f2"],
                    "features1": ["f3", "f4"],
                }
                matrix_metadata = {
                    "matrix_id": "hi",
                    "state": "active",
                    "label_name": "booking",
                    "end_time": datetime.datetime(2016, 3, 1, 0, 0),
                    "feature_start_time": datetime.datetime(2016, 1, 1, 0, 0),
                    "label_timespan": "1 month",
                    "test_duration": "1 month",
                    "indices": ["entity_id", "as_of_date"],
                }
                uuid = filename_friendly_hash(matrix_metadata)
                builder.build_matrix(
                    as_of_times=dates,
                    label_name="booking",
                    label_type="binary",
                    feature_dictionary=feature_dictionary,
                    matrix_metadata=matrix_metadata,
                    matrix_uuid=uuid,
                    matrix_type="test",
                )

                assert len(matrix_storage_engine.get_store(uuid).design_matrix) == 5
                # rerun
                builder.make_entity_date_table = Mock()
                builder.build_matrix(
                    as_of_times=dates,
                    label_name="booking",
                    label_type="binary",
                    feature_dictionary=feature_dictionary,
                    matrix_metadata=matrix_metadata,
                    matrix_uuid=uuid,
                    matrix_type="test",
                )
                assert not builder.make_entity_date_table.called

    def test_replace_true_rerun(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            ensure_db(engine)
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states,
            )
            matrix_metadata = matrix_metadata_creator(
                state="active", test_duration="1month", label_name="booking"
            )

            dates = [
                datetime.datetime(2016, 1, 1, 0, 0),
                datetime.datetime(2016, 2, 1, 0, 0),
                datetime.datetime(2016, 3, 1, 0, 0),
            ]

            feature_dictionary = {"features0": ["f1", "f2"], "features1": ["f3", "f4"]}
            uuid = filename_friendly_hash(matrix_metadata)
            build_args = dict(
                as_of_times=dates,
                label_name="booking",
                label_type="binary",
                feature_dictionary=feature_dictionary,
                matrix_metadata=matrix_metadata,
                matrix_uuid=uuid,
                matrix_type="test",
            )

            with get_matrix_storage_engine() as matrix_storage_engine:
                builder = MatrixBuilder(
                    db_config=db_config,
                    matrix_storage_engine=matrix_storage_engine,
                    experiment_hash=experiment_hash,
                    engine=engine,
                    replace=True,
                )

                builder.build_matrix(**build_args)

                assert len(matrix_storage_engine.get_store(uuid).design_matrix) == 5
                assert builder.sessionmaker().query(Matrix).get(uuid)
                # rerun
                builder.build_matrix(**build_args)
                assert len(matrix_storage_engine.get_store(uuid).design_matrix) == 5
                assert builder.sessionmaker().query(Matrix).get(uuid)
