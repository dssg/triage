import datetime
from unittest import TestCase, mock
import pandas as pd
import testing.postgresql

from contextlib import contextmanager

from triage import create_engine
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
    "triage_metadata": "triage_metadata",
}

experiment_hash = None


@contextmanager
def get_matrix_storage_engine():
    with TemporaryDirectory() as temp_dir:
        yield ProjectStorage(temp_dir).matrix_storage_engine()


def test_make_entity_date_table():
    """Test that the make_entity_date_table function contains the correct
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
        #ensure_db(engine)
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
    """Test that the make_entity_date_table function contains the correct
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
        #ensure_db(engine)
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


class TestMergeFeatureCSVs(TestCase):
    def test_feature_load_queries(self):
        """Tests if the number of queries for getting the features are the same as the number of feature tables in
        the feature schema.
        """
        
        dates = [
            datetime.datetime(2016, 1, 1, 0, 0),
            datetime.datetime(2016, 2, 1, 0, 0),
            datetime.datetime(2016, 3, 1, 0, 0),
            datetime.datetime(2016, 6, 1, 0, 0),
        ]

        features = [["f1", "f2"], ["f3", "f4"]]

        # create an engine and generate a table with fake feature data
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            #ensure_db(engine)
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
                    matrix_uuid="1234",
                    label_timespan="1m",
                )

                feature_dictionary = {
                    f"features{i}": feature_list
                    for i, feature_list in enumerate(features)
                }

                result = builder.feature_load_queries(
                    feature_dictionary=feature_dictionary, 
                    entity_date_table_name=entity_date_table_name
                )
                
                # lenght of the list should be the number of tables in feature schema
                assert len(result) == len(features)


    def test_stitch_csvs(self):
        """Tests if all the features and label were joined correctly in the csv
        """
        dates = [
            datetime.datetime(2016, 1, 1, 0, 0),
            datetime.datetime(2016, 2, 1, 0, 0),
            datetime.datetime(2016, 3, 1, 0, 0),
            datetime.datetime(2016, 6, 1, 0, 0),
        ]

        features = [["f1", "f2"], ["f3", "f4"]]

        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            #ensure_db(engine)
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

                feature_dictionary = {
                    f"features{i}": feature_list
                    for i, feature_list in enumerate(features)
                }

                # make the entity-date table
                entity_date_table_name = builder.make_entity_date_table(
                    as_of_times=dates,
                    label_type="binary",
                    label_name="booking",
                    state="active",
                    matrix_type="train",
                    matrix_uuid="1234",
                    label_timespan="1 month",
                )

                feature_queries = builder.feature_load_queries(
                    feature_dictionary=feature_dictionary,
                    entity_date_table_name=entity_date_table_name
                )

                label_query = builder.label_load_query(
                    label_name="booking",
                    label_type="binary",
                    entity_date_table_name=entity_date_table_name,
                    label_timespan='1 month'
                )

                matrix_store = matrix_storage_engine.get_store("1234")
                
                result = builder.stitch_csvs(
                    features_queries=feature_queries,
                    label_query=label_query,
                    matrix_store=matrix_store,
                    matrix_uuid="1234"
                )

                # chekc if entity_id and as_of_date are as index 
                should_be = ['entity_id', 'as_of_date']
                actual_indices = result.index.names

                TestCase().assertListEqual(should_be, actual_indices)

                # last element in the DF should be the label
                last_col = 'booking'
                output = result.columns.values[-1] # label name

                TestCase().assertEqual(last_col, output)

                # number of columns must be the sum of all the columns on each feature table + 1 for the label 
                TestCase().assertEqual(result.shape[1], 4+1, 
                                       "Number of features and label doesn't match")

                # number of rows 
                assert result.shape[0] ==  5
                TestCase().assertEqual(result.shape[0], 5, 
                                       "Number of rows doesn't match")

                # types of the final df should be float32
                types = set(result.apply(lambda x: x.dtype == 'float32').values)
                TestCase().assertTrue(types, "NOT all cols in matrix are float32!")


class TestBuildMatrix(TestCase):
    
    def test_train_matrix(self):
        dates = [
                    datetime.datetime(2016, 1, 1, 0, 0),
                    datetime.datetime(2016, 2, 1, 0, 0),
                    datetime.datetime(2016, 3, 1, 0, 0),
                ]

        features = [["f1", "f2"], ["f3", "f4"]]
        
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

                good_metadata = {
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

                feature_dictionary = {
                    f"features{i}": feature_list
                    for i, feature_list in enumerate(features)
                }

                uuid = filename_friendly_hash(good_metadata)
                builder.build_matrix(
                    as_of_times=dates,
                    label_name="booking",
                    label_type="binary",
                    feature_dictionary=feature_dictionary,
                    matrix_metadata=good_metadata,
                    matrix_uuid=uuid,
                    matrix_type="train",
                )
            
                assert len(matrix_storage_engine.get_store(uuid).design_matrix) == 5

                #engine_ = create_engine(postgresql.url())
                #assert (
                builder.sessionmaker().query(Matrix)#.get(uuid).feature_dictionary
                #   == feature_dictionary
                #)     
    
    def test_test_matrix(self):
        dates = [
                    datetime.datetime(2016, 1, 1, 0, 0),
                    datetime.datetime(2016, 2, 1, 0, 0),
                    datetime.datetime(2016, 3, 1, 0, 0),
                ]

        features = [["f1", "f2"], ["f3", "f4"]]

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

                good_metadata = {
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

                feature_dictionary = {
                    f"features{i}": feature_list
                    for i, feature_list in enumerate(features)
                }

                uuid = filename_friendly_hash(good_metadata)
                builder.build_matrix(
                    as_of_times=dates,
                    label_name="booking",
                    label_type="binary",
                    feature_dictionary=feature_dictionary,
                    matrix_metadata=good_metadata,
                    matrix_uuid=uuid,
                    matrix_type="test",
                )

                assert len(matrix_storage_engine.get_store(uuid).design_matrix) == 5
    

    def test_nullcheck(self):
        dates = [
                    datetime.datetime(2016, 1, 1, 0, 0),
                    datetime.datetime(2016, 2, 1, 0, 0),
                    datetime.datetime(2016, 3, 1, 0, 0),
                ]

        features = [["f1", "f2"], ["f3", "f4"]]

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

                good_metadata = {
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

                feature_dictionary = {
                    f"features{i}": feature_list
                    for i, feature_list in enumerate(features)
                }

                uuid = filename_friendly_hash(good_metadata)
                with self.assertRaises(ValueError):
                    builder.build_matrix(
                        as_of_times=dates,
                        label_name="booking",
                        label_type="binary",
                        feature_dictionary=feature_dictionary,
                        matrix_metadata=good_metadata,
                        matrix_uuid=uuid,
                        matrix_type="other",
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
                builder.make_entity_date_table = mock.Mock()
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
    