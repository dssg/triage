import csv
import datetime
import os
import uuid
from unittest import TestCase

import pandas as pd
import testing.postgresql
from mock import Mock
from sqlalchemy import create_engine

from triage.component import metta
from triage.component.architect import Planner, builders
from triage.component.architect.feature_group_creator import FeatureGroup

from .utils import (
    create_schemas,
    create_entity_date_df,
    convert_string_column_to_date,
    NamedTempFile,
    TemporaryDirectory,
)

# make some fake features data

states = [
    [0, '2016-01-01', False, True],
    [0, '2016-02-01', False, True],
    [0, '2016-03-01', False, True],
    [0, '2016-04-01', False, True],
    [0, '2016-05-01', False, True],
    [1, '2016-01-01', True, False],
    [1, '2016-02-01', True, False],
    [1, '2016-03-01', True, False],
    [1, '2016-04-01', True, False],
    [1, '2016-05-01', True, False],
    [2, '2016-01-01', True, False],
    [2, '2016-02-01', True, True],
    [2, '2016-03-01', True, False],
    [2, '2016-04-01', True, True],
    [2, '2016-05-01', True, False],
    [3, '2016-01-01', False, True],
    [3, '2016-02-01', True, True],
    [3, '2016-03-01', False, True],
    [3, '2016-04-01', True, True],
    [3, '2016-05-01', False, True],
    [4, '2016-01-01', True, True],
    [4, '2016-02-01', True, True],
    [4, '2016-03-01', True, True],
    [4, '2016-04-01', True, True],
    [4, '2016-05-01', True, True],
    [5, '2016-01-01', False, False],
    [5, '2016-02-01', False, False],
    [5, '2016-03-01', False, False],
    [5, '2016-04-01', False, False],
    [5, '2016-05-01', False, False]
]

features0_pre = [
    [0, '2016-01-01', 2, 1],
    [1, '2016-01-01', 1, 2],
    [0, '2016-02-01', 2, 3],
    [1, '2016-02-01', 2, 4],
    [0, '2016-03-01', 3, 3],
    [1, '2016-03-01', 3, 4],
    [0, '2016-04-01', 4, 3],
    [1, '2016-05-01', 5, 4]
]

features1_pre = [
    [2, '2016-01-01', 1, 1],
    [3, '2016-01-01', 1, 2],
    [2, '2016-02-01', 2, 3],
    [3, '2016-02-01', 2, 2],
    [0, '2016-03-01', 3, 3],
    [1, '2016-03-01', 3, 4],
    [2, '2016-03-01', 3, 3],
    [3, '2016-03-01', 3, 4],
    [3, '2016-03-01', 3, 4],
    [0, '2016-03-01', 3, 3],
    [4, '2016-03-01', 1, 4],
    [5, '2016-03-01', 2, 4]
]

# collate will ensure every entity/date combination in the state
# table have an imputed value in the features table, so ensure
# this is true for our test (filling with 9's):
f0_dict = {(r[0], r[1]) : r for r in features0_pre}
f1_dict = {(r[0], r[1]) : r for r in features1_pre}

for rec in states:
    ent_dt = (rec[0], rec[1])
    f0_dict[ent_dt] = f0_dict.get(ent_dt, list(ent_dt + (9, 9)))
    f1_dict[ent_dt] = f1_dict.get(ent_dt, list(ent_dt + (9, 9)))

features0 = sorted(f0_dict.values(), key=lambda x: (x[1], x[0]))
features1 = sorted(f1_dict.values(), key=lambda x: (x[1], x[0]))

features_tables = [features0, features1]

# make some fake labels data

labels = [
    [0, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [0, '2016-03-01', '1 month', 'booking', 'binary', 0],
    [0, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [0, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [0, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-05-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-01-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-03-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [1, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-05-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-01-01', '1 month', 'booking', 'binary', 0],
    [2, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [2, '2016-03-01', '1 month', 'booking', 'binary', 1],
    [2, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [2, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [2, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-05-01', '1 month', 'ems',     'binary', 1],
    [3, '2016-01-01', '1 month', 'booking', 'binary', 0],
    [3, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [3, '2016-03-01', '1 month', 'booking', 'binary', 1],
    [3, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [3, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [3, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [3, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [3, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [3, '2016-04-01', '1 month', 'ems',     'binary', 1],
    [3, '2016-05-01', '1 month', 'ems',     'binary', 0],
    [4, '2016-01-01', '1 month', 'booking', 'binary', 1],
    [4, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [4, '2016-03-01', '1 month', 'booking', 'binary', 0],
    [4, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [4, '2016-05-01', '1 month', 'booking', 'binary', 0],
    [4, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [4, '2016-02-01', '1 month', 'ems',     'binary', 1],
    [4, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [4, '2016-04-01', '1 month', 'ems',     'binary', 1],
    [4, '2016-05-01', '1 month', 'ems',     'binary', 1],
    [5, '2016-01-01', '1 month', 'booking', 'binary', 1],
    [5, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [5, '2016-03-01', '1 month', 'booking', 'binary', 0],
    [5, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [5, '2016-05-01', '1 month', 'booking', 'binary', 0],
    [5, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [5, '2016-02-01', '1 month', 'ems',     'binary', 1],
    [5, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [5, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [5, '2016-05-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [0, '2016-03-01', '3 month', 'booking', 'binary', 0],
    [0, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [0, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [0, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [0, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [0, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [0, '2016-04-01', '3 month', 'ems',     'binary', 0],
    [0, '2016-05-01', '3 month', 'ems',     'binary', 0],
    [1, '2016-01-01', '3 month', 'booking', 'binary', 0],
    [1, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [1, '2016-03-01', '3 month', 'booking', 'binary', 0],
    [1, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [1, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [1, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [1, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [1, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [1, '2016-04-01', '3 month', 'ems',     'binary', 0],
    [1, '2016-05-01', '3 month', 'ems',     'binary', 0],
    [2, '2016-01-01', '3 month', 'booking', 'binary', 0],
    [2, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [2, '2016-03-01', '3 month', 'booking', 'binary', 1],
    [2, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [2, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [2, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [2, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [2, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [2, '2016-04-01', '3 month', 'ems',     'binary', 0],
    [2, '2016-05-01', '3 month', 'ems',     'binary', 1],
    [3, '2016-01-01', '3 month', 'booking', 'binary', 0],
    [3, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [3, '2016-03-01', '3 month', 'booking', 'binary', 1],
    [3, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [3, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [3, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [3, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [3, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [3, '2016-04-01', '3 month', 'ems',     'binary', 1],
    [3, '2016-05-01', '3 month', 'ems',     'binary', 0],
    [3, '2016-05-01', '3 month', 'ems',     'binary', 0],
    [4, '2016-01-01', '3 month', 'booking', 'binary', 0],
    [4, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [4, '2016-03-01', '3 month', 'booking', 'binary', 1],
    [4, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [4, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [4, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [4, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [4, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [4, '2016-04-01', '3 month', 'ems',     'binary', 0],
    [4, '2016-05-01', '3 month', 'ems',     'binary', 1],
    [5, '2016-01-01', '3 month', 'booking', 'binary', 0],
    [5, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [5, '2016-03-01', '3 month', 'booking', 'binary', 1],
    [5, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [5, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [5, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [5, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [5, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [5, '2016-04-01', '3 month', 'ems',     'binary', 1],
    [5, '2016-05-01', '3 month', 'ems',     'binary', 0]
]

label_name = 'booking'
label_type = 'binary'

db_config = {
    'features_schema_name': 'features',
    'labels_schema_name': 'labels',
    'labels_table_name': 'labels',
    'sparse_state_table_name': 'staging.sparse_states',
}


def test_write_to_csv():
    """ Test the write_to_csv function by checking whether the csv contains the
    correct number of lines.
    """
    with testing.postgresql.Postgresql() as postgresql:
        # create an engine and generate a table with fake feature data
        engine = create_engine(postgresql.url())
        create_schemas(
            engine=engine,
            features_tables=features_tables,
            labels=labels,
            states=states
        )

        with TemporaryDirectory() as temp_dir:
            planner = Planner(
                feature_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                states = ['state_one AND state_two'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = engine,
                builder_class = builders.HighMemoryCSVBuilder
            )

            # for each table, check that corresponding csv has the correct # of rows
            for table in features_tables:
                planner.builder.write_to_csv(
                    '''
                        select *
                        from features.features{}
                    '''.format(features_tables.index(table)),
                    'test_csv.csv'
                )
                reader = csv.reader(planner.builder.open_fh_for_reading('test_csv.csv'))
                assert(len([row for row in reader]) == len(table) + 1)


def test_make_entity_date_table():
    """ Test that the make_entity_date_table function contains the correct
    values.
    """
    dates = [datetime.datetime(2016, 1, 1, 0, 0),
             datetime.datetime(2016, 2, 1, 0, 0),
             datetime.datetime(2016, 3, 1, 0, 0)]

    # make a dataframe of entity ids and dates to test against
    ids_dates = create_entity_date_df(
        labels=labels,
        states=states,
        as_of_dates=dates,
        state_one=True,
        state_two=True,
        label_name='booking',
        label_type='binary',
        label_timespan='1 month'
    )

    with testing.postgresql.Postgresql() as postgresql:
        # create an engine and generate a table with fake feature data
        engine = create_engine(postgresql.url())
        create_schemas(
            engine=engine,
            features_tables=features_tables,
            labels=labels,
            states=states
        )

        with TemporaryDirectory() as temp_dir:
            planner = Planner(
                feature_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                states = ['state_one AND state_two'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = engine
            )
            engine.execute(
                'CREATE TABLE features.tmp_entity_date (a int, b date);'
            )
            # call the function to test the creation of the table
            entity_date_table_name = planner.builder.make_entity_date_table(
                as_of_times=dates,
                label_type='binary',
                label_name='booking',
                state='state_one AND state_two',
                matrix_uuid='my_uuid',
                matrix_type='train',
                label_timespan='1 month'
            )

            # read in the table
            result = pd.read_sql(
                "select * from features.{} order by entity_id, as_of_date".format(entity_date_table_name),
                engine
            )
            labels_df = pd.read_sql('select * from labels.labels', engine)

            # compare the table to the test dataframe
            test = (result == ids_dates)
            assert(test.all().all())

def test_write_features_data():
    dates = [datetime.datetime(2016, 1, 1, 0, 0),
             datetime.datetime(2016, 2, 1, 0, 0)]

    # make dataframe for entity ids and dates
    ids_dates = create_entity_date_df(
        labels=labels,
        states=states,
        as_of_dates=dates,
        state_one=True,
        state_two=True,
        label_name='booking',
        label_type='binary',
        label_timespan='1 month'
    )

    features = [['f1', 'f2'], ['f3', 'f4']]
    # make dataframes of features to test against
    features_dfs = []
    for i, table in enumerate(features_tables):
        cols = ['entity_id', 'as_of_date'] + features[i]
        temp_df = pd.DataFrame(
            table,
            columns = cols
        )
        temp_df['as_of_date'] = convert_string_column_to_date(temp_df['as_of_date'])
        features_dfs.append(
            ids_dates.merge(
                right = temp_df,
                how = 'left',
                on = ['entity_id', 'as_of_date']
            )
        )

    # create an engine and generate a table with fake feature data
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_schemas(
            engine=engine,
            features_tables=features_tables,
            labels=labels,
            states=states
        )

        with TemporaryDirectory() as temp_dir:
            planner = Planner(
                feature_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                states = ['state_one AND state_two'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = engine,
            )

            # make the entity-date table
            entity_date_table_name = planner.builder.make_entity_date_table(
                as_of_times=dates,
                label_type='binary',
                label_name='booking',
                state = 'state_one AND state_two',
                matrix_type='train',
                matrix_uuid='my_uuid',
                label_timespan='1 month'
            )

            feature_dictionary = dict(
                ('features{}'.format(i), feature_list) for i, feature_list in enumerate(features)
            )

            features_csv_names = planner.builder.write_features_data(
                as_of_times=dates,
                feature_dictionary=feature_dictionary,
                entity_date_table_name=entity_date_table_name,
                matrix_uuid='my_uuid'
            )

            # get the queries and test them
            for feature_csv_name, df in zip(sorted(features_csv_names), features_dfs):
                df = df.reset_index()

                result = pd.read_csv(planner.builder.open_fh_for_reading(feature_csv_name))\
                    .reset_index()
                result['as_of_date'] = convert_string_column_to_date(result['as_of_date'])
                test = (result == df)
                assert(test.all().all())


def test_write_labels_data():
    """ Test the write_labels_data function by checking whether the query
    produces the correct labels
    """
    # set up labeling config variables
    dates = [datetime.datetime(2016, 1, 1, 0, 0),
             datetime.datetime(2016, 2, 1, 0, 0)]


    # make a dataframe of labels to test against
    labels_df = pd.DataFrame(
        labels,
        columns = [
            'entity_id',
            'as_of_date',
            'label_timespan',
            'label_name',
            'label_type',
            'label'
        ]
    )

    labels_df['as_of_date'] = convert_string_column_to_date(labels_df['as_of_date'])
    labels_df.set_index(['entity_id', 'as_of_date'])

    # create an engine and generate a table with fake feature data
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_schemas(
            engine,
            features_tables,
            labels,
            states
        )
        with TemporaryDirectory() as temp_dir:
            planner = Planner(
                feature_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                states = ['state_one AND state_two'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = engine,
            )

            # make the entity-date table
            entity_date_table_name = planner.builder.make_entity_date_table(
                as_of_times=dates,
                label_type='binary',
                label_name='booking',
                state = 'state_one AND state_two',
                matrix_type='train',
                matrix_uuid='my_uuid',
                label_timespan='1 month'
            )

            csv_filename = planner.builder.write_labels_data(
                label_name=label_name,
                label_type=label_type,
                label_timespan='1 month',
                matrix_uuid='my_uuid',
                entity_date_table_name=entity_date_table_name,
            )
            df = pd.DataFrame.from_dict({
                'entity_id': [2, 3, 4, 4],
                'as_of_date': ['2016-02-01', '2016-02-01', '2016-01-01', '2016-02-01'],
                'booking': [0, 0, 1, 0],
            }).set_index(['entity_id', 'as_of_date'])

            result = pd.read_csv(planner.builder.open_fh_for_reading(csv_filename))\
                .set_index(['entity_id', 'as_of_date'])
            test = (result == df)
            assert(test.all().all())


class TestMergeFeatureCSVs(TestCase):
    def test_badinput(self):
        """We assert column names, so replacing 'date' with 'as_of_date'
        should result in an error"""
        with TemporaryDirectory() as temp_dir:
            planner = Planner(
                feature_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                states = ['state_one AND state_two'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = None,
            )
            rowlists = [
                [
                    ('entity_id', 'date', 'f1'),
                    (1, 3, 3),
                    (4, 5, 6),
                    (7, 8, 9),
                ],
                [
                    ('entity_id', 'date', 'f2'),
                    (1, 2, 3),
                    (4, 5, 9),
                    (7, 8, 15),
                ],
                [
                    ('entity_id', 'date', 'f3'),
                    (1, 2, 2),
                    (4, 5, 20),
                    (7, 8, 56),
                ],
            ]

            filekeys = []
            for rows in rowlists:
                filekey = uuid.uuid4()
                planner.builder.open_fh_for_writing(filekey)
                filekeys.append(filekey)
                writer = csv.writer(planner.builder.filehandles[filekey])
                for row in rows:
                    writer.writerow(row)
            with self.assertRaises(KeyError):
                planner.builder.merge_feature_csvs(
                    filekeys,
                    matrix_directory=temp_dir,
                    matrix_uuid='1234'
                )

def test_generate_plans():
    matrix_set_definitions = [
        {
            'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
            'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
            'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
            'train_matrix': {
                'first_as_of_time': datetime.datetime(2010, 1, 1, 0, 0),
                'matrix_info_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 1, 0, 0),
                    datetime.datetime(2010, 1, 2, 0, 0),
                    datetime.datetime(2010, 1, 3, 0, 0),
                    datetime.datetime(2010, 1, 4, 0, 0),
                    datetime.datetime(2010, 1, 5, 0, 0)
                ]
            },
            'test_matrices': [{
                'first_as_of_time': datetime.datetime(2010, 1, 6, 0, 0),
                'matrix_info_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 6, 0, 0),
                    datetime.datetime(2010, 1, 7, 0, 0),
                    datetime.datetime(2010, 1, 8, 0, 0),
                    datetime.datetime(2010, 1, 9, 0, 0),
                    datetime.datetime(2010, 1, 10, 0, 0)
                ]
            }]
        },
        {
            'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
            'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
            'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
            'train_matrix': {
                'first_as_of_time': datetime.datetime(2010, 1, 6, 0, 0),
                'matrix_info_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 6, 0, 0),
                    datetime.datetime(2010, 1, 7, 0, 0),
                    datetime.datetime(2010, 1, 8, 0, 0),
                    datetime.datetime(2010, 1, 9, 0, 0),
                    datetime.datetime(2010, 1, 10, 0, 0)
                ]
            },
            'test_matrices': [{
                'first_as_of_time': datetime.datetime(2010, 1, 11, 0, 0),
                'matrix_info_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 11, 0, 0),
                    datetime.datetime(2010, 1, 12, 0, 0),
                    datetime.datetime(2010, 1, 13, 0, 0),
                    datetime.datetime(2010, 1, 14, 0, 0),
                    datetime.datetime(2010, 1, 15, 0, 0)
                ]
            }]
        }
    ]
    feature_dict_one = FeatureGroup(
        name='first_features',
        features_by_table={'features0': ['f1', 'f2'], 'features1': ['f1', 'f2']}
    )
    feature_dict_two = FeatureGroup(
        name='second_features',
        features_by_table={'features2': ['f3', 'f4'], 'features3': ['f5', 'f6']}
    )
    feature_dicts = [feature_dict_one, feature_dict_two]
    planner = Planner(
        feature_start_time = datetime.datetime(2010, 1, 1, 0, 0),
        label_names = ['booking'],
        label_types = ['binary'],
        cohort_name = 'prior_bookings',
        states = ['state_one AND state_two'],
        db_config = db_config,
        user_metadata = {},
        matrix_directory = '', # this test won't write anything
        engine = None # or look at the db!
    )

    updated_matrix_definitions, build_tasks = planner.generate_plans(matrix_set_definitions, feature_dicts)
    # test that it added uuids: we don't much care what they are
    matrix_uuids = []
    for matrix_def in updated_matrix_definitions:
        assert isinstance(matrix_def['train_uuid'], str)
        matrix_uuids.append(matrix_def['train_uuid'])
        for test_uuid in matrix_def['test_uuids']:
            assert isinstance(test_uuid, str)
    assert len(set(matrix_uuids)) == 4

    # not going to assert anything on the keys (uuids), just get out the values
    build_tasks = build_tasks.values()
    assert len(build_tasks) == 8 # 2 splits * 2 matrices per split * 2 feature dicts

    assert sum(1 for task in build_tasks if task['matrix_type'] == 'train') == 4
    assert sum(1 for task in build_tasks if task['matrix_type'] == 'test') == 4
    assert all(task for task in build_tasks if task['matrix_directory'] == '')
    assert sum(1 for task in build_tasks if task['feature_dictionary'] == feature_dict_one) == 4
    assert sum(1 for task in build_tasks if task['feature_dictionary'] == feature_dict_two) == 4
    assert sum(
        1 for task in build_tasks
        if task['matrix_metadata']['feature_groups'] == ['first_features']
    ) == 4
    assert sum(
        1 for task in build_tasks
        if task['matrix_metadata']['feature_groups'] == ['second_features']
    ) == 4
    assert sum(
        1 for task in build_tasks
        if task['matrix_metadata']['cohort_name'] == 'prior_bookings'
    ) == 8



class TestBuildMatrix(TestCase):
    def test_train_matrix(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states
            )

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                planner = Planner(
                    feature_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                    label_names = ['booking'],
                    label_types = ['binary'],
                    states = ['state_one AND state_two'],
                    db_config = db_config,
                    matrix_directory = temp_dir,
                    user_metadata = {},
                    engine = engine
                )
                feature_dictionary = FeatureGroup(
                    name='mygroup',
                    features_by_table={
                    'features0': ['f1', 'f2'],
                    'features1': ['f3', 'f4'],
                    }
                )
                matrix_metadata = {
                    'matrix_id': 'hi',
                    'state': 'state_one AND state_two',
                    'label_name': 'booking',
                    'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'feature_start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'label_timespan': '1 month'
                }
                uuid = metta.generate_uuid(matrix_metadata)
                planner.build_matrix(
                    as_of_times = dates,
                    label_name = 'booking',
                    label_type = 'binary',
                    feature_dictionary = feature_dictionary,
                    matrix_directory = temp_dir,
                    matrix_metadata = matrix_metadata,
                    matrix_uuid = uuid,
                    matrix_type = 'train'
                )

                matrix_filename = os.path.join(
                    temp_dir,
                    '{}.csv'.format(uuid)
                )
                with open(matrix_filename, 'r') as f:
                    reader = csv.reader(f)
                    assert(len([row for row in reader]) == 6)

    def test_test_matrix(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states
            )

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                planner = Planner(
                    feature_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                    label_names = ['booking'],
                    label_types = ['binary'],
                    states = ['state_one AND state_two'],
                    db_config = db_config,
                    matrix_directory = temp_dir,
                    user_metadata = {},
                    engine = engine
                )

                matrix_dates = {
                    'first_as_of_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'matrix_info_end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'as_of_times': dates
                }
                feature_dictionary = {
                    'features0': ['f1', 'f2'],
                    'features1': ['f3', 'f4'],
                }
                matrix_metadata = {
                    'matrix_id': 'hi',
                    'state': 'state_one AND state_two',
                    'label_name': 'booking',
                    'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'feature_start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'label_timespan': '1 month'
                }
                uuid = metta.generate_uuid(matrix_metadata)
                planner.build_matrix(
                    as_of_times = dates,
                    label_name = 'booking',
                    label_type = 'binary',
                    feature_dictionary = feature_dictionary,
                    matrix_directory = temp_dir,
                    matrix_metadata = matrix_metadata,
                    matrix_uuid = uuid,
                    matrix_type = 'test'
                )
                matrix_filename = os.path.join(
                    temp_dir,
                    '{}.csv'.format(uuid)
                )

                with open(matrix_filename, 'r') as f:
                    reader = csv.reader(f)
                    assert(len([row for row in reader]) == 6)

    def test_nullcheck(self):
        f0_dict = {(r[0], r[1]) : r for r in features0_pre}
        f1_dict = {(r[0], r[1]) : r for r in features1_pre}


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
                states=states
            )

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                planner = Planner(
                    feature_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                    label_names = ['booking'],
                    label_types = ['binary'],
                    states = ['state_one AND state_two'],
                    db_config = db_config,
                    matrix_directory = temp_dir,
                    user_metadata = {},
                    engine = engine
                )

                matrix_dates = {
                    'first_as_of_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'matrix_info_end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'as_of_times': dates
                }
                feature_dictionary = {
                    'features0': ['f1', 'f2'],
                    'features1': ['f3', 'f4'],
                }
                matrix_metadata = {
                    'matrix_id': 'hi',
                    'state': 'state_one AND state_two',
                    'label_name': 'booking',
                    'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'feature_start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'label_timespan': '1 month'
                }
                uuid = metta.generate_uuid(matrix_metadata)
                with self.assertRaises(ValueError):
                    planner.build_matrix(
                        as_of_times = dates,
                        label_name = 'booking',
                        label_type = 'binary',
                        feature_dictionary = feature_dictionary,
                        matrix_directory = temp_dir,
                        matrix_metadata = matrix_metadata,
                        matrix_uuid = uuid,
                        matrix_type = 'test'
                    )

    def test_replace(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states
            )

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                planner = Planner(
                    feature_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                    label_names = ['booking'],
                    label_types = ['binary'],
                    states = ['state_one AND state_two'],
                    db_config = db_config,
                    matrix_directory = temp_dir,
                    user_metadata = {},
                    engine = engine,
                    replace=False
                )

                matrix_dates = {
                    'first_as_of_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'matrix_info_end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'as_of_times': dates
                }
                feature_dictionary = {
                    'features0': ['f1', 'f2'],
                    'features1': ['f3', 'f4'],
                }
                matrix_metadata = {
                    'matrix_id': 'hi',
                    'state': 'state_one AND state_two',
                    'label_name': 'booking',
                    'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'feature_start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'label_timespan': '1 month'
                }
                uuid = metta.generate_uuid(matrix_metadata)
                planner.build_matrix(
                    as_of_times = dates,
                    label_name = 'booking',
                    label_type = 'binary',
                    feature_dictionary = feature_dictionary,
                    matrix_directory = temp_dir,
                    matrix_metadata = matrix_metadata,
                    matrix_uuid = uuid,
                    matrix_type = 'test'
                )

                matrix_filename = os.path.join(
                    temp_dir,
                    '{}.csv'.format(uuid)
                )

                with open(matrix_filename, 'r') as f:
                    reader = csv.reader(f)
                    assert(len([row for row in reader]) == 6)

                # rerun
                planner.builder.make_entity_date_table = Mock()
                planner.builder.build_matrix(
                    as_of_times = dates,
                    label_name = 'booking',
                    label_type = 'binary',
                    feature_dictionary = feature_dictionary,
                    matrix_directory = temp_dir,
                    matrix_metadata = matrix_metadata,
                    matrix_uuid = uuid,
                    matrix_type = 'test'
                )
                assert not planner.builder.make_entity_date_table.called
