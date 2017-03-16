import pandas as pd
import datetime
import sys
import tempfile


def convert_string_column_to_date(column):
    return(
        [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in column]
    )

def create_features_and_labels_schemas(engine, features_tables, labels):
    """ This function makes a features schema and populates it with the fake
    data from above.

    :param engine: a postgresql engine
    :type engine: sqlalchemy.engine
    """
    # create features schema and associated tables
    engine.execute('drop schema if exists features cascade; create schema features;')
    for table_number, table in enumerate(features_tables):
        create_features_table(table_number, table, engine)
    # create labels schema and table
    engine.execute('drop schema if exists labels cascade; create schema labels;')
    engine.execute(
        """
            create table labels.labels (
                entity_id int,
                as_of_date date,
                prediction_window interval,
                label_name char(30),
                label_type char(30),
                label int
            )
        """
    )
    for row in labels:
        engine.execute(
            'insert into labels.labels values (%s, %s, %s, %s, %s, %s)',
            row
        )

def create_features_table(table_number, table, engine):
    engine.execute(
        """
            create table features.features{} (
                entity_id int, as_of_date date, f1 int, f2 int
            )
        """.format(table_number)
    )
    for row in table:
        engine.execute(
            """
                insert into features.features{} values (%s, %s, %s, %s)
            """.format(table_number),
            row
        )

def create_entity_date_df(dates, features_tables):
    """ This function makes a pandas DataFrame that mimics the entity-date table
    for testing against.
    """
    dates = [date.date() for date in dates]

    # master dataframe to add other dfs to
    ids_dates = pd.DataFrame(
        [],
        columns = ['entity_id', 'as_of_date', 'f1', 'f2']
    )

    # convert each table to a dataframe and add it to master
    for table in features_tables:
        # temporary storage for the table
        temp_df = pd.DataFrame(
            table,
            columns = ['entity_id', 'as_of_date', 'f1', 'f2']
        )
        # add temporary table to master
        ids_dates = pd.concat([ids_dates, temp_df])

    # select only the relevant columns, drop duplicates, sort, and convert dates
    ids_dates = ids_dates[['entity_id', 'as_of_date']]
    ids_dates = ids_dates.drop_duplicates()
    ids_dates = ids_dates.sort_values(['entity_id', 'as_of_date'])
    ids_dates['as_of_date'] = [datetime.datetime.strptime(
        date,
        '%Y-%m-%d'
    ).date() for date in ids_dates['as_of_date']]

    ids_dates = ids_dates[ids_dates['as_of_date'].isin(dates)]
    print(ids_dates)
    print(dates)

    return(ids_dates.reset_index(drop = True))


def NamedTempFile():
    if sys.version_info >= (3,0,0):
        return tempfile.NamedTemporaryFile(mode='w+', newline='')
    else:
        return tempfile.NamedTemporaryFile()
