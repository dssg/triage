import pandas as pd
import datetime

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

def create_entity_date_df(dates, labels, as_of_dates, label_name,
                          label_type):
    """ This function makes a pandas DataFrame that mimics the entity-date table
    for testing against.
    """
    0, '2016-02-01', '1 month', 'booking', 'binary', 0
    labels_table = pd.DataFrame(labels, columns = [
        'entity_id',
        'as_of_date',
        'prediction_window',
        'label_name',
        'label_type',
        'label'
    ])
    dates = [date.date() for date in dates]
    labels_table = labels_table[labels_table['label_name'] == label_name]
    labels_table = labels_table[labels_table['label_type'] == label_type]
    ids_dates = labels_table[['entity_id', 'as_of_date']]
    ids_dates = ids_dates.sort_values(['entity_id', 'as_of_date'])
    ids_dates['as_of_date'] = [datetime.datetime.strptime(
        date,
        '%Y-%m-%d'
    ).date() for date in ids_dates['as_of_date']]
    ids_dates = ids_dates[ids_dates['as_of_date'].isin(dates)]
    print(ids_dates)
    print(dates)

    return(ids_dates.reset_index(drop = True))