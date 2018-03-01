import pdb
import copy
import threading
from itertools import product
import datetime
import logging

from . import setup_environment

log = logging.getLogger(__name__)


def create_labels_table(config, table_name):
    """Build the features table for the type of model (officer/dispatch) specified in the config file"""

    engine = setup_environment.get_database()
    if config['unit'] == 'officer':
        create_officer_labels_table(config, table_name, engine)
    # TODO:
    #if config['unit'] == 'dispatch':
    #    create_dispatch_labels_table(config, table_name)


def populate_labels_table(config, labels_config, table_name):
    """Calculate values for all features which are set to True (in the config file)
    for the appropriate run type (officer/dispatch)
    """
    engine = setup_environment.get_database()
    if config['unit'] == 'officer':
        populate_officer_labels_table(config, labels_config, table_name, engine)

def create_officer_labels_table(config, table_name, engine):
    """ Creates a features.table_name table within the features schema """


    # drop the old features table
    log.info("Dropping the old officer labels table: {}".format(table_name))
    engine.execute("DROP TABLE IF EXISTS features.{}".format(table_name) )

    # use the appropriate id column, depending on feature types (officer / dispatch)
    id_column = '{}_id'.format(config['unit'])

    # Create and execute a query to create a table with a column for each of the labels.
    log.info("Creating new officer feature table: {}...".format(table_name))
    create_query = (    "CREATE TABLE features.{} ( "
                        "   {}                   int, "
                        "   event_id             int, "
                        "   event_datetime       timestamp, "
                        "   event_type           text, "
                        "   value                text);"
                        .format(
                            table_name,
                            id_column))

    engine.execute(create_query)
    query_index = ("CREATE INDEX ON features.{} (event_datetime, officer_id)".format(table_name))
    query_index = ("CREATE INDEX ON features.{} (event_id)".format(table_name))
    engine.execute(query_index)

def column_date(nested_dict, dict_columns=dict()):
    temp_dict= {}
    if isinstance(nested_dict, dict):
        temp_dict[nested_dict['COLUMN']] = nested_dict['DATE_COLUMN']
        dict_columns.update(temp_dict)
        for val in nested_dict['VALUES']:
            if isinstance(val, dict):
                for key in val.keys():
                    column_date(val[key], dict_columns)
    return dict_columns

def populate_officer_labels_table(config, labels_config, table_name, engine):
    """ Populates officer labels table in the database using staging.incidents.
     """

    dict_columns = dict()
    for labels in labels_config.keys():
        dict_columns.update(column_date(labels_config[labels], dict_columns))

    query_list = []
    for column, date_column in dict_columns.items():
        query_list.append("SELECT officer_id, "
                          "       event_id, "
                          "       {event_datetime} as event_datetime, "
                          "       '{event_type}' as event_type, "
                          "       {event_type}::TEXT as value "
                          "    FROM staging.incidents "
                          "    WHERE {event_type}::TEXT is not NULL "
                          "    AND {event_datetime} is not NULL "
                          "    AND officer_id is not NULL "
                      .format(event_datetime=date_column,
                              event_type=column))

    query_join = " UNION ".join(query_list)
    insert_query = ( "INSERT INTO features.{0}  "
                     "         ( officer_id, "
                     "           event_id, "
                     "           event_datetime, "
                     "           event_type, "
                     "           value ) "
                     "         {1}  "
                     .format(table_name, query_join))

    engine.execute(insert_query)          
    
    # Create indexes
    create_event_id_idx = (""" Create index on features.{0} (officer_id, event_id); """.format(table_name))
    engine.execute(create_event_id_idx)
 
