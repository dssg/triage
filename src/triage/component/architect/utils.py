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
import subprocess
import gzip

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


def change_datetimes_on_metadata(metadata):
    for element in metadata.keys(): 
        if (element.endswith("_time")) or (element.endswith("_times")):
            if isinstance(metadata[element], list):
                metadata[element] = [str(ele) for ele in metadata[element]]
            else: 
                metadata[element] = str(metadata[element])
   
    return metadata


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


def _num_elements(x):
    """Extract the number of rows from the subprocess output"""
    return int(str(x.stdout, encoding="utf-8").split(" ")[0])


def check_rows_in_files(filenames, matrix_uuid):
    """Checks if the number of rows among all the CSV files for features and 
    and label for a matrix uuid are the same. 

    Args:
        filenames (List): List of CSV files to check the number of rows
        path_ (string): Path to get the temporal csv files
    """
    outputs = []
    for element in filenames:
        logging.debug(f"filename: {element}")
        just_filename = element.split("/")[-1]
        if (element.endswith(".csv")) and (just_filename.startswith(matrix_uuid)):
            cmd_line = "wc -l " + element
            outputs.append(subprocess.run(cmd_line, shell=True, capture_output=True))

    # get the number of rows from the subprocess
    rows = [_num_elements(output) for output in outputs]
    rows_set = set(rows)
    logging.debug(f"number of rows in files {rows_set}")

    if len(rows_set) == 1: 
        return True
    else:
        return False

def check_entity_ids_in_files(filenames, matrix_uuid):
    """Verifies if all the files in features and label have the same exact entity ids and knowledge dates"""
    # get first 2 columns on each file (entity_id, knowledge_date)
    for element in filenames: 
        logging.debug(f"getting entity id and knowledge date from features {element}")
        just_filename = element.split("/")[-1]
        prefix = element.split(".")[0]
        if (element.endswith(".csv")) and (just_filename.startswith(matrix_uuid)):
            cmd_line = f"cut -d ',' -f 1,2 {element} | sort -k 1,2 > {prefix}_sorted.csv"
            subprocess.run(cmd_line, shell=True)
    
    base_file = filenames[0]
    comparisons = []
    for i in range(1, len(filenames)):
        if (filenames[i].endswith(".csv")) and (filenames[i].startswith(matrix_uuid)):
            cmd_line = f"diff {base_file} {filenames[i]}"
            comparisons.append(subprocess.run(cmd_line, shell=True, capture_output=True))
    
    if len(comparisons) == 0:
        return True
    else:
        return False


def remove_entity_id_and_knowledge_dates(filenames, matrix_uuid):
    """drop entity id and knowledge date from all features and label files but one""" 
    correct_filenames = []

    for i in range(len(filenames)):
        just_filename = filenames[i].split("/")[-1]
        prefix = filenames[i].split(".")[0]
        if not (just_filename.endswith("_sorted.csv")) and (just_filename.startswith(matrix_uuid)):
            if prefix.endswith("_0"): 
                # only the first file will have entity_id and knowledge data but needs to also be sorted
                cmd_line = f"sort -k 1,2 {filenames[i]} > {prefix}_fixed.csv"
            else:
                cmd_line = f"sort -k 1,2 {filenames[i]} | cut -d ',' -f 3- > {prefix}_fixed.csv"
            subprocess.run(cmd_line, shell=True)
            # all files now the header in the last row (after being sorted)
            # from https://www.unix.com/shell-programming-and-scripting/128416-use-sed-move-last-line-top.html
            # move last line to first line
            cmd_line = f"sed -i '1h;1d;$!H;$!d;G' {prefix}_fixed.csv"
            subprocess.run(cmd_line, shell=True)
            correct_filenames.append(f"{prefix}_fixed.csv")

    return correct_filenames

    
def generate_list_of_files_to_remove(filenames, matrix_uuid):
    """Generate the list of all files that need to be removed"""
    # adding _sorted
    rm_files = []

    for element in filenames:
        rm_files.append(element)
        if (element.split("/")[-1].startswith(matrix_uuid)):
            prefix = element.split(".")[0]
            # adding sorted files 
            rm_files.append(prefix + "_sorted.csv")
            # adding fixed files
            rm_files.append(prefix + "_fixed.csv")

    logging.debug(f"Files to be removed {rm_files}")
    return rm_files


def generate_gzip(path, matrix_uuid):
    """
    Generates a gzip from the csv file with all the features (doesn't include the label)

    Args:
        path (string): _description_
        matrix_uuid (string): _description_
    """
    filename_ = f"{path}/{matrix_uuid}.csv"
    filepath_ = "/".join(filename_.split("/")[:-1])
    logger.debug(f"About to generate gzip for {matrix_uuid}.csv in {filepath_}")
    try:
        with open(filename_, 'rb') as f_in:
            with gzip.open(f"{filename_}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                logger.debug(f"gzip file {filename_}.gz generated")
    except FileNotFoundError as e:
        logger.error(f"File {filename_} not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred while generating gzip: {e}")


def remove_unnecessary_files(filenames, path_, matrix_uuid):
    """
    Removes the csvs generated for each feature, the label csv file,
    and the csv with all the features and label stitched togheter. 
    The csv with all merged is being deleted while generating the gzip.

    Args:
        filenames (list): list of filenames to remove from disk
        path_ (string): Path 
        matrix_uuid (string): ID of the matrix
    """
    # deleting features and label csvs
    for filename_ in filenames:
        cmd_line = ['rm', filename_] 
        logger.debug(f"removing files with command {cmd_line}")
        try:
            subprocess.run(cmd_line, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error removing file {filename_}: {e}")

    # deleting the merged csv
    cmd_line = ['rm', f'{path_}/{matrix_uuid}.csv']
    logger.debug(f"removing stitched csv with command {cmd_line}")
    try:
        subprocess.run(cmd_line, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error removing file {path_}/{matrix_uuid}.csv: {e}")
    
    # deleting the compressed CSV when the project path is S3
    if path_.startswith('/tmp'):
        filename_ = f"{path_}/{matrix_uuid}.csv.gz"
        cmd_line = ['rm', filename_]
        logger.debug(f"About to remove gzip file with command {cmd_line}")
        try:
            subprocess.run(cmd_line, check=True)
            logger.debug(f"gzip file {filename_} removed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error removing file {filename_}: {e}")
            