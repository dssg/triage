"""
Metta IO

Library for storing train-test sets.
"""
import copy
import yaml
import os
import pandas as pd
import warnings
import datetime
import uuid


warnings.filterwarnings("ignore")


def archive_train_test(train_config,
                       df_train,
                       test_config,
                       df_test,
                       directory='.',
                       format='hd5',
                       overwrite=False):
    """
    Main function for archiving train and test sets

    Parameters
    ----------
    (train_config, test_config): dict
        dicts to be yamled for train and test
    (df_train, df_test): df
        DataFrame of features and label (as last
        column) for training and testing set
    (title_train, title_test): str
        Unique name for train and test sets
    directory: str
        Relative path to where the data will be stored
    format: str
        format to save files in
        - hd5: HDF5
        - csv: Comma Separated Values
    overwrite: bool
        If true then identical matrices
        will be overridden.


    Returns
    -------
    (train_uuid, test_uuid): 'str'
       uuids for training and test set
    train_yaml: file
        Writes out to YAML file title.yaml
    data_file: file
        CSV of dataframe feature set title.csv
    """

    train_uuid = archive_matrix(
        train_config,
        df_train,
        overwrite=overwrite,
        directory=directory,
        format=format
    )

    test_uuid = archive_matrix(
        test_config,
        df_test,
        overwrite=overwrite,
        directory=directory,
        format=format,
        train_uuid=train_uuid
    )

    return train_uuid, test_uuid


def archive_matrix(
        matrix_config,
        df_matrix,
        overwrite=False,
        directory='.',
        format='hd5',
        train_uuid=None,
):
    """Store a design matrix.

    Parameters
    ----------
    matrix_config: dict
        dict to be yamled
    df_matrix: df
        DataFrame of features and label (as last column) or path to a CSV
    overwrite: bool
        If true will overwrite the same prexisting matrix.
    directory: str
        Relative path to where the data will be stored
    format: str
        format to save files in
        - hd5: HDF5
        - csv: Comma Separated Values
    train_uuid (optional): uuid of train set to associate with as a test set

    Returns
    -------
    uuid: str
        uuid for the stored set
    """

    if isinstance(df_matrix, pd.DataFrame):
        pass
    elif type(df_matrix) == str:
        abs_path_file = os.path.abspath(df_matrix)
        if not(os.path.isfile(abs_path_file)):
            raise IOError('Not a file: {}'.format(abs_path_file))

        if abs_path_file[-3:] == 'csv':
            df_matrix = pd.read_csv(abs_path_file)
        elif abs_path_file[-2:] == 'h5':
            df_matrix = pd.read_hdf(abs_path_file)
        else:
            raise IOError('Not a csv or h5 file: {}'.format(abs_path_file))
    else:
        raise IOError(
            'Not a dataframe of path to a dataframe: {}'.type(df_matrix))

    abs_path_dir = os.path.abspath(directory)
    if not os.path.exists(abs_path_dir):
        os.makedirs(abs_path_dir)

    uuid_fname = directory + '/' + '.matrix_uuids'
    set_uuids = load_uuids(uuid_fname)

    check_config_types(matrix_config)

    matrix_uuid = generate_uuid(matrix_config)

    matrix_config = copy.deepcopy(matrix_config)
    matrix_config['metta-uuid'] = matrix_uuid

    write_matrix = (overwrite) or (not(matrix_uuid in set_uuids))
    print('OVERWRITE', overwrite)

    if write_matrix:
        print('writing the matrix')
        _store_matrix(matrix_config, df_matrix, matrix_uuid, abs_path_dir,
                      format=format)

    if train_uuid:
        with open(abs_path_dir + '/' + 'matrix_pairs.txt', 'a') as outfile:
            outfile.write(','.join([train_uuid, matrix_uuid]) + '\n')

    return matrix_uuid


def _store_matrix(metadata, df_data, title, directory, format='hd5'):
    """
    Store matrix and associated meta-data


    Parameters
    ----------
    metadata: dict
        dictionary of config/meta data
    data_df: df
        df of data with the last column being the labels
    title: str
        unique name of dataset
    directory: str
        Relative path to where the data will be stored
    format: str
        format to save files in
        - hd5: HDF5 (default) compressoin level 5, complib zlib
        - csv: Comma Separated Values

    Returns
    -------
    metadata: file
        Writes out to YAML file title.yaml
    data_file: file
        CSV of dataframe feature set title.csv

    Raises
    ------
    IOError:
        label name is not the last column name
    """

    # check last column is the label
    last_col = df_data.columns.tolist()[-1]

    if not(metadata['label_name'] == last_col):
        raise IOError('label_name is not last column')

    yaml_fname = directory + '/' + title + '.yaml'

    with open(yaml_fname, 'w') as stream:
        yaml.dump(metadata, stream)

    if format == 'hd5':
        print(directory + '/' + title + '.h5')
        hdf = pd.HDFStore(directory + '/' + title + '.h5',
                          mode='w',
                          complevel=5,
                          complib="zlib")
        hdf.put(title, df_data, data_columns=True)
        hdf.close()
    elif format == 'csv':
        df_data.to_csv(directory + '/' + title + '.csv')

    with open(directory + '/' + '.matrix_uuids', 'a') as uuid_file:
        uuid_file.write(title + '\n')


def check_config_types(dict_config):
    """
    Enforce Datatypes in the dictionary


    Parameters
    ----------
    dict_config:
       Check the configuration types.
    The required types are:

    - start_time: datetime.datetime
    - end_time: datetime.datetime
    - prediction_window: int
    - label_name: str
    - matrix_id: human readable name for the data

    Raises
    ------
    IOError:
        missing required keys

    """
    set_required_names = set(
        ['start_time', 'end_time', 'prediction_window', 'label_name',
         'matrix_id'])

    if not(set_required_names.issubset(dict_config.keys())):
        raise IOError('missing required keys in dictionary',
                      set_required_names)

    # check that the start time and end times are correct
    assert isinstance(dict_config['start_time'], datetime.date)
    assert isinstance(dict_config['end_time'], datetime.date)
    assert isinstance(dict_config['prediction_window'], str)
    assert isinstance(dict_config['matrix_id'], str)


def load_uuids(uuid_fname):
    """
    Return a list of existing uuids.

    Parameters
    ---------
    directory: str
        path to working directory

    Returns
    -------
    uuids: set
       set of uuids. empty set of there is
       no .matrix_uuids file

    """

    if os.path.isfile(uuid_fname):
        with open(uuid_fname, 'r') as uuid_file:
            uuids = set([str_uuid.strip('\n') for str_uuid in uuid_file])
        return uuids
    else:
        return set([])


def generate_uuid(metadata):
    """ Generate a unique identifier given a dictionary of matrix metadata.

    :param metadata: metadata for the matrix
    :type metadata: dict
    :returns: unique name for the file
    :rtype: str
    """
    identifier = ''
    for key in sorted(metadata.keys()):
        identifier = '{0}_{1}'.format(identifier, str(metadata[key]))
    name_uuid = str(uuid.uuid3(uuid.NAMESPACE_DNS, identifier))
    return name_uuid


def recover_matrix(config, directory='.'):
    """Recover a matrix by either its config or uuid.

    Parameters
    ----------
    config: str or dict
        config metadata for the matrix or uuid
    directory: str
        path to search for the matrix

    Returns
    -------
    df_matrix: DataFrame
        DataFrame of specified matrix
    None:
        If no matrix matrix is found
    """

    if isinstance(config, dict):
        uuid = generate_uuid(config)
    else:
        uuid = config

    uuid_fname = directory + '/' + '.matrix_uuids'
    set_uuids = load_uuids(uuid_fname)

    if uuid in set_uuids:
        fname = directory + '/' + uuid + '.h5'
        df_matrix = pd.read_hdf(fname)
        return df_matrix
    else:
        return None
