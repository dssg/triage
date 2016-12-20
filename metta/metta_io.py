"""
Metta IO

Library for storing train-test sets.
"""
import yaml
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def archive_train_test(train_config, df_train, title_train,
                       test_config, df_test, title_test,
                       directory='.', format='hd5'):
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


    Returns
    -------
    train_yaml: file
        Writes out to YAML file title.yaml
    data_file: file
        CSV of dataframe feature set title.csv
    """

    abs_path_dir = os.path.abspath(directory)

    _store_matrix(train_config, df_train, title_train, abs_path_dir)
    _store_matrix(test_config, df_test, title_test, abs_path_dir)

    with open(abs_path_dir + '/' + 'matrix_pairs.txt', 'a') as outfile:
        outfile.write(','.join([title_train, title_test]) + '\n')


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
        - hd5: HDF5
        - csv: Comma Separated Values

    Returns
    -------
    metadata: file
        Writes out to YAML file title.yaml
    data_file: file
        CSV of dataframe feature set title.csv
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    # dump the config file into a yaml format
    with open(directory + '/' + title + '.yaml', 'w') as stream:
        yaml.dump(metadata, stream)

    if format == 'hd5':
        hdf = pd.HDFStore(directory + '/' + title + '.h5')
        hdf.put(title, df_data, data_columns=True)
    elif format == 'csv':
        df_data.to_csv(directory + '/' + title + '.csv')
