"""
Metta IO

Library for storing train-test sets.
"""
import yaml


def archive_train_test(train_config, df_train, title_train,
                       test_config, df_test, title_test):
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


    Returns
    -------
    train_yaml: file
        Writes out to YAML file title.yaml
    data_file: file
        CSV of dataframe feature set title.csv
    """

    _store_matrix(train_config, df_train, title_train)
    _store_matrix(test_config, df_test, title_test)

    with open('matrix_pairs.txt', 'a') as outfile:
        outfile.write(','.join([title_train, title_test]) + '\n')


def _store_matrix(metadata, df_data, title):
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


    Returns
    -------
    metadata: file
        Writes out to YAML file title.yaml
    data_file: file
        CSV of dataframe feature set title.csv
    """

    # dump the config file into a yaml format
    with open(title + '.yaml', 'w') as stream:
        yaml.dump(metadata, stream)

    # dump data into a csv
    df_data.to_csv(title + '.csv')
