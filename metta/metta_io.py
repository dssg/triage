"""
Interface

Interface to train test matrix
"""
import pandas as pd
import yaml

# Create a HDF5 container to load YAML
# and train and test sets.


def store_matrix(metadata, df_data, title):
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

    # dump the file into a yaml format
    with file(title + '.yaml', 'w') as stream:
        yaml.dump(metadata, stream)

    # dump data into a csv
    df_data.to_csv(title + '.csv')
