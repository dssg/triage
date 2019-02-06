"""Metta IO

Library for storing train-test sets.

"""
import copy
import yaml
import os
import datetime
import json
import hashlib
import shutil
import pandas as pd
import numpy as np

import logging

import s3fs

from urllib.parse import urlparse


def archive_train_test(
    train_config,
    df_train,
    test_config,
    df_test,
    directory=".",
    format="hd5",
    overwrite=False,
):
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
        train_config, df_train, overwrite=overwrite, directory=directory, format=format
    )

    test_uuid = archive_matrix(
        test_config,
        df_test,
        overwrite=overwrite,
        directory=directory,
        format=format,
        train_uuid=train_uuid,
    )

    return train_uuid, test_uuid


def archive_matrix(
    matrix_config,
    df_matrix,
    overwrite=False,
    directory=".",
    format="hd5",
    train_uuid=None,
):
    """Store a design matrix.

    Parameters
    ----------
    matrix_config: dict
        dict to be yamled
    df_matrix: DataFrame or str
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
        if not (os.path.isfile(abs_path_file)):
            raise IOError("Not a file: {}".format(abs_path_file))

        if abs_path_file[-3:] == "csv":
            pass
        elif abs_path_file[-2:] == "h5":
            df_matrix = pd.read_hdf(abs_path_file)
        else:
            raise IOError("Not a csv or h5 file: {}".format(abs_path_file))
    else:
        raise IOError("Not a dataframe or path to a dataframe: {}".type(df_matrix))

    check_config_types(matrix_config)

    matrix_uuid = generate_uuid(matrix_config)

    matrix_config = copy.deepcopy(matrix_config)
    matrix_config["metta-uuid"] = matrix_uuid

    fname = os.path.join(directory, matrix_uuid)

    # The output directory is local or in s3
    path_parsed = urlparse(directory)
    scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'

    if scheme in ("", "file"):  # Local file
        abs_path_dir = os.path.abspath(directory)
        if not os.path.exists(abs_path_dir):
            os.makedirs(abs_path_dir)
        write_matrix = (overwrite) or not (os.path.exists(fname + format))
    elif scheme == "s3":
        abs_path_dir = directory
        write_matrix = (overwrite) or not (s3fs.S3FileSystem().exists(fname + format))
    else:
        raise ValueError(
            f"""URL scheme not supported:
              {scheme} (from {fname})
        """
        )

    if write_matrix:
        _store_matrix(
            matrix_config, df_matrix, matrix_uuid, abs_path_dir, format=format
        )

    return matrix_uuid


def _store_matrix(metadata, df_data, title, directory, format="hd5"):
    """Store matrix and associated meta-data

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
    if isinstance(df_data, pd.DataFrame):
        last_col = df_data.columns.tolist()[-1]
    elif type(df_data) == str:
        abs_path_file = os.path.abspath(df_data)
        if not (os.path.isfile(abs_path_file)):
            raise IOError("Not a file: {}".format(abs_path_file))
        if abs_path_file[-3:] == "csv":
            headers = pd.read_csv(abs_path_file, nrows=1)
            last_col = headers.columns.tolist()[-1]

    if not (metadata["label_name"] == last_col):
        raise IOError("label_name is not last column")

    # The output directory is local or in s3
    path_parsed = urlparse(directory)
    scheme = path_parsed.scheme  # If '' of 'file' is a regular file or 's3'

    yaml_fname = os.path.join(directory, title + ".yaml")
    matrix_fname = os.path.join(directory, title + f".{format}")

    if scheme in ("", "file"):  # Local file
        with open(yaml_fname, "w") as stream:
            yaml.dump(metadata, stream)

        if format == "hd5":  # H5 only supported locally
            matrix_fname = os.path.join(directory, title + ".h5")
            for col in df_data.columns:
                if df_data[col].dtype == np.dtype("datetime64[ns]"):
                    df_data[col] = df_data[col].map(lambda x: x.timestamp())
                elif isinstance(df_data[col].dtype, object):
                    df_data[col] = df_data[col].astype(float)

            hdf = pd.HDFStore(
                matrix_fname, mode="w", complevel=5, complib="zlib",
            )
            hdf.put(title, df_data, data_columns=True)
            hdf.close()
        elif format == "csv":
            if isinstance(df_data, pd.DataFrame):
                if df_data.index.name:
                    df_data.to_csv(matrix_fname, index=False)
                else:
                    df_data.to_csv(matrix_fname)
            elif type(df_data) == str:
                if os.path.splitext(abs_path_file)[1] == "csv":
                    shutil.copyfile(abs_path_file, matrix_fname)
        else:
            raise ValueError(
                f"""
                  File format not supported:
                  {format} (from {yaml_fname})
            """
            )
    elif scheme == "s3":
        s3 = s3fs.S3FileSystem()
        with s3.open(yaml_fname, "wb") as stream:
            yaml.dump(metadata, stream, encoding="utf-8")

        if isinstance(df_data, pd.DataFrame):
            logging.debug(f"Data frame received: {df_data.shape}")
            logging.debug(
                f"Data frame size in memory: "
                + f"{df_data.memory_usage(index=True, deep=True).sum()} bytes"
            )

            if df_data.index.name:
                bytes_to_write = df_data.to_csv(None, index=False).encode()
            else:
                bytes_to_write = df_data.to_csv(None).encode()
            with s3.open(matrix_fname, "wb") as stream:
                stream.write(bytes_to_write)

            logging.debug(f"Data frame stored in {matrix_fname}")
        elif type(df_data) == str:
            logging.debug(f"Local file received: {df_data}")
            s3.put(df_data, matrix_fname)
            logging.debug(f"Local file stored in {matrix_fname}")
        else:
            raise ValueError(
                f"""
                  Type not supported:
                  {type(df_data)}
            """
            )

    else:
        raise ValueError(
            f"""
              URL scheme not supported:
              {scheme} (from {yaml_fname})
        """
        )


def check_config_types(dict_config):
    """
    Enforce Datatypes in the dictionary


    Parameters
    ----------
    dict_config:
       Check the configuration types.
    The required types are:

    - feature_start_time: datetime.datetime
    - end_time: datetime.datetime
    - label_timespan: str
    - label_name: str
    - matrix_id: human readable name for the data

    Raises
    ------
    IOError:
        missing required keys

    """
    set_required_names = set(
        ["feature_start_time", "end_time", "label_timespan", "label_name", "matrix_id"]
    )

    if not (set_required_names.issubset(dict_config.keys())):
        raise IOError("missing required keys in dictionary", set_required_names)

    # check that the start time and end times are correct
    assert isinstance(dict_config["feature_start_time"], datetime.date)
    assert isinstance(dict_config["end_time"], datetime.date)
    assert isinstance(dict_config["label_timespan"], str)
    assert isinstance(dict_config["matrix_id"], str)


def generate_uuid(metadata):
    """ Generate a unique identifier given a dictionary of matrix metadata.

    :param metadata: metadata for the matrix
    :type metadata: dict
    :returns: unique name for the file
    :rtype: str
    """

    def dt_handler(x):
        if isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
            return x.isoformat()
        raise TypeError("Unknown type")

    return hashlib.md5(
        json.dumps(metadata, default=dt_handler, sort_keys=True).encode("utf-8")
    ).hexdigest()


def recover_matrix(config, directory="."):
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

    fname = directory + "/" + uuid

    if os.path.isfile(fname + ".h5"):
        df_matrix = pd.read_hdf(fname + ".h5")
        return df_matrix
    elif os.path.isfile(fname + ".csv"):
        df_matrix = pd.read_csv(fname + ".csv")
        return df_matrix
    else:
        return None
