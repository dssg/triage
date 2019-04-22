from functools import partial
import pandas as pd
import numpy as np
import logging


def downcast_matrix(df):
    """Downcast the numeric values of a matrix.

    This will make the matrix use less memory by turning, for instance,
    int64 columns into int32 columns.

    First converts floats and then integers.

    Operates on the dataframe as passed, without doing anything to the index.
    Callers may pass an index-less dataframe if they wish to re-add the index afterwards
    and save memory on the index storage.
    """
    logging.debug("Downcasting matrix. Starting memory usage: %s", df.memory_usage())

    logging.debug(df.dtypes)

    new_df = df.apply(lambda x: x.astype(np.float32))

    logging.debug(new_df.dtypes)

    logging.debug("Downcasted matrix. Final memory usage: %s", new_df.memory_usage())

    return new_df
