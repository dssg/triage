from functools import partial
import pandas as pd
import numpy as np

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

def downcast_matrix(df):
    """Downcast the numeric values of a matrix.

    This will make the matrix use less memory by turning, for instance,
    int64 columns into int32 columns.

    First converts floats and then integers.

    Operates on the dataframe as passed, without doing anything to the index.
    Callers may pass an index-less dataframe if they wish to re-add the index afterwards
    and save memory on the index storage.
    """
    logger.spam("Downcasting matrix.")
    logger.spam(f"Starting memory usage: {df.memory_usage(deep=True).sum()} bytes")
    logger.spam(f"Initial types: \n {df.dtypes}")
    new_df = df.apply(lambda x: x.astype(np.float32))

    logger.spam("Downcasting matrix completed.")
    logger.spam(f"Final memory usage: {new_df.memory_usage(deep=True).sum()} bytes")
    logger.spam(f"Final data types: \n {new_df.dtypes}")

    return new_df
