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
    logger.spam(f"Starting memory usage: {df.memory_usage(deep=True).sum()/1000000} MB")
    logger.spam(f"Initial types: \n {df.dtypes}")
    logger.spam(f"Changing int64 to int32 (if any)")
    if df.select_dtypes("int64").shape[1] > 0: 
        new_df_ints = df.select_dtypes("int64").apply(lambda x: x.astype(np.int32))
    logger.spam("Changin float64 to float32 (if any)")
    if df.select_dtypes("float64").shape[1] > 0: 
        new_df_floats = df.select_dtypes("float64").apply(lambda x: x.astype(np.float32))
    
    new_df = pd.concat([new_df_ints, new_df_floats], axis=1)

    logger.spam("Downcasting matrix completed.")
    logger.spam(f"Final memory usage: {new_df.memory_usage(deep=True).sum()/1000000} MB")
    logger.spam(f"Final data types: \n {new_df.dtypes}")

    # explicitly delete the previous df to reduce use of memory
    del(df)

    return new_df
