from functools import partial
import pandas as pd
import numpy as np

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

def downcast_matrix(df):
    """Downcast the numeric values of a matrix.

    This will make the matrix use less memory by turning, every number into
    float32. It's more expensive in time to try to convert int64 into int32 
    than just convert the whole matrix in float32, which still is less memory
    intensive than the original matrix. 

    Operates on the dataframe as passed, without doing anything to the index.
    Callers may pass an index-less dataframe if they wish to re-add the index afterwards
    and save memory on the index storage.
    """
    logger.spam("Downcasting matrix.")
    logger.spam(f"Starting memory usage: {df.memory_usage(deep=True).sum()/1000000} MB")
    logger.spam(f"Initial types: \n {df.dtypes}")

    df = df.apply(lambda x: x.astype('float32'))
    
    logger.spam("Downcasting matrix completed.")
    logger.spam(f"Final memory usage: {df.memory_usage(deep=True).sum()/1000000} MB")
    logger.spam(f"Final data types: \n {df.dtypes}")

    return df
