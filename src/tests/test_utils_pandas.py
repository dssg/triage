from triage.component.catwalk.storage import MatrixStore
import pandas as pd
from triage.util.pandas import columns_with_nulls, downcast_matrix
from .utils import matrix_creator


def test_downcast_matrix():
    df = matrix_creator().set_index(MatrixStore.indices)
    downcasted_df = downcast_matrix(df)

    # make sure the contents are equivalent
    assert((downcasted_df == df).all().all())

    # make sure the memory usage is lower because there would be no point of this otherwise
    assert downcasted_df.memory_usage().sum() < df.memory_usage().sum()


def test_columns_with_nulls():
    assert columns_with_nulls(pd.DataFrame.from_dict({
        "feature_one": [0.5, 0.6, 0.5, 0.6],
        "feature_two": [0.5, 0.6, 0.5, 0.6],
    })) == []

    assert columns_with_nulls(pd.DataFrame.from_dict({
        "feature_one": [0.5, None, 0.5, 0.6],
        "feature_two": [0.5, 0.6, 0.5, 0.6],
    })) == ["feature_one"]
