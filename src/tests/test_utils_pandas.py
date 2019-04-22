from triage.util.pandas import downcast_matrix
from triage.component.catwalk.storage import MatrixStore
from .utils import matrix_creator


def test_downcast_matrix():
    df = matrix_creator().set_index(MatrixStore.indices)
    downcasted_df = downcast_matrix(df)

    # make sure the contents are equivalent
    assert((downcasted_df == df).all().all())

    # make sure the memory usage is lower because there would be no point of this otherwise
    assert downcasted_df.memory_usage().sum() < df.memory_usage().sum()
