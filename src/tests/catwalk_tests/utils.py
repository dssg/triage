import datetime
import pytest

import pandas

from triage.util.structs import FeatureNameList


@pytest.fixture
def sample_metadata():
    return {
        "feature_start_time": datetime.date(2012, 12, 20),
        "end_time": datetime.date(2016, 12, 20),
        "label_name": "label",
        "as_of_date_frequency": "1w",
        "max_training_history": "5y",
        "state": "default",
        "cohort_name": "default",
        "label_timespan": "1y",
        "metta-uuid": "1234",
        "feature_names": FeatureNameList(["ft1", "ft2"]),
        "feature_groups": ["all: True"],
        "indices": ["entity_id"],
    }


@pytest.fixture
def sample_df():
    return pandas.DataFrame.from_dict(
        {
            "entity_id": [1, 2],
            "feature_one": [3, 4],
            "feature_two": [5, 6],
            "label": ["good", "bad"],
        }
    ).set_index("entity_id")
