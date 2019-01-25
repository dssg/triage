import datetime

from triage.component.architect import Planner
from triage.component.architect.feature_group_creator import FeatureGroup


def test_Planner():
    matrix_set_definitions = [
        {
            "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
            "modeling_start_time": datetime.datetime(2010, 1, 1, 0, 0),
            "modeling_end_time": datetime.datetime(2010, 1, 16, 0, 0),
            "train_matrix": {
                "first_as_of_time": datetime.datetime(2010, 1, 1, 0, 0),
                "matrix_info_end_time": datetime.datetime(2010, 1, 6, 0, 0),
                "as_of_times": [
                    datetime.datetime(2010, 1, 1, 0, 0),
                    datetime.datetime(2010, 1, 2, 0, 0),
                    datetime.datetime(2010, 1, 3, 0, 0),
                    datetime.datetime(2010, 1, 4, 0, 0),
                    datetime.datetime(2010, 1, 5, 0, 0),
                ],
            },
            "test_matrices": [
                {
                    "first_as_of_time": datetime.datetime(2010, 1, 6, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 11, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0),
                    ],
                }
            ],
        },
        {
            "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
            "modeling_start_time": datetime.datetime(2010, 1, 1, 0, 0),
            "modeling_end_time": datetime.datetime(2010, 1, 16, 0, 0),
            "train_matrix": {
                "first_as_of_time": datetime.datetime(2010, 1, 6, 0, 0),
                "matrix_info_end_time": datetime.datetime(2010, 1, 11, 0, 0),
                "as_of_times": [
                    datetime.datetime(2010, 1, 6, 0, 0),
                    datetime.datetime(2010, 1, 7, 0, 0),
                    datetime.datetime(2010, 1, 8, 0, 0),
                    datetime.datetime(2010, 1, 9, 0, 0),
                    datetime.datetime(2010, 1, 10, 0, 0),
                ],
            },
            "test_matrices": [
                {
                    "first_as_of_time": datetime.datetime(2010, 1, 11, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0),
                        datetime.datetime(2010, 1, 15, 0, 0),
                    ],
                }
            ],
        },
    ]
    feature_dict_one = FeatureGroup(
        name="first_features",
        features_by_table={"features0": ["f1", "f2"], "features1": ["f1", "f2"]},
    )
    feature_dict_two = FeatureGroup(
        name="second_features",
        features_by_table={"features2": ["f3", "f4"], "features3": ["f5", "f6"]},
    )
    feature_dicts = [feature_dict_one, feature_dict_two]
    planner = Planner(
        feature_start_time=datetime.datetime(2010, 1, 1, 0, 0),
        label_names=["booking"],
        label_types=["binary"],
        cohort_names=["prior_bookings"],
        user_metadata={},
    )

    updated_matrix_definitions, build_tasks = planner.generate_plans(
        matrix_set_definitions, feature_dicts
    )
    # test that it added uuids: we don't much care what they are
    matrix_uuids = []
    for matrix_def in updated_matrix_definitions:
        assert isinstance(matrix_def["train_uuid"], str)
        matrix_uuids.append(matrix_def["train_uuid"])
        for test_uuid in matrix_def["test_uuids"]:
            assert isinstance(test_uuid, str)
    assert len(set(matrix_uuids)) == 4

    # not going to assert anything on the keys (uuids), just get out the values
    build_tasks = build_tasks.values()
    assert len(build_tasks) == 8  # 2 splits * 2 matrices per split * 2 feature dicts

    assert sum(1 for task in build_tasks if task["matrix_type"] == "train") == 4
    assert sum(1 for task in build_tasks if task["matrix_type"] == "test") == 4
    assert (
        sum(1 for task in build_tasks if task["feature_dictionary"] == feature_dict_one)
        == 4
    )
    assert (
        sum(1 for task in build_tasks if task["feature_dictionary"] == feature_dict_two)
        == 4
    )
    assert (
        sum(
            1
            for task in build_tasks
            if task["matrix_metadata"]["feature_groups"] == ["first_features"]
        )
        == 4
    )
    assert (
        sum(
            1
            for task in build_tasks
            if task["matrix_metadata"]["feature_groups"] == ["second_features"]
        )
        == 4
    )
    assert (
        sum(
            1
            for task in build_tasks
            if task["matrix_metadata"]["cohort_name"] == "prior_bookings"
        )
        == 8
    )
