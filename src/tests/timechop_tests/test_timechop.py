import datetime
from unittest import TestCase

from triage.util.conf import convert_str_to_relativedelta

from triage.component.timechop import Timechop


class test_calculate_train_test_split_times(TestCase):
    def test_valid_input(self):
        expected_result = [
            datetime.datetime(2015, 3, 1, 0, 0),
            datetime.datetime(2015, 6, 1, 0, 0),
            datetime.datetime(2015, 9, 1, 0, 0),
            datetime.datetime(2015, 12, 1, 0, 0),
            datetime.datetime(2016, 3, 1, 0, 0),
            datetime.datetime(2016, 6, 1, 0, 0),
        ]
        chopper = Timechop(
            feature_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2017, 1, 1, 0, 0),
            label_start_time=datetime.datetime(2015, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2017, 1, 1, 0, 0),
            model_update_frequency="3 months",
            training_as_of_date_frequencies=["1 day"],
            test_as_of_date_frequencies=["1 day"],
            max_training_histories=["1 year"],
            test_durations=["6 months"],
            test_label_timespans=["1 months"],
            training_label_timespans=["3 days"],
        )

        # this should throw an exception because last possible label date is after
        # end of feature time
        result = chopper.calculate_train_test_split_times(
            training_label_timespan=convert_str_to_relativedelta("3 days"),
            test_duration="6 months",
            test_label_timespan=convert_str_to_relativedelta("1 month"),
        )

        assert result == expected_result

    def test_labels_after_features(self):
        chopper = Timechop(
            feature_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2016, 1, 1, 0, 0),
            label_start_time=datetime.datetime(2015, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2017, 1, 1, 0, 0),
            model_update_frequency="3 months",
            training_as_of_date_frequencies=["1 day"],
            test_as_of_date_frequencies=["1 day"],
            max_training_histories=["1 year"],
            test_durations=["6 months"],
            test_label_timespans=["1 months"],
            training_label_timespans=["3 days"],
        )

        # this should throw an exception because last possible label date is after
        # end of feature time
        with self.assertRaises(ValueError):
            chopper.calculate_train_test_split_times(
                training_label_timespan=convert_str_to_relativedelta("3 days"),
                test_duration="6 months",
                test_label_timespan=convert_str_to_relativedelta("1 month"),
            )

    def test_no_valid_label_dates(self):
        chopper = Timechop(
            feature_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2016, 1, 1, 0, 0),
            label_start_time=datetime.datetime(2015, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2015, 2, 1, 0, 0),
            model_update_frequency="3 months",
            training_as_of_date_frequencies=["1 day"],
            test_as_of_date_frequencies=["1 day"],
            max_training_histories=["1 year"],
            test_durations=["6 months"],
            test_label_timespans=["1 months"],
            training_label_timespans=["3 days"],
        )

        # this should raise an error because there are no valid label dates in
        # the labeling time (label span is longer than labeling time)
        with self.assertRaises(ValueError):
            chopper.calculate_train_test_split_times(
                training_label_timespan=convert_str_to_relativedelta("3 days"),
                test_duration="6 months",
                test_label_timespan=convert_str_to_relativedelta("1 month"),
            )


def test_calculate_as_of_times_one_day_freq():
    expected_result = [
        datetime.datetime(2011, 1, 1, 0, 0),
        datetime.datetime(2011, 1, 2, 0, 0),
        datetime.datetime(2011, 1, 3, 0, 0),
        datetime.datetime(2011, 1, 4, 0, 0),
        datetime.datetime(2011, 1, 5, 0, 0),
        datetime.datetime(2011, 1, 6, 0, 0),
        datetime.datetime(2011, 1, 7, 0, 0),
        datetime.datetime(2011, 1, 8, 0, 0),
        datetime.datetime(2011, 1, 9, 0, 0),
        datetime.datetime(2011, 1, 10, 0, 0),
        datetime.datetime(2011, 1, 11, 0, 0),
    ]
    chopper = Timechop(
        feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
        feature_end_time=datetime.datetime(2012, 1, 1, 0, 0),
        label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
        label_end_time=datetime.datetime(2012, 1, 1, 0, 0),
        model_update_frequency="1 year",
        training_as_of_date_frequencies=["1 days"],
        test_as_of_date_frequencies=["7 days"],
        max_training_histories=["10 days", "1 year"],
        test_durations=["1 month"],
        test_label_timespans=["1 day"],
        training_label_timespans=["3 months"],
    )
    result = chopper.calculate_as_of_times(
        as_of_start_limit=datetime.datetime(2011, 1, 1, 0, 0),
        as_of_end_limit=datetime.datetime(2011, 1, 11, 0, 0),
        data_frequency=convert_str_to_relativedelta("1 days"),
    )
    assert result == expected_result


def test_calculate_as_of_times_three_day_freq():
    expected_result = [
        datetime.datetime(2011, 1, 1, 0, 0),
        datetime.datetime(2011, 1, 4, 0, 0),
        datetime.datetime(2011, 1, 7, 0, 0),
        datetime.datetime(2011, 1, 10, 0, 0),
    ]
    chopper = Timechop(
        feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
        feature_end_time=datetime.datetime(2012, 1, 1, 0, 0),
        label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
        label_end_time=datetime.datetime(2012, 1, 1, 0, 0),
        model_update_frequency="1 year",
        training_as_of_date_frequencies=["1 days"],
        test_as_of_date_frequencies=["7 days"],
        max_training_histories=["10 days", "1 year"],
        test_durations=["1 month"],
        test_label_timespans=["1 day"],
        training_label_timespans=["3 months"],
    )
    result = chopper.calculate_as_of_times(
        as_of_start_limit=datetime.datetime(2011, 1, 1, 0, 0),
        as_of_end_limit=datetime.datetime(2011, 1, 11, 0, 0),
        data_frequency=convert_str_to_relativedelta("3 days"),
        forward=True,
    )
    assert result == expected_result


class test_generate_matrix_definitions(TestCase):
    def test_look_back_time_equal_modeling_start(self):
        # TODO: rework this test since the test label window of 3 months
        # cannot be satisfied by the 10 day difference between modeling
        # start and end times, so it's not a very realistic case
        expected_result = {
            "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
            "label_start_time": datetime.datetime(2010, 1, 1, 0, 0),
            "feature_end_time": datetime.datetime(2010, 1, 11, 0, 0),
            "label_end_time": datetime.datetime(2010, 1, 11, 0, 0),
            "train_matrix": {
                "first_as_of_time": datetime.datetime(2010, 1, 1, 0, 0),
                "last_as_of_time": datetime.datetime(2010, 1, 5, 0, 0),
                "matrix_info_end_time": datetime.datetime(2010, 1, 6, 0, 0),
                "as_of_times": [
                    datetime.datetime(2010, 1, 1, 0, 0),
                    datetime.datetime(2010, 1, 2, 0, 0),
                    datetime.datetime(2010, 1, 3, 0, 0),
                    datetime.datetime(2010, 1, 4, 0, 0),
                    datetime.datetime(2010, 1, 5, 0, 0),
                ],
                "training_label_timespan": "1 day",
                "training_as_of_date_frequency": "1 days",
                "max_training_history": "5 days",
            },
            "test_matrices": [
                {
                    "first_as_of_time": datetime.datetime(2010, 1, 6, 0, 0),
                    "last_as_of_time": datetime.datetime(2010, 1, 9, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 10, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                    ],
                    "test_label_timespan": "1 day",
                    "test_as_of_date_frequency": "3 days",
                    "test_duration": "5 days",
                }
            ],
        }
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 11, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 11, 0, 0),
            model_update_frequency="5 days",
            training_as_of_date_frequencies=["1 days"],
            test_as_of_date_frequencies=["3 days"],
            max_training_histories=["5 days"],
            test_durations=["5 days"],
            test_label_timespans=["1 day"],
            training_label_timespans=["1 day"],
        )
        result = chopper.generate_matrix_definitions(
            train_test_split_time=datetime.datetime(2010, 1, 6, 0, 0),
            training_as_of_date_frequency="1 days",
            max_training_history="5 days",
            test_duration="5 days",
            test_label_timespan="1 day",
            training_label_timespan="1 day",
        )
        assert result == expected_result

    def test_look_back_time_before_modeling_start(self):
        expected_result = {
            "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
            "label_start_time": datetime.datetime(2010, 1, 1, 0, 0),
            "feature_end_time": datetime.datetime(2010, 1, 11, 0, 0),
            "label_end_time": datetime.datetime(2010, 1, 11, 0, 0),
            "train_matrix": {
                "first_as_of_time": datetime.datetime(2010, 1, 1, 0, 0),
                "last_as_of_time": datetime.datetime(2010, 1, 5, 0, 0),
                "matrix_info_end_time": datetime.datetime(2010, 1, 6, 0, 0),
                "as_of_times": [
                    datetime.datetime(2010, 1, 1, 0, 0),
                    datetime.datetime(2010, 1, 2, 0, 0),
                    datetime.datetime(2010, 1, 3, 0, 0),
                    datetime.datetime(2010, 1, 4, 0, 0),
                    datetime.datetime(2010, 1, 5, 0, 0),
                ],
                "training_label_timespan": "1 day",
                "training_as_of_date_frequency": "1 days",
                "max_training_history": "10 days",
            },
            "test_matrices": [
                {
                    "first_as_of_time": datetime.datetime(2010, 1, 6, 0, 0),
                    "last_as_of_time": datetime.datetime(2010, 1, 9, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 10, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                    ],
                    "test_label_timespan": "1 day",
                    "test_as_of_date_frequency": "3 days",
                    "test_duration": "5 days",
                },
                {
                    "first_as_of_time": datetime.datetime(2010, 1, 6, 0, 0),
                    "last_as_of_time": datetime.datetime(2010, 1, 6, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 7, 0, 0),
                    "as_of_times": [datetime.datetime(2010, 1, 6, 0, 0)],
                    "test_label_timespan": "1 day",
                    "test_as_of_date_frequency": "6 days",
                    "test_duration": "5 days",
                },
            ],
        }
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 11, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 11, 0, 0),
            model_update_frequency="5 days",
            training_as_of_date_frequencies=["1 days"],
            test_as_of_date_frequencies=["3 days", "6 days"],
            max_training_histories=["10 days"],
            test_durations=["5 days"],
            test_label_timespans=["1 day"],
            training_label_timespans=["1 day"],
        )
        result = chopper.generate_matrix_definitions(
            train_test_split_time=datetime.datetime(2010, 1, 6, 0, 0),
            training_as_of_date_frequency="1 days",
            max_training_history="10 days",
            test_duration="5 days",
            test_label_timespan="1 day",
            training_label_timespan="1 day",
        )
        assert result == expected_result


class test_chop_time(TestCase):
    def test_evenly_divisible_values(self):
        expected_result = [
            {
                "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
                "label_start_time": datetime.datetime(2010, 1, 1, 0, 0),
                "feature_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "label_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "train_matrix": {
                    "first_as_of_time": datetime.datetime(2010, 1, 1, 0, 0),
                    "last_as_of_time": datetime.datetime(2010, 1, 4, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 5, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 1, 0, 0),
                        datetime.datetime(2010, 1, 2, 0, 0),
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                    ],
                    "training_label_timespan": "1 day",
                    "training_as_of_date_frequency": "1 days",
                    "max_training_history": "5 days",
                },
                "test_matrices": [
                    {
                        "first_as_of_time": datetime.datetime(2010, 1, 5, 0, 0),
                        "last_as_of_time": datetime.datetime(2010, 1, 9, 0, 0),
                        "matrix_info_end_time": datetime.datetime(2010, 1, 10, 0, 0),
                        "as_of_times": [
                            datetime.datetime(2010, 1, 5, 0, 0),
                            datetime.datetime(2010, 1, 6, 0, 0),
                            datetime.datetime(2010, 1, 7, 0, 0),
                            datetime.datetime(2010, 1, 8, 0, 0),
                            datetime.datetime(2010, 1, 9, 0, 0),
                        ],
                        "test_label_timespan": "1 day",
                        "test_as_of_date_frequency": "1 days",
                        "test_duration": "5 days",
                    }
                ],
            },
            {
                "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
                "label_start_time": datetime.datetime(2010, 1, 1, 0, 0),
                "feature_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "label_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "train_matrix": {
                    "first_as_of_time": datetime.datetime(2010, 1, 4, 0, 0),
                    "last_as_of_time": datetime.datetime(2010, 1, 9, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 10, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0),
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                    ],
                    "training_label_timespan": "1 day",
                    "training_as_of_date_frequency": "1 days",
                    "max_training_history": "5 days",
                },
                "test_matrices": [
                    {
                        "first_as_of_time": datetime.datetime(2010, 1, 10, 0, 0),
                        "last_as_of_time": datetime.datetime(2010, 1, 14, 0, 0),
                        "matrix_info_end_time": datetime.datetime(2010, 1, 15, 0, 0),
                        "as_of_times": [
                            datetime.datetime(2010, 1, 10, 0, 0),
                            datetime.datetime(2010, 1, 11, 0, 0),
                            datetime.datetime(2010, 1, 12, 0, 0),
                            datetime.datetime(2010, 1, 13, 0, 0),
                            datetime.datetime(2010, 1, 14, 0, 0),
                        ],
                        "test_label_timespan": "1 day",
                        "test_as_of_date_frequency": "1 days",
                        "test_duration": "5 days",
                    }
                ],
            },
        ]
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            model_update_frequency="5 days",
            training_as_of_date_frequencies=["1 days"],
            test_as_of_date_frequencies=["1 days"],
            max_training_histories=["5 days"],
            test_durations=["5 days"],
            test_label_timespans=["1 day"],
            training_label_timespans=["1 day"],
        )
        result = chopper.chop_time()
        assert result == expected_result

    def test_training_label_timespan_longer_than_1_day(self):
        expected_result = [
            {
                "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
                "label_start_time": datetime.datetime(2010, 1, 1, 0, 0),
                "feature_end_time": datetime.datetime(2010, 1, 19, 0, 0),
                "label_end_time": datetime.datetime(2010, 1, 19, 0, 0),
                "train_matrix": {
                    "first_as_of_time": datetime.datetime(2010, 1, 1, 0, 0),
                    "last_as_of_time": datetime.datetime(2010, 1, 4, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 9, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 1, 0, 0),
                        datetime.datetime(2010, 1, 2, 0, 0),
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                    ],
                    "training_label_timespan": "5 days",
                    "training_as_of_date_frequency": "1 days",
                    "max_training_history": "5 days",
                },
                "test_matrices": [
                    {
                        "first_as_of_time": datetime.datetime(2010, 1, 9, 0, 0),
                        "last_as_of_time": datetime.datetime(2010, 1, 13, 0, 0),
                        "matrix_info_end_time": datetime.datetime(2010, 1, 18, 0, 0),
                        "as_of_times": [
                            datetime.datetime(2010, 1, 9, 0, 0),
                            datetime.datetime(2010, 1, 10, 0, 0),
                            datetime.datetime(2010, 1, 11, 0, 0),
                            datetime.datetime(2010, 1, 12, 0, 0),
                            datetime.datetime(2010, 1, 13, 0, 0),
                        ],
                        "test_label_timespan": "5 days",
                        "test_as_of_date_frequency": "1 days",
                        "test_duration": "5 days",
                    }
                ],
            }
        ]
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 19, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 19, 0, 0),
            model_update_frequency="5 days",
            training_as_of_date_frequencies=["1 days"],
            test_as_of_date_frequencies=["1 days"],
            max_training_histories=["5 days"],
            test_durations=["5 days"],
            test_label_timespans=["5 days"],
            training_label_timespans=["5 days"],
        )
        result = chopper.chop_time()
        assert result == expected_result

    def test_unevenly_divisible_lookback_duration(self):
        expected_result = [
            {
                "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
                "label_start_time": datetime.datetime(2010, 1, 1, 0, 0),
                "feature_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "label_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "train_matrix": {
                    "first_as_of_time": datetime.datetime(2010, 1, 1, 0, 0),
                    "last_as_of_time": datetime.datetime(2010, 1, 4, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 5, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 1, 0, 0),
                        datetime.datetime(2010, 1, 2, 0, 0),
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                    ],
                    "training_label_timespan": "1 day",
                    "training_as_of_date_frequency": "1 days",
                    "max_training_history": "7 days",
                },
                "test_matrices": [
                    {
                        "first_as_of_time": datetime.datetime(2010, 1, 5, 0, 0),
                        "last_as_of_time": datetime.datetime(2010, 1, 9, 0, 0),
                        "matrix_info_end_time": datetime.datetime(2010, 1, 10, 0, 0),
                        "as_of_times": [
                            datetime.datetime(2010, 1, 5, 0, 0),
                            datetime.datetime(2010, 1, 6, 0, 0),
                            datetime.datetime(2010, 1, 7, 0, 0),
                            datetime.datetime(2010, 1, 8, 0, 0),
                            datetime.datetime(2010, 1, 9, 0, 0),
                        ],
                        "test_label_timespan": "1 day",
                        "test_as_of_date_frequency": "1 days",
                        "test_duration": "5 days",
                    }
                ],
            },
            {
                "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
                "label_start_time": datetime.datetime(2010, 1, 1, 0, 0),
                "feature_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "label_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "train_matrix": {
                    "first_as_of_time": datetime.datetime(2010, 1, 2, 0, 0),
                    "last_as_of_time": datetime.datetime(2010, 1, 9, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 10, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 2, 0, 0),
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0),
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                    ],
                    "training_label_timespan": "1 day",
                    "training_as_of_date_frequency": "1 days",
                    "max_training_history": "7 days",
                },
                "test_matrices": [
                    {
                        "first_as_of_time": datetime.datetime(2010, 1, 10, 0, 0),
                        "last_as_of_time": datetime.datetime(2010, 1, 14, 0, 0),
                        "matrix_info_end_time": datetime.datetime(2010, 1, 15, 0, 0),
                        "as_of_times": [
                            datetime.datetime(2010, 1, 10, 0, 0),
                            datetime.datetime(2010, 1, 11, 0, 0),
                            datetime.datetime(2010, 1, 12, 0, 0),
                            datetime.datetime(2010, 1, 13, 0, 0),
                            datetime.datetime(2010, 1, 14, 0, 0),
                        ],
                        "test_label_timespan": "1 day",
                        "test_as_of_date_frequency": "1 days",
                        "test_duration": "5 days",
                    }
                ],
            },
        ]
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            model_update_frequency="5 days",
            training_as_of_date_frequencies=["1 days"],
            test_as_of_date_frequencies=["1 days"],
            max_training_histories=["7 days"],
            test_durations=["5 days"],
            test_label_timespans=["1 day"],
            training_label_timespans=["1 day"],
        )
        result = chopper.chop_time()
        assert result == expected_result

    def test_unevenly_divisible_update_window(self):
        expected_result = [
            {
                "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
                "label_start_time": datetime.datetime(2010, 1, 3, 0, 0),
                "feature_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "label_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "train_matrix": {
                    "first_as_of_time": datetime.datetime(2010, 1, 3, 0, 0),
                    "last_as_of_time": datetime.datetime(2010, 1, 4, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 5, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                    ],
                    "training_label_timespan": "1 day",
                    "training_as_of_date_frequency": "1 days",
                    "max_training_history": "5 days",
                },
                "test_matrices": [
                    {
                        "first_as_of_time": datetime.datetime(2010, 1, 5, 0, 0),
                        "last_as_of_time": datetime.datetime(2010, 1, 9, 0, 0),
                        "matrix_info_end_time": datetime.datetime(2010, 1, 10, 0, 0),
                        "as_of_times": [
                            datetime.datetime(2010, 1, 5, 0, 0),
                            datetime.datetime(2010, 1, 6, 0, 0),
                            datetime.datetime(2010, 1, 7, 0, 0),
                            datetime.datetime(2010, 1, 8, 0, 0),
                            datetime.datetime(2010, 1, 9, 0, 0),
                        ],
                        "test_label_timespan": "1 day",
                        "test_as_of_date_frequency": "1 days",
                        "test_duration": "5 days",
                    }
                ],
            },
            {
                "feature_start_time": datetime.datetime(1990, 1, 1, 0, 0),
                "label_start_time": datetime.datetime(2010, 1, 3, 0, 0),
                "feature_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "label_end_time": datetime.datetime(2010, 1, 16, 0, 0),
                "train_matrix": {
                    "first_as_of_time": datetime.datetime(2010, 1, 4, 0, 0),
                    "last_as_of_time": datetime.datetime(2010, 1, 9, 0, 0),
                    "matrix_info_end_time": datetime.datetime(2010, 1, 10, 0, 0),
                    "as_of_times": [
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0),
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                    ],
                    "training_label_timespan": "1 day",
                    "training_as_of_date_frequency": "1 days",
                    "max_training_history": "5 days",
                },
                "test_matrices": [
                    {
                        "first_as_of_time": datetime.datetime(2010, 1, 10, 0, 0),
                        "last_as_of_time": datetime.datetime(2010, 1, 14, 0, 0),
                        "matrix_info_end_time": datetime.datetime(2010, 1, 15, 0, 0),
                        "as_of_times": [
                            datetime.datetime(2010, 1, 10, 0, 0),
                            datetime.datetime(2010, 1, 11, 0, 0),
                            datetime.datetime(2010, 1, 12, 0, 0),
                            datetime.datetime(2010, 1, 13, 0, 0),
                            datetime.datetime(2010, 1, 14, 0, 0),
                        ],
                        "test_label_timespan": "1 day",
                        "test_as_of_date_frequency": "1 days",
                        "test_duration": "5 days",
                    }
                ],
            },
        ]
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 3, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            model_update_frequency="5 days",
            training_as_of_date_frequencies=["1 days"],
            test_as_of_date_frequencies=["1 days"],
            max_training_histories=["5 days"],
            test_durations=["5 days"],
            test_label_timespans=["1 day"],
            training_label_timespans=["1 day"],
        )
        result = chopper.chop_time()
        assert result == expected_result


class test__init__(TestCase):
    def test_bad_feature_start_time(self):
        with self.assertRaises(ValueError):
            Timechop(
                feature_start_time=datetime.datetime(2011, 1, 1, 0, 0),
                feature_end_time=datetime.datetime(2010, 1, 16, 0, 0),
                label_start_time=datetime.datetime(2010, 1, 3, 0, 0),
                label_end_time=datetime.datetime(2010, 1, 16, 0, 0),
                model_update_frequency="5 days",
                training_as_of_date_frequencies=["1 days"],
                test_as_of_date_frequencies=["1 days"],
                max_training_histories=["5 days"],
                test_durations=["5 days"],
                test_label_timespans=["1 day"],
                training_label_timespans=["1 day"],
            )
