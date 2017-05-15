from timechop.timechop import Timechop
import datetime
from unittest import TestCase
import warnings

class test_calculate_update_times(TestCase):
    def test_valid_input(self):
        expected_result = [
            datetime.datetime(2011, 1, 1, 0, 0),
            datetime.datetime(2012, 1, 1, 0, 0)
        ]
        chopper = Timechop(
            beginning_of_time=datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time=datetime.datetime(2013, 1, 1, 0, 0),
            update_window='1 year',
            train_example_frequency='1 day',
            test_example_frequency='1 day',
            train_durations=['1 year'],
            test_durations=['1 month'],
            train_label_windows=['1 day'],
            test_label_windows=['3 months']
        )
        result = chopper.calculate_matrix_end_times(
            train_duration='1 year',
            train_label_window='1 day',
            test_label_window='3 months'
        )
        assert(result == expected_result)

    def test_invalid_input(self):
        chopper = Timechop(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2011, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2011, 2, 1, 0, 0),
            update_window = '5 months',
            train_example_frequency = '1 day',
            test_example_frequency = '1 day',
            train_durations = ['1 year'],
            test_durations = ['1 month'],
            train_label_windows=['1 day'],
            test_label_windows=['3 months']
        )
        with self.assertRaises(ValueError):
            chopper.calculate_matrix_end_times(
                train_duration='1 year',
                train_label_window='1 day',
                test_label_window='3 months'
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
    ]
    chopper = Timechop(
        beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
        modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
        modeling_end_time = datetime.datetime(2012, 1, 1, 0, 0),
        update_window = '1 year',
        train_example_frequency = '1 days',
        test_example_frequency = '7 days',
        train_durations = ['10 days', '1 year'],
        test_durations = ['1 month'],
        train_label_windows=['1 day'],
        test_label_windows=['3 months']
    )
    result = chopper.calculate_as_of_times(
        matrix_start_time = datetime.datetime(2011, 1, 1, 0, 0),
        matrix_end_time = datetime.datetime(2011, 1, 11, 0, 0),
        example_frequency = '1 days',
    )
    assert(result == expected_result)


def test_calculate_as_of_times_three_day_freq():
    expected_result = [
        datetime.datetime(2011, 1, 1, 0, 0),
        datetime.datetime(2011, 1, 4, 0, 0),
        datetime.datetime(2011, 1, 7, 0, 0),
        datetime.datetime(2011, 1, 10, 0, 0),
    ]
    chopper = Timechop(
        beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
        modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
        modeling_end_time = datetime.datetime(2012, 1, 1, 0, 0),
        update_window = '1 year',
        train_example_frequency = '1 days',
        test_example_frequency = '7 days',
        train_durations = ['10 days', '1 year'],
        test_durations = ['1 month'],
        train_label_windows=['1 day'],
        test_label_windows=['3 months']
    )
    result = chopper.calculate_as_of_times(
        matrix_start_time = datetime.datetime(2011, 1, 1, 0, 0),
        matrix_end_time = datetime.datetime(2011, 1, 11, 0, 0),
        example_frequency = '3 days',
    )
    assert(result == expected_result)


class test_generate_matrix_definition(TestCase):
    def test_look_back_time_equal_modeling_start(self):
        expected_result = {
            'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
            'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
            'modeling_end_time': datetime.datetime(2010, 1, 11, 0, 0),
            'train_matrix': {
                'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 1, 0, 0),
                    datetime.datetime(2010, 1, 2, 0, 0),
                    datetime.datetime(2010, 1, 3, 0, 0),
                    datetime.datetime(2010, 1, 4, 0, 0),
                    datetime.datetime(2010, 1, 5, 0, 0)
                ],
                'label_window': '1 day',
                'example_frequency': '1 days',
                'train_duration': '5 days'
            },
            'test_matrices': [{
                'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 6, 0, 0),
                    datetime.datetime(2010, 1, 9, 0, 0),
                ],
                'label_window': '3 months',
                'example_frequency': '3 days'
            }]
        }
        chopper = Timechop(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 11, 0, 0),
            update_window = '5 days',
            train_example_frequency = '1 days',
            test_example_frequency = '3 days',
            train_durations = ['5 days'],
            test_durations = ['5 days'],
            train_label_windows=['1 day'],
            test_label_windows=['3 months']
        )
        result = chopper.generate_matrix_definition(
            train_matrix_end_time = datetime.datetime(2010, 1, 6, 0, 0),
            train_duration = '5 days',
            train_label_window='1 day',
            test_label_window='3 months'
        )
        assert result == expected_result

    def test_look_back_time_before_modeling_start(self):
        expected_result = {
            'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
            'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
            'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
            'train_matrix': {
                'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 1, 0, 0),
                    datetime.datetime(2010, 1, 2, 0, 0),
                    datetime.datetime(2010, 1, 3, 0, 0),
                    datetime.datetime(2010, 1, 4, 0, 0),
                    datetime.datetime(2010, 1, 5, 0, 0)
                ],
                'label_window': '1 day',
                'example_frequency': '1 days',
                'train_duration': '10 days'
            },
            'test_matrices': [
                {
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                    ],
                    'label_window': '3 months',
                    'example_frequency': '7 days'
                },
                {
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                    ],
                    'label_window': '3 months',
                    'example_frequency': '7 days'
                }
            ]
        }
        # this tests that (a) 5 and 10 day prediction duration return distinct
        # test matrices in a list and (b) 10 and 15 day durations produce
        # redundanct test matrices (because 15 days after training period is 
        # beyond the end of the modeling time), only one of which is returned
        chopper = Timechop(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
            update_window = '5 days',
            train_example_frequency = '1 days',
            test_example_frequency = '7 days',
            train_durations = ['10 days'],
            test_durations = ['5 days', '10 days', '15 days'],
            train_label_windows=['1 day'],
            test_label_windows=['3 months']
        )
        result = chopper.generate_matrix_definition(
            train_matrix_end_time = datetime.datetime(2010, 1, 6, 0, 0),
            train_duration = '10 days',
            train_label_window='1 day',
            test_label_window='3 months'
        )
        assert result == expected_result


class test_chop_time(TestCase):
    def test_evenly_divisible_values(self):
        expected_result = [
            {
                'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
                'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 1, 0, 0),
                        datetime.datetime(2010, 1, 2, 0, 0),
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0)
                    ],
                    'label_window': '1 day',
                    'example_frequency': '1 days',
                    'train_duration': '5 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                    ],
                    'label_window': '3 months',
                    'example_frequency': '3 days'
                }]
            },
            {
                'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
                'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0)
                    ],
                    'label_window': '1 day',
                    'example_frequency': '1 days',
                    'train_duration': '5 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0)
                    ],
                    'label_window': '3 months',
                    'example_frequency': '3 days'
                }]
            }
        ]
        chopper = Timechop(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
            update_window = '5 days',
            train_example_frequency = '1 days',
            test_example_frequency = '3 days',
            train_durations = ['5 days'],
            test_durations = ['5 days'],
            train_label_windows=['1 day'],
            test_label_windows=['3 months']
        )
        result = chopper.chop_time()
        assert(result == expected_result)

    def test_unevenly_divisible_lookback_duration(self):
        expected_result = [
            {
                'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
                'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 1, 0, 0),
                        datetime.datetime(2010, 1, 2, 0, 0),
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0)
                    ],
                    'label_window': '1 day',
                    'example_frequency': '1 days',
                    'train_duration': '7 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                    ],
                    'label_window': '3 months',
                    'example_frequency': '7 days'
                }]
            },
            {
                'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
                'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 4, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0),
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0)
                    ],
                    'label_window': '1 day',
                    'example_frequency': '1 days',
                    'train_duration': '7 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 11, 0, 0),
                    ],
                    'label_window': '3 months',
                    'example_frequency': '7 days'
                }]
            }
        ]

        chopper = Timechop(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
            update_window = '5 days',
            train_example_frequency = '1 days',
            test_example_frequency = '7 days',
            train_durations = ['7 days'],
            test_durations = ['5 days'],
            train_label_windows=['1 day'],
            test_label_windows=['3 months']
        )
        
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter("always")
            result = chopper.chop_time()
            assert result == expected_result
            assert len(w) == 0

    def test_unevenly_divisible_update_window(self):
        expected_result = [
            {
                'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
                'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 3, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 8, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0),
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0)
                    ],
                    'label_window': '1 day',
                    'example_frequency': '1 days',
                    'train_duration': '5 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 8, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 13, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0)
                    ],
                    'label_window': '3 months',
                    'example_frequency': '4 days'
                }]
            }
        ]

        chopper = Timechop(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
            update_window = '8 days',
            train_example_frequency = '1 days',
            test_example_frequency = '4 days',
            train_durations = ['5 days'],
            test_durations = ['5 days'],
            train_label_windows=['1 day'],
            test_label_windows=['3 months']
        )
        
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter("always")
            result = chopper.chop_time()
            assert result == expected_result
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert 'update' in str(w[-1].message)


class test__init__(TestCase):
    def test_bad_beginning_of_time(self):
        with self.assertRaises(ValueError):
            chopper = Timechop(
                beginning_of_time = datetime.datetime(2011, 1, 1, 0, 0),
                modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
                update_window = '6 days',
                train_example_frequency = '1 days',
                test_example_frequency = '7 days',
                train_durations = ['5 days'],
                test_durations = ['5 days'],
                train_label_windows=['1 day'],
                test_label_windows=['3 months']
            )
