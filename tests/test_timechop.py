from timechop.timechop import Inspections
import datetime
from unittest import TestCase
import warnings

class test_calculate_update_times(TestCase):
    def test_valid_input(self):
        expected_result = [
            datetime.datetime(2011, 1, 1, 0, 0),
            datetime.datetime(2012, 1, 1, 0, 0)
        ]
        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2013, 1, 1, 0, 0),
            update_window = '1 year',
            look_back_durations = ['1 year']
        )
        result = chopper.calculate_matrix_end_times('1 year')
        assert(result == expected_result)

    def test_invalid_input(self):
        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2011, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2011, 2, 1, 0, 0),
            update_window = '5 months',
            look_back_durations = ['1 year']
        )
        with self.assertRaises(ValueError):
            chopper.calculate_matrix_end_times('1 year')


def test_calculate_as_of_times():
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
    chopper = Inspections(
        beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
        modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
        modeling_end_time = datetime.datetime(2012, 1, 1, 0, 0),
        update_window = '1 year',
        look_back_durations = ['10 days', '1 year']
    )
    result = chopper.calculate_as_of_times(
        matrix_start_time = datetime.datetime(2011, 1, 1, 0, 0),
        matrix_end_time = datetime.datetime(2011, 1, 11, 0, 0)
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
                ]
            },
            'test_matrices': [{
                'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 6, 0, 0),
                    datetime.datetime(2010, 1, 7, 0, 0),
                    datetime.datetime(2010, 1, 8, 0, 0),
                    datetime.datetime(2010, 1, 9, 0, 0),
                    datetime.datetime(2010, 1, 10, 0, 0)
                ]
            }]
        }
        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 11, 0, 0),
            update_window = '5 days',
            look_back_durations = ['5 days']
        )
        result = chopper.generate_matrix_definition(
            train_matrix_end_time = datetime.datetime(2010, 1, 6, 0, 0),
            look_back_duration = '5 days'
        )
        assert result == expected_result

    def test_look_back_time_before_modeling_start(self):
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
                ]
            },
            'test_matrices': [{
                'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 6, 0, 0),
                    datetime.datetime(2010, 1, 7, 0, 0),
                    datetime.datetime(2010, 1, 8, 0, 0),
                    datetime.datetime(2010, 1, 9, 0, 0),
                    datetime.datetime(2010, 1, 10, 0, 0)
                ]
            }]
        }
        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 11, 0, 0),
            update_window = '5 days',
            look_back_durations = ['10 days']
        )
        result = chopper.generate_matrix_definition(
            train_matrix_end_time = datetime.datetime(2010, 1, 6, 0, 0),
            look_back_duration = '10 days'
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
                    ]
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0),
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0),
                        datetime.datetime(2010, 1, 15, 0, 0)
                    ]
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
                    ]
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0),
                        datetime.datetime(2010, 1, 15, 0, 0)
                    ]
                }]
            }
        ]
        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
            update_window = '5 days',
            look_back_durations = ['5 days']
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
                    ]
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0),
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0),
                        datetime.datetime(2010, 1, 15, 0, 0)
                    ]
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
                    ]
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0),
                        datetime.datetime(2010, 1, 15, 0, 0)
                    ]
                }]
            }
        ]

        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
            update_window = '5 days',
            look_back_durations = ['7 days']
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
                    ]
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 8, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0),
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0),
                        datetime.datetime(2010, 1, 15, 0, 0)
                    ]
                }]
            }
        ]

        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
            update_window = '8 days',
            look_back_durations = ['5 days']
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
            chopper = Inspections(
                beginning_of_time = datetime.datetime(2011, 1, 1, 0, 0),
                modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
                modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
                update_window = '6 days',
                look_back_durations = ['5 days']
            )
