from timechop.timechop import Inspections
import datetime
from unittest import TestCase

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
        result = chopper.calculate_matrix_end_times()
        assert(result == expected_result)

    def test_invalid_input(self):
        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2012, 1, 1, 0, 0),
            update_window = '5 months',
            look_back_durations = ['1 year']
        )
        with self.assertRaises(ValueError):
            chopper.calculate_matrix_end_times()


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
    def test_valid_input(self):
        expected_result = {
            'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
            'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
            'modeling_end_time': datetime.datetime(2010, 1, 11, 0, 0),
            'train_matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
            'train_matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
            'train_as_of_times': [
                datetime.datetime(2010, 1, 1, 0, 0),
                datetime.datetime(2010, 1, 2, 0, 0),
                datetime.datetime(2010, 1, 3, 0, 0),
                datetime.datetime(2010, 1, 4, 0, 0),
                datetime.datetime(2010, 1, 5, 0, 0)
            ],
            'test_matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
            'test_matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
            'test_as_of_times': [
                datetime.datetime(2010, 1, 6, 0, 0),
                datetime.datetime(2010, 1, 7, 0, 0),
                datetime.datetime(2010, 1, 8, 0, 0),
                datetime.datetime(2010, 1, 9, 0, 0),
                datetime.datetime(2010, 1, 10, 0, 0)
            ],
        }
        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 11, 0, 0),
            update_window = '5 days',
            look_back_durations = ['5 days']
        )
        result = chopper.generate_matrix_definition(
            train_matrix_end_time = datetime.datetime(2010, 1, 06, 0, 0),
            look_back_duration = '5 days'
        )
        assert(result == expected_result)

    def test_invalid_input(self):
        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 11, 0, 0),
            update_window = '5 days',
            look_back_durations = ['10 days']
        )
        with self.assertRaises(ValueError):
            chopper.generate_matrix_definition(
                train_matrix_end_time = datetime.datetime(2010, 1, 6, 0, 0),
                look_back_duration = '10 days'
            )


class test_chop_time(TestCase):
    def test_valid_input(self):
        expected_result = [
            {
                'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
                'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'train_matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                'train_as_of_times': [
                    datetime.datetime(2010, 1, 1, 0, 0),
                    datetime.datetime(2010, 1, 2, 0, 0),
                    datetime.datetime(2010, 1, 3, 0, 0),
                    datetime.datetime(2010, 1, 4, 0, 0),
                    datetime.datetime(2010, 1, 5, 0, 0)
                ],
                'test_matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                'test_matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'test_as_of_times': [
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
                ],
            },
            {
                'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
                'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                'train_matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                'train_as_of_times': [
                    datetime.datetime(2010, 1, 6, 0, 0),
                    datetime.datetime(2010, 1, 7, 0, 0),
                    datetime.datetime(2010, 1, 8, 0, 0),
                    datetime.datetime(2010, 1, 9, 0, 0),
                    datetime.datetime(2010, 1, 10, 0, 0)
                ],
                'test_matrix_start_time': datetime.datetime(2010, 1, 11, 0, 0),
                'test_matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'test_as_of_times': [
                    datetime.datetime(2010, 1, 11, 0, 0),
                    datetime.datetime(2010, 1, 12, 0, 0),
                    datetime.datetime(2010, 1, 13, 0, 0),
                    datetime.datetime(2010, 1, 14, 0, 0),
                    datetime.datetime(2010, 1, 15, 0, 0)
                ],
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

    def test_bad_look_back_time(self):
        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
            update_window = '5 days',
            look_back_durations = ['6 days']
        )
        with self.assertRaises(ValueError):
            chopper.chop_time()

    def test_bad_update_window(self):
        chopper = Inspections(
            beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
            modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
            modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
            update_window = '6 days',
            look_back_durations = ['5 days']
        )
        with self.assertRaises(ValueError):
            chopper.chop_time()


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
