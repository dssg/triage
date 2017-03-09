from timechop.utils import convert_str_to_relativedelta
import datetime
import unittest

class test_convert_str_to_relativedelta(unittest.TestCase):

    def test_valid_input(self):
        date = datetime.datetime(2016, 1, 1, 0, 0)
        delta_strings = ['1 year', '2 months', '3 days']
        deltas = [convert_str_to_relativedelta(delta_string) for delta_string in delta_strings]

        assert(date + deltas[0] == datetime.datetime(2017,  1,  1, 0, 0))
        assert(date - deltas[0] == datetime.datetime(2015,  1,  1, 0, 0))
        assert(date + deltas[1] == datetime.datetime(2016,  3,  1, 0, 0))
        assert(date - deltas[1] == datetime.datetime(2015, 11,  1, 0, 0))
        assert(date + deltas[2] == datetime.datetime(2016,  1,  4, 0, 0))
        assert(date - deltas[2] == datetime.datetime(2015, 12, 29, 0, 0))

    def test_bad_input(self):
        bad_delta_string =  '4y'
        with self.assertRaises(ValueError):
            convert_str_to_relativedelta(bad_delta_string)
