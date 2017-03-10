from timechop.utils import convert_str_to_relativedelta
from timechop.utils import parse_delta_string
import datetime
import unittest
import warnings

class test_convert_str_to_relativedelta(unittest.TestCase):
    def test_valid_input(self):
        date = datetime.datetime(2016, 1, 1, 0, 0)
        delta_strings = ['1 year', '2 months', '3 days', '3s', '2y']
        deltas = [convert_str_to_relativedelta(delta_string) for delta_string in delta_strings]

        assert(date + deltas[0] == datetime.datetime(2017,  1,  1,  0, 0))
        assert(date - deltas[0] == datetime.datetime(2015,  1,  1,  0, 0))
        assert(date + deltas[1] == datetime.datetime(2016,  3,  1,  0, 0))
        assert(date - deltas[1] == datetime.datetime(2015, 11,  1,  0, 0))
        assert(date + deltas[2] == datetime.datetime(2016,  1,  4,  0, 0))
        assert(date - deltas[2] == datetime.datetime(2015, 12, 29,  0, 0))
        assert(date + deltas[3] == datetime.datetime(2016,  1,  1,  0,  0,  3))
        assert(date - deltas[3] == datetime.datetime(2015, 12, 31, 23, 59, 57))
        assert(date + deltas[4] == datetime.datetime(2018,  1,  1,  0, 0))
        assert(date - deltas[4] == datetime.datetime(2014,  1,  1,  0, 0))

    def test_bad_input(self):
        bad_delta_string = '4 tacos'
        with self.assertRaises(ValueError):
            convert_str_to_relativedelta(bad_delta_string)

    def test_warning_for_m(self):
        delta_strings = ['1m', '2M']
        for delta_string in delta_strings:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                convert_str_to_relativedelta(delta_string)
                assert len(w) == 1
                assert issubclass(w[-1].category, RuntimeWarning)
                assert 'months' in str(w[-1].message)


class test_parse_delta_string(unittest.TestCase):
    def test_valid_input(self):
        delta_strings = ['1 year', '2 months', '3 days', '3s', '2y']
        expected_results = [
            ('year', 1),
            ('months', 2),
            ('days', 3),
            ('s', 3),
            ('y', 2)
        ]        
        for index, delta_string in enumerate(delta_strings):
            result = parse_delta_string(delta_string)
            assert(result == expected_results[index])

    def test_invalid_input(self):
        delta_strings = ['one year', '1yr', 'one_year']
        for delta_string in delta_strings:
            with self.assertRaises(ValueError):
                parse_delta_string(delta_string)
