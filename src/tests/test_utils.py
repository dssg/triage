import datetime
import unittest

from triage.util.conf import convert_str_to_relativedelta, parse_delta_string


class test_convert_str_to_relativedelta(unittest.TestCase):
    def test_valid_input(self):
        date = datetime.datetime(2016, 1, 1, 0, 0)
        for test in (
            {
                "interval": "1 year",
                "addition_result": datetime.datetime(2017, 1, 1, 0, 0),
                "subtraction_result": datetime.datetime(2015, 1, 1, 0, 0),
            },
            {
                "interval": "2 months",
                "addition_result": datetime.datetime(2016, 3, 1, 0, 0),
                "subtraction_result": datetime.datetime(2015, 11, 1, 0, 0),
            },
            {
                "interval": "3 days",
                "addition_result": datetime.datetime(2016, 1, 4, 0, 0),
                "subtraction_result": datetime.datetime(2015, 12, 29, 0, 0),
            },
            {
                "interval": "3 seconds",
                "addition_result": datetime.datetime(2016, 1, 1, 0, 0, 3),
                "subtraction_result": datetime.datetime(2015, 12, 31, 23, 59, 57),
            },
            {
                "interval": "2year",
                "addition_result": datetime.datetime(2018, 1, 1, 0, 0),
                "subtraction_result": datetime.datetime(2014, 1, 1, 0, 0),
            },
            {
                "interval": "4 weeks",
                "addition_result": datetime.datetime(2016, 1, 29, 0, 0),
                "subtraction_result": datetime.datetime(2015, 12, 4, 0, 0),
            },
            {
                "interval": "5 hours",
                "addition_result": datetime.datetime(2016, 1, 1, 5, 0),
                "subtraction_result": datetime.datetime(2015, 12, 31, 19, 0),
            },
            {
                "interval": "10minutes",
                "addition_result": datetime.datetime(2016, 1, 1, 0, 10),
                "subtraction_result": datetime.datetime(2015, 12, 31, 23, 50),
            },
            {
                "interval": "1microsecond",
                "addition_result": datetime.datetime(2016, 1, 1, 0, 0, 0, 1),
                "subtraction_result": datetime.datetime(
                    2015, 12, 31, 23, 59, 59, 999999
                ),
            },
            {
                "interval": "5m",
                "addition_result": datetime.datetime(2016, 1, 1, 0, 5),
                "subtraction_result": datetime.datetime(2015, 12, 31, 23, 55),
            },
        ):
            delta = convert_str_to_relativedelta(test["interval"])
            assert date + delta == test["addition_result"]
            assert date - delta == test["subtraction_result"]

    def test_bad_input(self):
        for bad_delta_string in ("4 tacos", "3"):
            with self.assertRaises(ValueError):
                convert_str_to_relativedelta(bad_delta_string)


class test_parse_delta_string(unittest.TestCase):
    def test_valid_input(self):
        delta_strings = ["1 year", "2 months", "3 days", "3s", "2y", "10m"]
        expected_results = [
            ("year", 1),
            ("months", 2),
            ("days", 3),
            ("s", 3),
            ("y", 2),
            ("m", 10),
        ]
        for index, delta_string in enumerate(delta_strings):
            result = parse_delta_string(delta_string)
            assert result == expected_results[index]

    def test_invalid_input(self):
        delta_strings = ["one year", "one_year", "ay"]
        for delta_string in delta_strings:
            with self.assertRaises(ValueError):
                parse_delta_string(delta_string)
