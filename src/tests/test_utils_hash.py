import datetime
import re

from triage.util.hash import filename_friendly_hash


def test_filename_friendly_hash():
    data = {
        "stuff": "stuff",
        "other_stuff": "more_stuff",
        "a_datetime": datetime.datetime(2015, 1, 1),
        "a_date": datetime.date(2016, 1, 1),
        "a_number": 5.0,
    }
    output = filename_friendly_hash(data)
    assert isinstance(output, str)
    assert re.match("^[\w]+$", output) is not None

    # make sure ordering keys differently doesn't change the hash
    new_output = filename_friendly_hash(
        {
            "other_stuff": "more_stuff",
            "stuff": "stuff",
            "a_datetime": datetime.datetime(2015, 1, 1),
            "a_date": datetime.date(2016, 1, 1),
            "a_number": 5.0,
        }
    )
    assert new_output == output

    # make sure new data hashes to something different
    new_output = filename_friendly_hash({"stuff": "stuff", "a_number": 5.0})
    assert new_output != output


def test_filename_friendly_hash_stability():
    nested_data = {"one": "two", "three": {"four": "five", "six": "seven"}}
    output = filename_friendly_hash(nested_data)
    # 1. we want to make sure this is stable across different runs
    # so hardcode an expected value
    assert output == "9a844a7ebbfd821010b1c2c13f7391e6"
    other_nested_data = {"one": "two", "three": {"six": "seven", "four": "five"}}
    new_output = filename_friendly_hash(other_nested_data)
    assert output == new_output
