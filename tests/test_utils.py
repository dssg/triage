from triage.utils import temporal_splits, filename_friendly_hash
import datetime
import logging
import re


def assert_split(output, expected):
    for output_row, expected_row in zip(output, expected):
        assert expected_row['train_start'] == output_row['train_start'].date().isoformat()
        assert expected_row['train_end'] == output_row['train_end'].date().isoformat()
        assert expected_row['test_start'] == output_row['test_start'].date().isoformat()
        assert expected_row['test_end'] == output_row['test_end'].date().isoformat()
        assert len(expected_row['train_as_of_dates']) == len(output_row['train_as_of_dates'])
        for output_feature_date, expected_feature_date in zip(sorted(output_row['train_as_of_dates']), sorted(expected_row['train_as_of_dates'])):
            assert expected_feature_date == output_feature_date.date().isoformat()
        assert len(expected_row['test_as_of_dates']) == len(output_row['test_as_of_dates'])
        for output_feature_date, expected_feature_date in zip(sorted(output_row['test_as_of_dates']), sorted(expected_row['test_as_of_dates'])):
            assert expected_feature_date == output_feature_date.date().isoformat()


def test_temporal_splits():
    splits = [split for split in temporal_splits(
        start_time='2014-04-01',
        end_time='2016-04-01',
        update_window=6,
        prediction_windows=[6],
        feature_frequency=6,
        test_frequency=3
    )]
    expected = [
        {
            'train_start': '2014-09-30',
            'train_end': '2015-09-30',
            'test_start': '2015-10-01',
            'test_end': '2016-04-01',
            'train_as_of_dates': ['2014-09-30', '2015-03-30'],
            'test_as_of_dates': ['2015-10-01', '2016-01-01'],
        },
        {
            'train_start': '2014-03-30',
            'train_end': '2015-09-30',
            'test_start': '2015-10-01',
            'test_end': '2016-04-01',
            'train_as_of_dates': ['2014-03-30', '2014-09-30', '2015-03-30'],
            'test_as_of_dates': ['2015-10-01', '2016-01-01'],
        },
        {
            'train_start': '2014-03-30',
            'train_end': '2015-03-31',
            'test_start': '2015-04-01',
            'test_end': '2015-10-01',
            'train_as_of_dates': ['2014-03-30', '2014-09-30'],
            'test_as_of_dates': ['2015-04-01', '2015-07-01'],
        }
    ]
    assert_split(splits, expected)


def test_filename_friendly_hash():
    data = {
        'stuff': 'stuff',
        'other_stuff': 'more_stuff',
        'a_datetime': datetime.datetime(2015, 1, 1),
        'a_date': datetime.date(2016, 1, 1),
        'a_number': 5.0
    }
    output = filename_friendly_hash(data)
    assert isinstance(output, str)
    assert re.match('^[\w]+$', output) is not None

    # make sure ordering keys differently doesn't change the hash
    new_output = filename_friendly_hash({
        'other_stuff': 'more_stuff',
        'stuff': 'stuff',
        'a_datetime': datetime.datetime(2015, 1, 1),
        'a_date': datetime.date(2016, 1, 1),
        'a_number': 5.0
    })
    assert new_output == output

    # make sure new data hashes to something different
    new_output = filename_friendly_hash({
        'stuff': 'stuff',
        'a_number': 5.0
    })
    assert new_output != output


def test_filename_friendly_hash_stability():
    nested_data = {
        'one': 'two',
        'three': {
            'four': 'five',
            'six': 'seven'
        }
    }
    output = filename_friendly_hash(nested_data)
    # 1. we want to make sure this is stable across different runs
    # so hardcode an expected value
    assert output == '9a844a7ebbfd821010b1c2c13f7391e6'
    other_nested_data = {
        'one': 'two',
        'three': {
            'six': 'seven',
            'four': 'five'
        }
    }
    new_output = filename_friendly_hash(other_nested_data)
    assert output == new_output
