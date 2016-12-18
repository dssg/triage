from triage.utils import temporal_splits
import logging


def assert_split(output, expected):
    for output_row, expected_row in zip(output, expected):
        assert expected_row['train_start'] == output_row['train_start'].date().isoformat()
        assert expected_row['train_end'] == output_row['train_end'].date().isoformat()
        assert expected_row['test_start'] == output_row['test_start'].date().isoformat()
        assert expected_row['test_end'] == output_row['test_end'].date().isoformat()
        assert len(expected_row['feature_dates']) == len(output_row['feature_dates'])
        logging.warning(expected_row['train_start'])
        logging.warning(expected_row['train_end'])
        logging.warning(expected_row['feature_dates'])
        logging.warning(output_row['feature_dates'])
        for output_feature_date, expected_feature_date in zip(sorted(output_row['feature_dates']), sorted(expected_row['feature_dates'])):
            assert expected_feature_date == output_feature_date.date().isoformat()


def test_temporal_splits():
    # input: start date, end date, update window (months), prediction windows (months)
    splits = [split for split in temporal_splits('2014-04-01', '2016-04-01', 6, [6])]
    expected = [
        {
            'train_start': '2014-09-30',
            'train_end': '2015-09-30',
            'test_start': '2015-10-01',
            'test_end': '2016-04-01',
            'feature_dates': ['2014-09-30', '2015-03-30'],
        },
        {
            'train_start': '2014-03-30',
            'train_end': '2015-09-30',
            'test_start': '2015-10-01',
            'test_end': '2016-04-01',
            'feature_dates': ['2014-03-30', '2014-09-30', '2015-03-30']
        },
        {
            'train_start': '2014-03-30',
            'train_end': '2015-03-31',
            'test_start': '2015-04-01',
            'test_end': '2015-10-01',
            'feature_dates': ['2014-03-30', '2014-09-30']
        }
    ]
    assert_split(splits, expected)
