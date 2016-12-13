from triage.utils import temporal_splits


def assert_split(output, expected):
    for output_row, expected_row in zip(output, expected):
        (
            expected_train_start,
            expected_train_end,
            expected_test_start,
            expected_test_end,
            expected_prediction_window
        ) = expected_row
        assert expected_train_start == output_row['train_start'].date().isoformat()
        assert expected_train_end == output_row['train_end'].date().isoformat()
        assert expected_test_start == output_row['test_start'].date().isoformat()
        assert expected_test_end == output_row['test_end'].date().isoformat()
        assert expected_prediction_window == output_row['prediction_window']


def test_temporal_splits():
    # input: start date, end date, update window (months), prediction windows (months)
    splits = [split for split in temporal_splits('2014-04-01', '2016-04-01', 6, [6])]
    expected = [
        # train start, train end, test start, test end, prediction window (months)
        ('2014-09-30', '2015-09-30', '2015-10-01', '2016-04-01', 6),
        ('2014-03-30', '2015-09-30', '2015-10-01', '2016-04-01', 6),
        ('2014-03-30', '2015-03-31', '2015-04-01', '2015-10-01', 6)
    ]
    assert_split(splits, expected)
