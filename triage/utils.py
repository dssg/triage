from datetime import datetime
from dateutil.relativedelta import relativedelta


def temporal_splits(start_time, end_time, update_window, prediction_windows):
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

    for window in prediction_windows:
        test_end_time = end_time_date
        test_end_max = start_time_date + 2 * relativedelta(months=+window)
        while (test_end_time >= test_end_max):
            test_start_time = test_end_time - relativedelta(months=+window)
            train_end_time = test_start_time - relativedelta(days=+1)
            train_start_time = train_end_time - relativedelta(months=+window)
            while (train_start_time >= start_time_date):
                train_start_time -= relativedelta(months=+window)
                yield {
                    'train_start': train_start_time,
                    'train_end': train_end_time,
                    'test_start': test_start_time,
                    'test_end': test_end_time,
                    'prediction_window': window
                }
            test_end_time -= relativedelta(months=+update_window)
