from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
import tempfile
import botocore


def split_s3_path(path):
    """
    Args:
        path: a string representing an s3 path including a bucket
            (bucket_name/prefix/prefix2)
    Returns:
        A tuple containing the bucket name and full prefix)
    """
    return path.split('/', 1)


def upload_object_to_key(obj, cache_key):
    """Pickles object and uploads it to the given s3 key

    Args:
        obj (object) any picklable Python object
        cache_key (boto3.s3.Object) an s3 key
    """
    with tempfile.NamedTemporaryFile('w+b') as f:
        pickle.dump(obj, f)
        f.seek(0)
        cache_key.upload_file(f.name)


def key_exists(key):
    try:
        key.load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise
    else:
        return True


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
                    'feature_dates': generate_as_of_dates(
                        train_start_time,
                        train_end_time,
                        window
                    )
                }
            test_end_time -= relativedelta(months=+update_window)


def generate_as_of_dates(start_date, end_date, prediction_window):
    as_of_dates = []
    as_of_date = start_date
    while as_of_date <= end_date - relativedelta(months=prediction_window):
        as_of_dates.append(as_of_date)
        as_of_date += relativedelta(months=prediction_window)

    return as_of_dates
