import datetime
from dateutil.relativedelta import relativedelta
import pickle
import tempfile
import hashlib
import botocore
import pandas
import yaml
import json


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


def download_object(cache_key):
    with tempfile.NamedTemporaryFile() as f:
        cache_key.download_fileobj(f)
        f.seek(0)
        return pickle.load(f)


def model_cache_key(project_path, model_id, s3_conn):
    """Generates an s3 key for a given model_id

    Args:
        model_id (string) a unique model id

    Returns:
        (boto3.s3.Object) an s3 key, which may or may not have contents
    """
    bucket_name, prefix = split_s3_path(project_path)
    path = '/'.join([prefix, 'trained_models', model_id])
    return s3_conn.Object(bucket_name, path)


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


def get_matrix_and_metadata(matrix_path, metadata_path):
    """Retrieve a matrix in hdf format and
    metadata about the matrix in yaml format

    Returns: (tuple) matrix, metadata
    """
    matrix = pandas.read_hdf(matrix_path)
    with open(metadata_path) as f:
        metadata = yaml.load(f)
    return matrix, metadata


def temporal_splits(
    start_time,
    end_time,
    update_window,
    prediction_windows,
    feature_frequency,
    test_frequency
):
    start_time_date = datetime.datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.datetime.strptime(end_time, '%Y-%m-%d')

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
                    'train_as_of_dates': generate_as_of_dates(
                        train_start_time,
                        train_end_time,
                        feature_frequency
                    ),
                    'test_start': test_start_time,
                    'test_end': test_end_time,
                    'test_as_of_dates': generate_as_of_dates(
                        test_start_time,
                        test_end_time,
                        test_frequency
                    ),
                    'prediction_window': window
                }
            test_end_time -= relativedelta(months=+update_window)


def generate_as_of_dates(start_date, end_date, prediction_window):
    as_of_dates = []
    as_of_date = start_date
    while as_of_date <= end_date - relativedelta(months=prediction_window):
        as_of_dates.append(as_of_date)
        as_of_date += relativedelta(months=prediction_window)

    return as_of_dates


def filename_friendly_hash(inputs):
    def dt_handler(x):
        if isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
            return x.isoformat()
        raise TypeError("Unknown type")
    return hashlib.md5(
        json.dumps(inputs, default=dt_handler, sort_keys=True)
            .encode('utf-8')
    ).hexdigest()


class Batch:
    # modified from
    # http://codereview.stackexchange.com/questions/118883/split-up-an-iterable-into-batches
    def __init__(self, iterable, limit=None):
        self.iterator = iter(iterable)
        self.limit = limit
        try:
            self.current = next(self.iterator)
        except StopIteration:
            self.on_going = False
        else:
            self.on_going = True

    def group(self):
        yield self.current
        # start enumerate at 1 because we already yielded the last saved item
        for num, item in enumerate(self.iterator, 1):
            self.current = item
            if num == self.limit:
                break
            yield item
        else:
            self.on_going = False

    def __iter__(self):
        while self.on_going:
            yield self.group()
