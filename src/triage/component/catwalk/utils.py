import datetime
import pickle
import tempfile
import hashlib
import botocore
import pandas
import random
import yaml
import json
from results_schema import Experiment, Model
from retrying import retry
from sqlalchemy.orm import sessionmaker
import sqlalchemy
import csv
import postgres_copy


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


def filename_friendly_hash(inputs):
    def dt_handler(x):
        if isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
            return x.isoformat()
        raise TypeError("Unknown type")
    return hashlib.md5(
        json.dumps(inputs, default=dt_handler, sort_keys=True)
            .encode('utf-8')
    ).hexdigest()


def retry_if_db_error(exception):
    return isinstance(exception, sqlalchemy.exc.OperationalError)


DEFAULT_RETRY_KWARGS = {
    'retry_on_exception': retry_if_db_error,
    'wait_exponential_multiplier': 1000, # wait 2^x*1000ms between each retry
    'stop_max_attempt_number': 14,
    # with this configuration, last wait will be ~2 hours
    # for a total of ~4.5 hours waiting
}


db_retry = retry(**DEFAULT_RETRY_KWARGS)


@db_retry
def save_experiment_and_get_hash(config, db_engine):
    experiment_hash = filename_friendly_hash(config)
    session = sessionmaker(bind=db_engine)()
    session.merge(Experiment(
        experiment_hash=experiment_hash,
        config=config
    ))
    session.commit()
    session.close()
    return experiment_hash


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


def sort_predictions_and_labels(predictions_proba, labels, sort_seed):
    random.seed(sort_seed)
    predictions_proba_sorted, labels_sorted = zip(*sorted(
        zip(predictions_proba, labels),
        key=lambda pair: (pair[0], random.random()), reverse=True)
    )
    return predictions_proba_sorted, labels_sorted


@db_retry
def retrieve_model_id_from_hash(db_engine, model_hash):
    """Retrieves a model id from the database that matches the given hash

    Args:
        db_engine (sqlalchemy.engine) A database engine
        model_hash (str) The model hash to lookup

    Returns: (int) The model id (if found in DB), None (if not)
    """
    session = sessionmaker(bind=db_engine)()
    try:
        saved = session.query(Model)\
            .filter_by(model_hash=model_hash)\
            .one_or_none()
        return saved.model_id if saved else None
    finally:
        session.close()


@db_retry
def save_db_objects(db_engine, db_objects):
    """Saves a collection of SQLAlchemy model objects to the database using a COPY command

    Args:
        db_engine (sqlalchemy.engine)
        db_objects (list) SQLAlchemy model objects, corresponding to a valid table
    """
    with tempfile.TemporaryFile(mode='w+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for db_object in db_objects:
            writer.writerow([
                getattr(db_object, col.name)
                for col in db_object.__table__.columns
            ])
        f.seek(0)
        postgres_copy.copy_from(f, type(db_objects[0]), db_engine, format='csv')
