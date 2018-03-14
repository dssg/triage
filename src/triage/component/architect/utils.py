import csv
import functools
import operator
import tempfile

import postgres_copy
from retrying import retry
import sqlalchemy


def feature_list(feature_dictionary):
    """Convert a feature dictionary to a sorted list.

    Args: feature_dictionary (dict)

    Returns: sorted list of feature names

    """
    return sorted(
        functools.reduce(
            operator.concat,
            (feature_dictionary[key] for key in feature_dictionary.keys())
        )
    )


def str_in_sql(values):
    return ','.join("'{}'".format(value) for value in values)

def retry_if_db_error(exception):
    return isinstance(exception, sqlalchemy.exc.OperationalError)

DEFAULT_RETRY_KWARGS = {
    'retry_on_exception': retry_if_db_error,
    'wait_exponential_multiplier': 1000,  # wait 2^x*1000ms between each retry
    'stop_max_attempt_number': 14,
    # with this configuration, last wait will be ~2 hours
    # for a total of ~4.5 hours waiting
}

db_retry = retry(**DEFAULT_RETRY_KWARGS)

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
