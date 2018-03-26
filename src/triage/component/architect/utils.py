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
def save_matrix_object(db_engine, matrix_object):
    """Saves a collection of SQLAlchemy model objects to the database using a COPY command

    Args:
        db_engine (sqlalchemy.engine)
        db_objects (list) SQLAlchemy model objects, corresponding to a valid table
    """
    with tempfile.TemporaryFile(mode='w+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            getattr(matrix_object, col.name)
            for col in matrix_object.__table__.columns
        ])
        f.seek(0)
        #postgres_copy.copy_from(f, type(matrix_object), db_engine, format='csv')

        conn = db_engine.raw_connection()
        cursor = conn.cursor()
        relation = '"model_metadata"."matrices"'
        formatted_flags = "(FORMAT 'csv')"
        copy = 'COPY {} FROM STDIN {}'.format(
            relation,
            formatted_flags
        )
        cursor.copy_expert(copy, f)
        conn.commit()
        conn.close()
