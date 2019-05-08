# coding: utf-8

import csv
import sqlalchemy
import wrapt
from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy.engine.url import make_url
from retrying import retry
from functools import partial
from itertools import chain

import postgres_copy

from ohio import PipeTextIO

class SerializableDbEngine(wrapt.ObjectProxy):
    """A sqlalchemy engine that can be serialized across process boundaries.

    Works by saving all kwargs used to create the engine and reconstructs them later.  As a result, the state won't be saved upon serialization/deserialization.
    """

    __slots__ = ("url", "creator", "kwargs")

    def __init__(self, url, *, creator=sqlalchemy.create_engine, **kwargs):
        self.url = make_url(url)
        self.creator = creator
        self.kwargs = kwargs

        engine = creator(url, **kwargs)
        super().__init__(engine)

    def __reduce__(self):
        return (self.__reconstruct__, (self.url, self.creator, self.kwargs))

    def __reduce_ex__(self, protocol):
        # wrapt requires reduce_ex to be implemented
        return self.__reduce__()

    @classmethod
    def __reconstruct__(cls, url, creator, kwargs):
        return cls(url, creator=creator, **kwargs)


create_engine = SerializableDbEngine


@contextmanager
def scoped_session(db_engine):
    """Provide a transactional scope around a series of operations."""
    session = Session(bind=db_engine)
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_for_update(db_engine, orm_class, primary_key):
    with scoped_session(db_engine) as session:
        obj = session.query(orm_class).get(primary_key)
        yield obj
        session.merge(obj)


def retry_if_db_error(exception):
    return isinstance(exception, sqlalchemy.exc.OperationalError)


DEFAULT_RETRY_KWARGS = {
    "retry_on_exception": retry_if_db_error,
    "wait_exponential_multiplier": 1000,  # wait 2^x*1000ms between each retry
    "stop_max_attempt_number": 14,
    # with this configuration, last wait will be ~2 hours
    # for a total of ~4.5 hours waiting
}


db_retry = retry(**DEFAULT_RETRY_KWARGS)


def _write_csv(file_like, db_objects, type_of_object):
    writer = csv.writer(file_like, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    for db_object in db_objects:
        if type(db_object) != type_of_object:
            raise TypeError("Cannot copy collection of objects to db as they are not all "
                            f"of the same type. First object was {type_of_object} "
                            f"and later encountered a {type(db_object)}")
        writer.writerow(
            [getattr(db_object, col.name) for col in db_object.__table__.columns]
        )


@db_retry
def save_db_objects(db_engine, db_objects):
    """Saves a collection of SQLAlchemy model objects to the database using a COPY command

    Args:
        db_engine (sqlalchemy.engine)
        db_objects (iterable) SQLAlchemy model objects, corresponding to a valid table
    """
    db_objects = iter(db_objects)
    first_object = next(db_objects)
    type_of_object = type(first_object)

    with PipeTextIO(partial(
            _write_csv,
            db_objects=chain((first_object,), db_objects),
            type_of_object=type_of_object
    )) as pipe:
        postgres_copy.copy_from(pipe, type_of_object, db_engine, format="csv")
