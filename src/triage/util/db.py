# coding: utf-8

import sqlalchemy
import wrapt
from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy.engine.url import make_url

import json
import functools

from psycopg.types.range import DateRange, TimestampRange
from datetime import date, datetime


def serialize_to_database(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, date):
        return str(obj.isoformat())

    if isinstance(obj, (DateRange, TimestampRange)):
        return f"[{obj.lower}, {obj.upper}]"

    return obj


def json_dumps(d):
    return json.dumps(d, default=serialize_to_database)



class SerializableDbEngine(wrapt.ObjectProxy):
    """A sqlalchemy engine that can be serialized across process boundaries.

    Works by saving all kwargs used to create the engine and reconstructs them later.  As a result, the state won't be saved upon serialization/deserialization.
    """

    __slots__ = ("url", "creator", "kwargs")

    def __init__(self, url, *, creator=sqlalchemy.create_engine, **kwargs):
        original_url = make_url(url)
        if original_url.drivername in {"postgresql", "postgresql+psycopg2"}:
            normalized_url = original_url.set(drivername="postgresql+psycopg")
        else:
            normalized_url = original_url

        self.url = normalized_url
        self.creator = creator
        self.kwargs = kwargs

        # Pass URL object directly instead of str() to preserve password
        engine = creator(normalized_url, **kwargs)
        super().__init__(engine)

    def __reduce__(self):
        return (self.__reconstruct__, (self.url, self.creator, self.kwargs))

    def __reduce_ex__(self, protocol):
        # wrapt requires reduce_ex to be implemented
        return self.__reduce__()

    @classmethod
    def __reconstruct__(cls, url, creator, kwargs):
        return cls(url, creator=creator, **kwargs)


create_engine = functools.partial(SerializableDbEngine, json_serializer=json_dumps)

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
    """ Gets object from the database to updated it """
    with scoped_session(db_engine) as session:
        obj = session.query(orm_class).get(primary_key)
        yield obj
        session.merge(obj)
