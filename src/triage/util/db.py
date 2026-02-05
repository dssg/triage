# coding: utf-8
import functools
import json

import sqlalchemy
import wrapt

from triage.logging import get_logger

logger = get_logger(__name__)

from contextlib import contextmanager
from datetime import date, datetime

from psycopg.types.range import DateRange, TimestamptzRange
from sqlalchemy import inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session


def serialize_to_database(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, date):
        return str(obj.isoformat())

    if isinstance(obj, (DateRange, TimestamptzRange)):
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

    def get_inspector(self):
        return inspect(self.__wrapped__)

    @classmethod
    def __reconstruct__(cls, url, creator, kwargs):
        return cls(url, creator=creator, **kwargs)


create_engine = functools.partial(SerializableDbEngine, json_serializer=json_dumps)


@contextmanager
def scoped_session(db_engine):
    """Provide a transactional scope around a series of operations."""
    # session = sessionmaker(db_engine, future=True)
    session = Session(
        bind=db_engine,
        future=True,
        expire_on_commit=False,
    )

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_for_update(db_engine, orm_class, primary_key):
    """Gets object from the database to updated it"""
    logger.spam(f"ORM class: {orm_class} with primary key {primary_key}")
    with scoped_session(db_engine) as session:
        obj = session.get(orm_class, primary_key)
        logger.spam(f"obj from get_for_update: {obj}")
        yield obj
        session.merge(obj)
