# coding: utf-8

import sqlalchemy
import wrapt
from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy.engine.url import make_url


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
