# coding: utf-8

import sqlalchemy
import wrapt

import json

import functools

def fix_data(obj):
    """JSON serializer for objects not serializable by default json code"""
    from psycopg2.extras import DateRange, DateTimeRange
    from datetime import date, datetime

    if isinstance(obj, (datetime, date)):
        return str(obj.isoformat())

    if isinstance(obj, DateRange) or isinstance(obj, DateTimeRange):
        return f"[{obj.lower}, {obj.upper}]"

    return obj


def json_dumps(d):
    return json.dumps(d, default=fix_data)


class SerializableDbEngine(wrapt.ObjectProxy):
    """A sqlalchemy engine that can be serialized across process boundaries.

    Works by saving all kwargs used to create the engine and reconstructs them later.
    As a result, the state won't be saved upon serialization/deserialization.
    """

    __slots__ = ("url", "creator", "kwargs")

    def __init__(self, url, *, creator=sqlalchemy.create_engine, **kwargs):
        self.url = url
        self.creator = creator
        self.kwargs = kwargs

        engine = creator(url, **kwargs)
        super().__init__(engine)

    def __reduce__(self):
        return (self.__reconstruct__, (self.url, self.creator, self.kwargs))

    @classmethod
    def __reconstruct__(cls, url, creator, kwargs):
        return cls(url, creator=creator, **kwargs)


create_engine = functools.partial(SerializableDbEngine, json_serializer=json_dumps)
