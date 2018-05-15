import sqlalchemy
import wrapt


class SerializableDbEngine(wrapt.ObjectProxy):
    """A sqlalchemy engine that can be serialized across process boundaries.

    Works by saving all kwargs used to create the engine and reconstructs them later.
    As a result, the state won't be saved upon serialization/deserialization.
    """

    __slots__ = ('url', 'creator', 'kwargs')

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


create_engine = SerializableDbEngine
