import io
from ohio import IOClosed


class StreamBytesIOBase(io.BufferedIOBase):
    """Readable file-like abstract base class.
    Concrete classes may implemented method `__next_chunk__` to return
    chunks (or all) of the bytes to be read.
    """
    def __init__(self):
        self._remainder = ''

    def __next_chunk__(self):
        raise NotImplementedError("StreamBytesIOBase subclasses must implement __next_chunk__")

    def readable(self):
        if self.closed:
            raise IOClosed()

        return True

    def _read1(self, size=None):
        while not self._remainder:
            try:
                self._remainder = self.__next_chunk__()
            except StopIteration:
                break

        result = self._remainder[:size]
        self._remainder = self._remainder[len(result):]

        return result

    def read(self, size=None):
        if self.closed:
            raise IOClosed()

        if size is not None and size < 0:
            size = None

        result = b''

        while size is None or size > 0:
            content = self._read1(size)
            if not content:
                break

            if size is not None:
                size -= len(content)

            result += content

        return result

    def readline(self):
        if self.closed:
            raise IOClosed()

        result = ''

        while True:
            index = self._remainder.find('\n')
            if index == -1:
                result += self._remainder
                try:
                    self._remainder = self.__next_chunk__()
                except StopIteration:
                    self._remainder = ''
                    break
            else:
                result += self._remainder[:(index + 1)]
                self._remainder = self._remainder[(index + 1):]
                break

        return result


class IteratorBytesIO(StreamBytesIOBase):
    """Readable file-like interface for iterable byte streams."""

    def __init__(self, iterable):
        super().__init__()
        self.__iterator__ = iter(iterable)

    def __next_chunk__(self):
        return next(self.__iterator__)
