import contextlib
import signal


class timeout(contextlib.ContextDecorator):
    """Set a timeout (seconds) upon expiration of which an exception is
    raised.

    As a context manager:

        with timeout(0.5):
            ... work ...

    ...or decorating a function:

        @timeout(0.5)
        def work():
            ...

    The exception may be configured, either by passing an exception
    instance, or by specifying an overriding exception class and/or
    value(s):

        with timeout(0.5, RuntimeError("Work took too long")):
            ... work ...

        with timeout(0.5, RuntimeError, "Work took too long"):
            ... work ...

        with timeout(0.5, exc=RuntimeError):
            ... work ...

        with timeout(0.5, value="Work took too long"):
            ... work ...

        with timeout(0.5, value=(2, "Took too long", 'work.py')):
            ... work ...

    The timeout exception defaults to
    `TimeoutError("Operation timed out")`.

    Note: `timeout` is implemented via `signal`, and as such may not be
    applied outside of a process's main thread.

    """
    empty = object()
    timeout_exception = TimeoutError
    timeout_message = "Operation timed out"

    def __init__(self, timeout, exc=empty, value=empty):
        self.last_handler = None

        self.timeout = timeout

        if exc is self.empty:
            exc = self.timeout_exception
        elif not callable(exc) and value is not self.empty:
            raise TypeError("Unsupported invocation: cannot call "
                            "{!r} with value {!r}".format(exc, value))

        if value is self.empty:
            value = (self.timeout_message,)
        elif isinstance(value, str) or not hasattr(value, '__iter__'):
            value = (value,)

        self.exc = exc(*value) if callable(exc) else exc

    def __enter__(self):
        if self.timeout:
            self.last_handler = signal.signal(signal.SIGALRM, self.handle_signal)
            signal.setitimer(signal.ITIMER_REAL, self.timeout)

    def __exit__(self, *_excinfo):
        if self.timeout:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.last_handler)

    def handle_signal(self, _signum, _frame):
        raise self.exc
