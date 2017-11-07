import time

import pytest

from triage.timeout import timeout


def test_context_manager_success():
    with timeout(0.02):
        time.sleep(0.015)


def test_context_manager_failure():
    with pytest.raises(TimeoutError):
        with timeout(0.01):
            time.sleep(0.015)


def test_decorator_success():
    @timeout(0.01)
    def snooze(seconds):
        time.sleep(seconds)

    snooze(0.005)


def test_decorator_failure():
    @timeout(0.01)
    def snooze(seconds):
        time.sleep(seconds)

    with pytest.raises(TimeoutError):
        snooze(0.015)


def test_custom_exception():
    exc = RuntimeError("Overslept")

    with pytest.raises(RuntimeError) as exc_info:
        with timeout(0.01, exc):
            time.sleep(0.015)

    assert exc_info.value is exc


def test_custom_exception_class():
    with pytest.raises(RuntimeError):
        with timeout(0.01, RuntimeError):
            time.sleep(0.015)


def test_custom_exception_message():
    value = "Overslept"

    with pytest.raises(TimeoutError) as exc_info:
        with timeout(0.01, value=value):
            time.sleep(0.015)

    assert exc_info.value.args == (value,)


def test_custom_exception_values():
    value = (1, 2, 3)

    with pytest.raises(TimeoutError) as exc_info:
        with timeout(0.01, value=value):
            time.sleep(0.015)

    assert (
        exc_info.value.errno,
        exc_info.value.strerror,
        exc_info.value.filename,
    ) == value
