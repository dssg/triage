"""Logging utilities wrapping loguru while keeping legacy semantics."""

from __future__ import annotations

import logging
import sys
from typing import Any

from icecream import ic
from loguru import logger as _logger

__all__ = ["configure_logging", "get_logger", "LoguruAdapter", "ic"]


_CUSTOM_LEVELS = (
    ("SPAM", 5, "<blue>"),
    ("VERBOSE", 9, "<cyan>"),
    ("NOTICE", 25, "<magenta>"),
)

for name, number, color in _CUSTOM_LEVELS:
    try:
        _logger.level(name)
    except ValueError:
        _logger.level(name, number, color)


def _format_message(message: str, args: tuple[Any, ...]) -> str:
    if not args:
        return message
    try:
        return message % args
    except (TypeError, ValueError):
        try:
            return message.format(*args)
        except Exception:
            return " ".join(str(item) for item in (message, *args))


class LoguruAdapter:
    """Mimic the verboselogs logger API while delegating to loguru."""

    def __init__(self, name: str | None = None) -> None:
        self._logger = _logger.bind(module=name) if name else _logger

    def _log(self, method: str, message: str, *args: Any, **kwargs: Any) -> None:
        formatted = _format_message(message, args)
        getattr(self._logger, method)(formatted, **kwargs)

    def _log_custom(self, level: str, message: str, *args: Any, **kwargs: Any) -> None:
        formatted = _format_message(message, args)
        self._logger.log(level, formatted, **kwargs)

    def trace(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("trace", message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("debug", message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("info", message, *args, **kwargs)

    def success(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("success", message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("warning", message, *args, **kwargs)

    warn = warning

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("error", message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log("critical", message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        formatted = _format_message(message, args)
        self._logger.opt(exception=True).error(formatted, **kwargs)

    def spam(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_custom("SPAM", message, *args, **kwargs)

    def verbose(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_custom("VERBOSE", message, *args, **kwargs)

    def notice(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_custom("NOTICE", message, *args, **kwargs)

    def log(self, level: str, message: str, *args: Any, **kwargs: Any) -> None:
        formatted = _format_message(message, args)
        self._logger.log(level, formatted, **kwargs)

    def bind(self, **kwargs: Any) -> "LoguruAdapter":
        adapter = LoguruAdapter()
        adapter._logger = self._logger.bind(**kwargs)
        return adapter


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - integration glue
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Bind the module name from the logging record to match our format
        _logger.bind(module=record.name).opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(default_level: str = "INFO") -> None:
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=default_level,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:<7}</level> | "
            "<cyan>{extra[module]}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    ic.configureOutput(
        includeContext=True,
        prefix="",
        outputFunction=lambda text: _logger.opt(depth=1).debug(text),
    )


def get_logger(name: str | None = None) -> LoguruAdapter:
    return LoguruAdapter(name)
