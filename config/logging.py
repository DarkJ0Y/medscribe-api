"""JSON logging setup + per-request correlation id.

The correlation id lives in a :class:`~contextvars.ContextVar` rather than being
threaded through call signatures, so a service can be logged without being handed
an HTTP concern. ``api/`` sets it per request (Step 5); everything below reads it
implicitly. That is what keeps ``services/`` free of framework types while still
producing traceable logs.

Structured output is the point: these logs are meant to be shipped and queried,
and a lab value or a provider latency buried in an interpolated string is not
queryable.
"""

from __future__ import annotations

import json
import logging
import logging.config
import uuid
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from config.settings import Settings

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

#: ``logging.LogRecord`` attributes that are structure, not payload. Anything on
#: a record outside this set arrived via ``logger.info(..., extra={...})`` and is
#: merged into the JSON object.
_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def new_request_id() -> str:
    return uuid.uuid4().hex


def set_request_id(request_id: str) -> Token[str | None]:
    """Bind ``request_id`` to the current context.

    Returns the token so the caller can :func:`reset_request_id`; without that,
    ids leak between requests served by the same worker task.
    """
    return _request_id.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


def get_request_id() -> str | None:
    return _request_id.get()


class JsonFormatter(logging.Formatter):
    """Renders each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if (request_id := get_request_id()) is not None:
            payload["request_id"] = request_id

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        # default=str keeps a stray Path, Decimal or dataclass from turning a log
        # call into an unhandled TypeError at the worst possible moment.
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    """Install the JSON formatter on the root and uvicorn loggers.

    uvicorn's loggers are reconfigured explicitly with ``propagate: false``:
    leaving them alone would emit every line twice, once plain and once JSON.
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": f"{__name__}.JsonFormatter"}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"handlers": ["console"], "level": settings.log_level},
            "loggers": {
                name: {
                    "handlers": ["console"],
                    "level": settings.log_level,
                    "propagate": False,
                }
                for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
            },
        }
    )
