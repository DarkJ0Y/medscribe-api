"""Pure-ASGI middleware: correlation ids and an early body-size guard.

Written against the raw ASGI interface rather than Starlette's
``BaseHTTPMiddleware`` on purpose. ``BaseHTTPMiddleware`` runs the downstream app
in a separate task, which breaks ``ContextVar`` propagation -- exactly what the
correlation id relies on -- and it wraps the request body in a stream that must be
consumed before the size can be inspected. Both middlewares here need to act
*before* the body is read, so they operate on ``scope`` directly.

Why the size guard lives here at all
------------------------------------
Starlette's multipart parser consumes the **entire** request body before an
endpoint function is ever called. By the time a route handler could inspect
``UploadFile.size``, a 1 GB upload has already been received and spooled. So a
size check in the handler is a policy check, not a resource guard.

This middleware closes that gap the only way it can be closed in-process: by
rejecting on ``Content-Length`` before a single body byte is read. It is a
backstop, not a substitute for a reverse proxy -- see the class docstring.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.errors import error_body
from config.logging import new_request_id, reset_request_id, set_request_id

logger = logging.getLogger(__name__)

_JSON_CONTENT_TYPE = b"application/json"


async def _send_json(send: Send, status_code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", _JSON_CONTENT_TYPE),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestIdMiddleware:
    """Bind a correlation id to the request context and echo it in the response.

    An inbound id is honoured so a trace can span services; otherwise one is
    generated. The ``ContextVar`` token is always reset, because a worker task is
    reused across requests and a leaked id would attach the wrong trace to the
    next caller's logs.

    This middleware is also where an unhandled exception is converted into the
    error envelope. That is not scope creep -- it is the innermost point that can
    still add a response header. Starlette's ``ServerErrorMiddleware`` sits outside
    all user middleware, so a 500 it produces never passes through the ``send``
    wrapper below and cannot carry ``X-Request-ID``. Handling the exception here
    keeps the correlation id in the header *and* the body for every response the
    service emits, with no exception. See DECISIONS.md D17.
    """

    def __init__(self, app: ASGIApp, *, header_name: str = "X-Request-ID") -> None:
        self.app = app
        self._header = header_name.lower().encode("latin-1")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = self._read_header(scope)
        request_id = inbound or new_request_id()
        token = set_request_id(request_id)
        encoded = request_id.encode("latin-1")

        # Also stash it in the ASGI scope, not only the ContextVar. Starlette's
        # ServerErrorMiddleware sits OUTSIDE user middleware, so the handler for an
        # unhandled exception runs after the `finally` below has already reset the
        # ContextVar -- and a 500 is precisely when a caller needs the correlation
        # id. The scope travels with the request, so api/errors.py can still read it
        # there. See DECISIONS.md D17.
        scope.setdefault("state", {})["request_id"] = request_id

        response_started = False

        async def send_with_header(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != self._header
                ]
                headers.append((self._header, encoded))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        except Exception:
            if response_started:
                # Headers are already on the wire; there is no valid response left
                # to send. Let it propagate so the server closes the connection.
                raise
            logger.exception(
                "Unhandled exception.",
                extra={"path": scope.get("path"), "method": scope.get("method")},
            )
            await _send_json(
                send_with_header,
                500,
                error_body(
                    "internal_error",
                    "An unexpected internal error occurred. Quote the request_id "
                    "when reporting it.",
                ),
            )
        finally:
            reset_request_id(token)

    def _read_header(self, scope: Scope) -> str | None:
        for key, value in scope.get("headers", ()):
            if key.lower() == self._header:
                # Bound the length: this value reaches log lines and response
                # headers, so an unbounded caller-supplied string is a liability.
                candidate = value.decode("latin-1", errors="replace").strip()[:64]
                return candidate or None
        return None


class MaxBodySizeMiddleware:
    """Reject oversized requests on ``Content-Length``, before the body is read.

    Deliberate limitations, since a guard that is trusted for more than it does is
    worse than no guard:

    * A request with no ``Content-Length`` (chunked transfer encoding) passes
      through. The per-endpoint check in ``services/upload_policy.py`` still
      enforces the real limit, just after buffering.
    * The limit applied here is the **larger** of the audio and image limits,
      because the route -- and therefore which limit applies -- is not yet known.
      The precise per-endpoint limit is enforced downstream.

    The actual defence against a hostile multi-gigabyte upload is
    ``client_max_body_size`` in nginx or its equivalent at the ingress. This
    middleware makes the common case cheap and honest, nothing more.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = self._content_length(scope)
        if declared is not None and declared > self.max_bytes:
            logger.info(
                "Rejected oversized request on Content-Length.",
                extra={
                    "content_length": declared,
                    "max_bytes": self.max_bytes,
                    "path": scope.get("path"),
                },
            )
            await _send_json(
                send,
                413,
                error_body(
                    "file_too_large",
                    f"Request body is {declared} bytes, which exceeds the "
                    f"{self.max_bytes} byte limit.",
                    {"size_bytes": declared, "max_bytes": self.max_bytes},
                ),
            )
            return

        await self.app(scope, receive, send)

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        for key, value in scope.get("headers", ()):
            if key.lower() == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    # A malformed header is not a size we can trust; let the
                    # downstream policy check handle the request.
                    return None
        return None


#: Convenience alias for the ASGI callable signature used above.
Handler = Callable[[Scope, Receive, Send], Awaitable[None]]
