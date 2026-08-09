"""Domain exception -> structured JSON HTTP response translation.

Owns the mapping ``services.errors.*`` -> HTTP status codes and installs the
app-wide exception handlers. Services raise framework-free errors; only this
module knows a status code exists. That is what lets the same services back a CLI
or a queue consumer without carrying HTTP semantics around.

Four handlers are installed so that *every* failure exits through
:class:`~api.schemas.common.ErrorResponse` -- domain errors, request-validation
failures, routing 404s and unexpected crashes alike. A client that can parse one
error can parse all of them.

Status codes are plain integers rather than ``starlette.status`` constants on
purpose: Starlette has begun renaming them (``HTTP_422_UNPROCESSABLE_ENTITY`` is
already deprecated in favour of ``HTTP_422_UNPROCESSABLE_CONTENT``), and the
numbers are what the HTTP contract actually specifies.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.schemas.common import ErrorResponse
from config.logging import get_request_id
from services.errors import (
    ConfigurationError,
    CorruptUploadError,
    DomainError,
    EmptyUploadError,
    FileTooLargeError,
    InputValidationError,
    NotALabReportError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnreadableImageError,
    UnsupportedMediaTypeError,
)

logger = logging.getLogger(__name__)

BAD_REQUEST: Final[int] = 400
NOT_FOUND: Final[int] = 404
METHOD_NOT_ALLOWED: Final[int] = 405
PAYLOAD_TOO_LARGE: Final[int] = 413
UNPROCESSABLE: Final[int] = 422
INTERNAL_ERROR: Final[int] = 500
BAD_GATEWAY: Final[int] = 502
GATEWAY_TIMEOUT: Final[int] = 504

#: Domain error -> HTTP status. Looked up along the MRO, so a subclass added later
#: inherits its parent's status instead of silently falling through to 500.
#:
#: Note on 400 vs 415: ``UnsupportedMediaTypeError`` maps to 400 because the brief
#: specifies 400/413 for rejected uploads. 415 Unsupported Media Type would be the
#: more precise code and is a one-line change here -- the error body is identical
#: either way, and clients are expected to branch on ``error.code`` rather than on
#: the status. See DECISIONS.md D16.
_STATUS_BY_TYPE: Final[dict[type[DomainError], int]] = {
    # Caller's fault -- 4xx
    EmptyUploadError: BAD_REQUEST,
    UnsupportedMediaTypeError: BAD_REQUEST,
    CorruptUploadError: BAD_REQUEST,
    InputValidationError: BAD_REQUEST,
    FileTooLargeError: PAYLOAD_TOO_LARGE,
    # Well-formed request, unprocessable content -- 422
    NotALabReportError: UNPROCESSABLE,
    UnreadableImageError: UNPROCESSABLE,
    # Our fault -- 5xx
    ProviderTimeoutError: GATEWAY_TIMEOUT,
    ProviderUnavailableError: BAD_GATEWAY,
    ConfigurationError: INTERNAL_ERROR,
    DomainError: INTERNAL_ERROR,
}

_ERROR_DESCRIPTIONS: Final[dict[int, str]] = {
    BAD_REQUEST: "Empty upload, or a media type outside the allowlist.",
    PAYLOAD_TOO_LARGE: "Upload exceeds the configured size limit.",
    UNPROCESSABLE: "Well-formed request whose content could not be processed.",
    INTERNAL_ERROR: "Unexpected server error.",
    BAD_GATEWAY: "Upstream provider failed.",
    GATEWAY_TIMEOUT: "Upstream provider exceeded its deadline.",
}


def status_for(exc: DomainError) -> int:
    """Resolve the status code for ``exc``, walking its MRO."""
    for klass in type(exc).__mro__:
        if klass in _STATUS_BY_TYPE:
            return _STATUS_BY_TYPE[klass]  # type: ignore[index]
    return INTERNAL_ERROR


def request_id_for(request: Request | None = None) -> str | None:
    """Resolve the correlation id, preferring the ASGI scope over the ContextVar.

    The scope is authoritative because Starlette's ``ServerErrorMiddleware`` runs
    *outside* user middleware: when an unhandled exception reaches it, the
    ContextVar has already been reset by ``RequestIdMiddleware``'s ``finally``
    block, while the scope still carries the id. Falling back to the ContextVar
    covers callers that have no ``Request`` -- notably the size guard, which runs
    before any route is matched.
    """
    if request is not None:
        scoped = request.scope.get("state", {}).get("request_id")
        if isinstance(scoped, str) and scoped:
            return scoped
    return get_request_id()


def error_body(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build the response envelope.

    Shared with :mod:`api.middleware`, which cannot raise a domain error because it
    runs before a route is resolved.
    """
    return {
        "error": {"code": code, "message": message, "details": details or {}},
        "request_id": request_id if request_id is not None else get_request_id(),
    }


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    *,
    request: Request | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_body(code, message, details, request_id=request_id_for(request)),
    )


def openapi_errors(*codes: int) -> dict[int | str, dict[str, Any]]:
    """Reusable OpenAPI ``responses`` fragments, so the docs list what can go wrong."""
    return {
        code: {"description": _ERROR_DESCRIPTIONS.get(code, "Error"), "model": ErrorResponse}
        for code in codes
    }


def install_exception_handlers(app: FastAPI) -> None:
    """Register the four handlers that guarantee a uniform error envelope."""

    @app.exception_handler(DomainError)
    async def _domain_error(request: Request, exc: DomainError) -> JSONResponse:
        code = status_for(exc)
        # 5xx is our problem and deserves a stack trace; 4xx is routine client
        # behaviour and would only add noise at exception level.
        if code >= INTERNAL_ERROR:
            logger.exception("Domain error surfaced as %s.", code, extra={"error_code": exc.code})
        else:
            logger.info(
                "Rejected request: %s",
                exc.code,
                extra={"error_code": exc.code, "status_code": code, "path": request.url.path},
            )
        return error_response(code, exc.code, exc.message, exc.details, request=request)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """FastAPI's own 422, re-shaped into our envelope.

        This is what an invalid ``language`` field produces. ``details.fields``
        keeps pydantic's per-field diagnostics, which are genuinely useful, while
        the outer shape stays identical to every other error.
        """
        fields = [
            {
                "location": list(error.get("loc", ())),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        return error_response(
            UNPROCESSABLE,
            "request_validation_failed",
            "The request could not be validated. See details.fields.",
            {"fields": fields},
            request=request,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """404s, 405s and anything raised as an HTTPException."""
        code_name = {
            NOT_FOUND: "not_found",
            METHOD_NOT_ALLOWED: "method_not_allowed",
        }.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code_name, str(exc.detail), request=request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Last resort, and normally *not* the path taken.

        ``RequestIdMiddleware`` catches unhandled exceptions first, because it is
        the innermost layer that can still attach ``X-Request-ID`` to the response.
        This handler exists for an app assembled without that middleware -- a bare
        ``create_app`` variant, or a future entrypoint that mounts the routers
        directly -- so that a crash can never escape as an unstructured response.

        Either way the exception is logged in full while the response deliberately
        carries no exception text, module path or traceback: those leak internals
        to callers. The ``request_id`` is what connects the sanitized response to
        the full log entry, and it is read from the ASGI scope because by the time
        Starlette's ServerErrorMiddleware reaches this handler the ContextVar has
        already been reset.
        """
        logger.exception(
            "Unhandled exception.",
            extra={"path": request.url.path, "method": request.method},
        )
        return error_response(
            INTERNAL_ERROR,
            "internal_error",
            "An unexpected internal error occurred. Quote the request_id when reporting it.",
            request=request,
        )
