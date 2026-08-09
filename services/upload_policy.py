"""Upload admission rules, shared by both services.

Kept in one module because the two endpoints must agree on what "too large" and
"unsupported" mean, and because the extension/content-type interaction (D8) is
subtle enough that two independent implementations would eventually disagree.

The admission rule, stated once
-------------------------------
An upload is accepted when it **satisfies one** allowlist and **contradicts
neither**:

===================== ===================== =========================
filename extension    content type          outcome
===================== ===================== =========================
``.wav`` (allowed)    ``audio/wav``         accept
``.wav`` (allowed)    ``application/...``   accept -- generic type, no claim
``.wav`` (allowed)    ``image/png``         reject -- direct contradiction
(none)                ``audio/wav``         accept -- nothing to contradict
``.pdf``              ``audio/wav``         reject -- extension contradicts
``.pdf``              ``application/pdf``   reject -- satisfies neither
(none)                (none)                reject -- nothing to go on
===================== ===================== =========================

Neither signal is trusted alone: browsers send ``application/octet-stream`` for
perfectly valid audio, and a filename extension is entirely caller-controlled.
"""

from __future__ import annotations

from typing import Final

from services.domain import FilePayload
from services.errors import (
    EmptyUploadError,
    FileTooLargeError,
    UnsupportedMediaTypeError,
)

#: Content types that assert nothing about the format, so they cannot contradict
#: an otherwise-acceptable extension.
_GENERIC_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "",
        "application/octet-stream",
        "binary/octet-stream",
        "application/binary",
        "*/*",
    }
)


def normalize_content_type(raw: str | None) -> str | None:
    """Lower-case and drop parameters (``audio/wav; charset=binary``)."""
    if raw is None:
        return None
    base = raw.split(";", 1)[0].strip().lower()
    return base or None


def validate_upload(
    payload: FilePayload,
    *,
    max_bytes: int,
    allowed_extensions: tuple[str, ...],
    allowed_content_types: tuple[str, ...],
) -> None:
    """Admit or reject an upload.

    Checks run cheapest-first, and the order is also the order of increasing
    specificity, so the error a caller receives names the first real problem:
    emptiness, then size, then media type.

    Raises:
        EmptyUploadError: zero bytes.
        FileTooLargeError: over ``max_bytes``.
        UnsupportedMediaTypeError: fails the admission rule above.
    """
    if payload.size_bytes == 0:
        raise EmptyUploadError()

    if payload.size_bytes > max_bytes:
        raise FileTooLargeError(size_bytes=payload.size_bytes, max_bytes=max_bytes)

    extension = payload.extension
    content_type = normalize_content_type(payload.content_type)

    extension_allowed = bool(extension) and extension in allowed_extensions
    content_type_allowed = content_type is not None and content_type in allowed_content_types
    content_type_is_generic = content_type is None or content_type in _GENERIC_CONTENT_TYPES

    satisfies = extension_allowed or content_type_allowed
    contradicts = (bool(extension) and not extension_allowed) or (
        not content_type_is_generic and not content_type_allowed
    )

    if not satisfies or contradicts:
        raise UnsupportedMediaTypeError(
            content_type=content_type,
            extension=extension or None,
            allowed=allowed_extensions,
        )
