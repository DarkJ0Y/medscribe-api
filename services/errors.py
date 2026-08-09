"""Framework-free domain exceptions.

Every error carries a stable machine-readable ``code`` and an optional
``details`` mapping. Neither carries an HTTP status code: the mapping from these
classes onto 400/413/415/502/504 lives in ``api/errors.py``, which is the only
module that knows HTTP exists. That split is what lets the same services back a
CLI, a queue consumer or a test harness unchanged.

Deliberately absent: a ``NoSpeechDetectedError``. Silent or ambient-noise-only
audio is a legitimate outcome, not a failure -- the spec asks for it to be
handled *gracefully*, so it is reported as a successful
:class:`~services.domain.TranscriptionResult` with ``speech_detected=False``
rather than raised. See DECISIONS.md D6.
"""

from __future__ import annotations

from typing import Any, ClassVar


class DomainError(Exception):
    """Base class for every expected failure in the domain.

    ``code`` is part of the public API contract -- clients branch on it -- so
    treat renaming one as a breaking change.
    """

    code: ClassVar[str] = "domain_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# Input validation (caller's fault)
# ---------------------------------------------------------------------------


class InputValidationError(DomainError):
    """An upload that cannot be processed as submitted."""

    code: ClassVar[str] = "invalid_input"


class EmptyUploadError(InputValidationError):
    """Zero-byte upload, or a multipart part with no file attached."""

    code: ClassVar[str] = "empty_upload"

    def __init__(self, message: str = "Uploaded file is empty.") -> None:
        super().__init__(message)


class FileTooLargeError(InputValidationError):
    """Upload exceeds the configured ceiling.

    Both sizes are reported so a client can tell how far over it was without a
    second round trip.
    """

    code: ClassVar[str] = "file_too_large"

    def __init__(self, *, size_bytes: int, max_bytes: int) -> None:
        super().__init__(
            f"Uploaded file is {size_bytes} bytes, which exceeds the "
            f"{max_bytes} byte limit.",
            details={"size_bytes": size_bytes, "max_bytes": max_bytes},
        )
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


class UnsupportedMediaTypeError(InputValidationError):
    """Extension or content type outside the allowlist."""

    code: ClassVar[str] = "unsupported_media_type"

    def __init__(
        self,
        *,
        content_type: str | None,
        extension: str | None,
        allowed: tuple[str, ...],
    ) -> None:
        seen = content_type or extension or "unknown"
        super().__init__(
            f"Unsupported media type {seen!r}. Allowed: {', '.join(allowed)}.",
            details={
                "content_type": content_type,
                "extension": extension,
                "allowed": list(allowed),
            },
        )
        self.content_type = content_type
        self.extension = extension
        self.allowed = allowed


class CorruptUploadError(InputValidationError):
    """Media type is allowed but the bytes are not decodable as that format."""

    code: ClassVar[str] = "corrupt_upload"


# ---------------------------------------------------------------------------
# Extraction outcomes (nobody's fault, but not a usable result)
# ---------------------------------------------------------------------------


class NotALabReportError(DomainError):
    """The image was read successfully but is not a lab report.

    Raised instead of returning a mostly-empty report so the caller gets a clean
    structured refusal rather than plausible-looking garbage assembled from
    whatever text happened to be in the photograph. ``details`` carries what we
    did see, which is what makes the response debuggable.
    """

    code: ClassVar[str] = "not_a_lab_report"

    def __init__(
        self,
        message: str = "The uploaded image does not appear to be a medical lab report.",
        *,
        lines_detected: int = 0,
        result_rows_detected: int = 0,
        rows_required: int | None = None,
    ) -> None:
        super().__init__(
            message,
            details={
                "lines_detected": lines_detected,
                "result_rows_detected": result_rows_detected,
                "rows_required": rows_required,
            },
        )


class UnreadableImageError(DomainError):
    """OCR returned nothing at all -- blank, black or hopelessly blurred."""

    code: ClassVar[str] = "unreadable_image"


# ---------------------------------------------------------------------------
# Provider / configuration (our fault)
# ---------------------------------------------------------------------------


class ProviderError(DomainError):
    """Base for failures originating in an adapter's upstream dependency."""

    code: ClassVar[str] = "provider_error"

    def __init__(self, message: str, *, provider: str, details: dict[str, Any] | None = None):
        merged = {"provider": provider, **(details or {})}
        super().__init__(message, details=merged)
        self.provider = provider


class ProviderUnavailableError(ProviderError):
    """Upstream refused, errored, or could not be reached."""

    code: ClassVar[str] = "provider_unavailable"


class ProviderTimeoutError(ProviderError):
    """Upstream exceeded the configured deadline."""

    code: ClassVar[str] = "provider_timeout"


class ConfigurationError(DomainError):
    """The service is misconfigured -- raised at wiring time, not per request.

    The canonical case is ``USE_MOCK_ADAPTERS=false`` without the ``real`` extra
    installed or without an API key. Failing loudly here beats a confusing
    ImportError from deep inside an adapter on the first live request.
    """

    code: ClassVar[str] = "configuration_error"
