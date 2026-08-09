"""Shared envelopes: ErrorResponse, ErrorDetail, HealthResponse.

Every failure the service emits -- domain error, request-validation failure,
routing 404, or an unexpected crash -- comes back in the *same* shape. A client
that can parse one error can parse all of them, and can branch on the stable
``error.code`` rather than on prose or on a status code that several distinct
failures share.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """The machine-readable half of an error response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "file_too_large",
                "message": "Uploaded file is 31457280 bytes, which exceeds the "
                "26214400 byte limit.",
                "details": {"size_bytes": 31457280, "max_bytes": 26214400},
            }
        }
    )

    code: str = Field(
        description=(
            "Stable identifier for the failure. Branch on this, not on the message. "
            "Renaming one is a breaking change."
        ),
        examples=["file_too_large", "not_a_lab_report", "unsupported_media_type"],
    )
    message: str = Field(description="Human-readable explanation. May change without notice.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured context for the failure -- observed sizes, allowed formats, "
            "how many rows were detected. Contents vary by code."
        ),
    )


class ErrorResponse(BaseModel):
    """Body returned with every non-2xx response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "not_a_lab_report",
                    "message": "The uploaded image contains text but does not appear "
                    "to be a medical lab report: 0 recognizable test row(s) were found "
                    "and at least 2 are required.",
                    "details": {
                        "lines_detected": 13,
                        "result_rows_detected": 0,
                        "rows_required": 2,
                    },
                },
                "request_id": "9f2c1e7b8a4d4c6fb0e3a1d5c7f90b21",
            }
        }
    )

    error: ErrorDetail
    request_id: str | None = Field(
        default=None,
        description=(
            "Correlation id for this request, also returned in the X-Request-ID "
            "header and attached to every log line the request produced. Quote it "
            "when reporting a problem."
        ),
    )


class HealthResponse(BaseModel):
    """Liveness/readiness payload, also used by the container healthcheck."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "app": "medscribe",
                "version": "0.1.0",
                "adapter_mode": "mock",
            }
        }
    )

    status: str = "ok"
    app: str
    version: str
    adapter_mode: str = Field(
        description=(
            "'mock' or 'real'. Surfaced deliberately: it must be impossible to "
            "mistake a mock deployment for one backed by live models."
        )
    )
