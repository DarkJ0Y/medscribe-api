"""The cross-cutting HTTP contract: one error envelope, correlation ids, docs.

A client that can parse one error must be able to parse all of them, so every class
of failure -- domain error, request validation, routing, and an unexpected crash --
is checked against the same shape here.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from main import create_app

pytestmark = pytest.mark.integration


def _assert_envelope(body: dict[str, Any], code: str) -> None:
    assert set(body) == {"error", "request_id"}
    assert set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == code
    assert body["error"]["message"], "an error must explain itself"
    assert isinstance(body["error"]["details"], dict)
    assert body["request_id"], "every error must be traceable to the logs"


@pytest.mark.parametrize(
    ("method", "path", "kwargs", "status", "code"),
    [
        # domain error -> 4xx
        (
            "post",
            "/api/v1/transcribe",
            {"files": {"file": ("a.exe", b"MZ", "application/x-msdownload")}},
            400,
            "unsupported_media_type",
        ),
        # request validation (missing required file part)
        (
            "post",
            "/api/v1/transcribe",
            {"data": {"language": "bn"}},
            422,
            "request_validation_failed",
        ),
        # routing
        ("get", "/api/v1/does-not-exist", {}, 404, "not_found"),
        ("get", "/api/v1/transcribe", {}, 405, "method_not_allowed"),
        ("delete", "/health", {}, 405, "method_not_allowed"),
    ],
)
def test_every_failure_class_uses_the_same_envelope(
    client: TestClient,
    method: str,
    path: str,
    kwargs: dict[str, Any],
    status: int,
    code: str,
) -> None:
    response = getattr(client, method)(path, **kwargs)

    assert response.status_code == status
    _assert_envelope(response.json(), code)


def test_an_unhandled_exception_is_sanitized_but_traceable(
    settings: Settings, image_bytes: Any
) -> None:
    """A crash must not leak internals, yet must stay correlatable to the logs.

    The exception is converted by RequestIdMiddleware rather than Starlette's
    ServerErrorMiddleware, which is what lets the 500 carry X-Request-ID at all -- see
    DECISIONS.md D17.
    """
    secret = "db password is hunter2"

    class ExplodingService:
        async def extract(self, image: Any) -> Any:
            raise RuntimeError(f"secret internal detail: {secret}")

    app = create_app(settings)
    app.state.document_service = ExplodingService()

    with TestClient(app, raise_server_exceptions=False) as crashing:
        response = crashing.post(
            "/api/v1/documents/extract",
            files={"file": ("cbc_report.png", image_bytes("cbc_report"), "image/png")},
        )

    assert response.status_code == 500
    body = response.json()
    _assert_envelope(body, "internal_error")

    # Nothing internal escapes...
    assert secret not in response.text
    assert "RuntimeError" not in response.text
    assert "Traceback" not in response.text
    assert ".py" not in response.text
    # ...but the response is still traceable, in both the body and the header.
    assert body["request_id"] == response.headers["x-request-id"]


def test_correlation_ids_are_generated_and_honoured(client: TestClient) -> None:
    inbound = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
    assert inbound.headers["x-request-id"] == "trace-abc-123"

    first = client.get("/health").headers["x-request-id"]
    second = client.get("/health").headers["x-request-id"]
    assert first != second
    assert len(first) == 32  # uuid4 hex


def test_health_reports_the_adapter_mode(client: TestClient) -> None:
    """`adapter_mode` is surfaced so a mock deployment can never be mistaken for one
    backed by live models."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "medscribe",
        "version": "0.1.0",
        "adapter_mode": "mock",
    }


def test_openapi_documents_both_endpoints_and_their_failures(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    for path in ("/api/v1/transcribe", "/api/v1/documents/extract"):
        assert path in spec["paths"]
        responses = spec["paths"][path]["post"]["responses"]
        # Documented failure modes, not just the happy path.
        for status in ("400", "413", "422", "502", "504"):
            assert status in responses, f"{path} does not document {status}"

    assert "ErrorResponse" in spec["components"]["schemas"]
    # The language field is documented as a closed enum.
    schemas = spec["components"]["schemas"]
    language_enums = [
        schema["enum"]
        for schema in schemas.values()
        if isinstance(schema, dict) and schema.get("enum") and "bn" in schema.get("enum", [])
    ]
    assert any(set(enum) == {"bn", "en", "auto"} for enum in language_enums)
