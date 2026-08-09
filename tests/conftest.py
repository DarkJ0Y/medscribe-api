"""Shared fixtures: Settings override, TestClient, testdata loaders.

Settings are always constructed with ``_env_file=None`` so a developer's local
``.env`` can never change what the suite asserts. The mock adapters are the
default, which is what lets the whole suite run offline with no API key and no
model weights -- the same property ``docker compose up`` depends on.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from config.settings import PROJECT_ROOT, Settings
from main import create_app
from services.domain import FilePayload, OcrLine

TESTDATA = PROJECT_ROOT / "testdata"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture(scope="session")
def client(settings: Settings) -> Iterator[TestClient]:
    """ASGI client over the real application.

    ``raise_server_exceptions=False`` because an unhandled exception is converted
    to a sanitized 500 by ``RequestIdMiddleware`` (D17); the suite asserts on that
    response, which is what a caller actually sees.
    """
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# ---------------------------------------------------------------- testdata
@pytest.fixture(scope="session")
def audio_bytes() -> Any:
    def load(name: str) -> bytes:
        return (TESTDATA / "transcription" / "audio" / f"{name}.wav").read_bytes()

    return load


@pytest.fixture(scope="session")
def image_bytes() -> Any:
    def load(name: str) -> bytes:
        return (TESTDATA / "ocr" / "images" / f"{name}.png").read_bytes()

    return load


@pytest.fixture(scope="session")
def audio_payload(audio_bytes: Any) -> Any:
    def build(name: str) -> FilePayload:
        return FilePayload(
            data=audio_bytes(name), filename=f"{name}.wav", content_type="audio/wav"
        )

    return build


@pytest.fixture(scope="session")
def image_payload(image_bytes: Any) -> Any:
    def build(name: str) -> FilePayload:
        return FilePayload(
            data=image_bytes(name), filename=f"{name}.png", content_type="image/png"
        )

    return build


@pytest.fixture(scope="session")
def ocr_lines() -> Any:
    """Load an OCR fixture's lines directly, bypassing the adapter.

    Lets the parser be tested against the corpus without routing through replay,
    so a parser failure cannot be mistaken for a fixture-resolution failure.
    """

    def load(name: str) -> list[OcrLine]:
        payload = json.loads(
            (TESTDATA / "ocr" / "responses" / f"{name}.json").read_text(encoding="utf-8")
        )
        return [
            OcrLine(
                text=item["text"],
                line_number=item["line_number"],
                confidence=item["confidence"],
            )
            for item in payload["lines"]
        ]

    return load


@pytest.fixture(scope="session")
def transcription_fixture() -> Any:
    def load(name: str) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(
            (TESTDATA / "transcription" / "responses" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )
        return payload

    return load


@pytest.fixture(scope="session")
def manifest() -> Any:
    def load(kind: str) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(
            (TESTDATA / kind / "manifest.json").read_text(encoding="utf-8")
        )
        return payload

    return load


@pytest.fixture(scope="session")
def testdata_dir() -> Path:
    return TESTDATA
