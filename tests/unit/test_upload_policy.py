"""The upload admission truth table (D8).

Neither the filename extension nor the content type is trustworthy alone: browsers
send ``application/octet-stream`` for valid audio, and an extension is entirely
caller-controlled. The rule is "satisfies one, contradicts neither", and each row
below is one line of that table.
"""

from __future__ import annotations

import pytest

from config.settings import Settings
from services.domain import FilePayload
from services.errors import (
    EmptyUploadError,
    FileTooLargeError,
    UnsupportedMediaTypeError,
)
from services.upload_policy import normalize_content_type, validate_upload

pytestmark = pytest.mark.unit

MAX = 1024


def _admit(settings: Settings, filename: str | None, content_type: str | None, data: bytes) -> None:
    validate_upload(
        FilePayload(data=data, filename=filename, content_type=content_type),
        max_bytes=MAX,
        allowed_extensions=settings.allowed_audio_extensions,
        allowed_content_types=settings.allowed_audio_content_types,
    )


@pytest.mark.parametrize(
    ("filename", "content_type", "why"),
    [
        ("a.wav", "audio/wav", "both agree"),
        ("a.WAV", "audio/wav", "extension case is irrelevant"),
        ("a.wav", "audio/wav; charset=binary", "parameters are stripped"),
        ("a.wav", "application/octet-stream", "generic type asserts nothing"),
        ("a.wav", "binary/octet-stream", "generic type asserts nothing"),
        ("a.wav", None, "no type sent at all"),
        ("a.wav", "", "empty type"),
        ("blob", "audio/mpeg", "no extension, but the type is allowed"),
        (None, "audio/flac", "no filename, but the type is allowed"),
        ("recording.m4a", "audio/x-m4a", "both agree"),
    ],
)
def test_accepts_valid_combinations(
    settings: Settings, filename: str | None, content_type: str | None, why: str
) -> None:
    _admit(settings, filename, content_type, b"xx")  # must not raise


@pytest.mark.parametrize(
    ("filename", "content_type", "why"),
    [
        ("a.wav", "image/png", "allowed extension CONTRADICTED by the type"),
        ("a.pdf", "audio/wav", "allowed type CONTRADICTED by the extension"),
        ("a.pdf", "application/pdf", "satisfies neither list"),
        ("virus.exe", "application/x-msdownload", "satisfies neither list"),
        (None, None, "nothing to go on"),
        ("noextension", None, "nothing to go on"),
    ],
)
def test_rejects_invalid_combinations(
    settings: Settings, filename: str | None, content_type: str | None, why: str
) -> None:
    with pytest.raises(UnsupportedMediaTypeError) as caught:
        _admit(settings, filename, content_type, b"xx")

    assert caught.value.code == "unsupported_media_type"
    # The error tells the caller what would have been accepted.
    assert ".wav" in caught.value.details["allowed"]


def test_checks_run_cheapest_and_most_specific_first(settings: Settings) -> None:
    """Emptiness beats size beats media type, so the reported error is the real one.

    A caller who sent an empty file with the wrong extension should be told the file
    is empty, not be sent chasing the extension.
    """
    with pytest.raises(EmptyUploadError):
        _admit(settings, "a.pdf", "application/pdf", b"")

    with pytest.raises(FileTooLargeError) as caught:
        _admit(settings, "a.pdf", "application/pdf", b"x" * (MAX + 1))
    assert caught.value.details == {"size_bytes": MAX + 1, "max_bytes": MAX}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Audio/WAV", "audio/wav"),
        ("audio/wav; charset=binary", "audio/wav"),
        ("  image/png  ", "image/png"),
        ("", None),
        (None, None),
    ],
)
def test_normalizes_content_type(raw: str | None, expected: str | None) -> None:
    assert normalize_content_type(raw) == expected
