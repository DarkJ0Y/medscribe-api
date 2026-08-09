"""POST /api/v1/transcribe over the real ASGI stack."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

ENDPOINT = "/api/v1/transcribe"

RESPONSE_KEYS = {
    "transcript",
    "detected_language",
    "duration_seconds",
    "provider",
    "speech_detected",
    "warnings",
}


@pytest.mark.parametrize(
    ("fixture", "language", "expected_language", "duration"),
    [
        ("bn_prescription", "bn", "bn", 8.64),
        ("en_lab_query", "en", "en", 11.2),
        ("bn_en_code_switch", "auto", "bn", 9.41),
        ("en_lab_query", "auto", "en", 11.2),
    ],
)
def test_transcribes_speech(
    client: TestClient,
    audio_bytes: Any,
    fixture: str,
    language: str,
    expected_language: str,
    duration: float,
) -> None:
    response = client.post(
        ENDPOINT,
        files={"file": (f"{fixture}.wav", audio_bytes(fixture), "audio/wav")},
        data={"language": language},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS, "response contract changed"
    assert body["detected_language"] == expected_language
    assert body["duration_seconds"] == pytest.approx(duration)
    assert body["provider"] == "mock-whisper"
    assert body["speech_detected"] is True
    assert body["transcript"]
    assert response.headers["x-request-id"]


def test_language_field_defaults_to_auto(client: TestClient, audio_bytes: Any) -> None:
    response = client.post(
        ENDPOINT, files={"file": ("a.wav", audio_bytes("en_lab_query"), "audio/wav")}
    )

    assert response.status_code == 200
    assert response.json()["detected_language"] == "en"


@pytest.mark.parametrize("fixture", ["silence", "ambient_noise"])
def test_no_speech_returns_200_not_an_error(
    client: TestClient, audio_bytes: Any, fixture: str
) -> None:
    """The client did nothing wrong and the request was processed successfully; the
    recording simply contains no speech. A 4xx would be incorrect (D6)."""
    response = client.post(
        ENDPOINT, files={"file": (f"{fixture}.wav", audio_bytes(fixture), "audio/wav")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["speech_detected"] is False
    assert body["transcript"] == ""
    assert body["detected_language"] == "unknown"
    assert any("No speech detected" in warning for warning in body["warnings"])


def test_which_silence_signal_fires_for_which_fixture(
    client: TestClient, audio_bytes: Any
) -> None:
    """The asymmetry between the two silence signals, which is why both exist.

    Digital silence trips *both* checks -- RMS 0.000 is under the floor and the
    provider also scores it 0.974. Ambient noise trips only the provider score,
    because its RMS of 0.020 is deliberately ABOVE the 0.005 floor. Asserting that
    RMS does NOT fire for ambient noise is the point: it proves the provider signal
    is load-bearing rather than redundant.
    """

    def warnings_for(fixture: str) -> str:
        response = client.post(
            ENDPOINT, files={"file": (f"{fixture}.wav", audio_bytes(fixture), "audio/wav")}
        )
        return " ".join(response.json()["warnings"])

    silence = warnings_for("silence")
    assert "RMS" in silence
    assert "no_speech_probability" in silence

    noise = warnings_for("ambient_noise")
    assert "no_speech_probability" in noise
    assert "RMS" not in noise, (
        "amplitude must not fire for ambient noise -- if it does, the fixture no "
        "longer proves that the provider signal is necessary"
    )


def test_hallucinated_text_never_reaches_the_client(
    client: TestClient, audio_bytes: Any, transcription_fixture: Any
) -> None:
    """ambient_noise.wav records the real Whisper failure mode: a fluent, confident,
    entirely invented sentence over room noise."""
    invented = transcription_fixture("ambient_noise")["text"]
    assert invented, "fixture must contain hallucinated text"

    response = client.post(
        ENDPOINT, files={"file": ("ambient_noise.wav", audio_bytes("ambient_noise"), "audio/wav")}
    )

    assert response.status_code == 200
    assert response.json()["transcript"] == ""
    assert "Thank you" not in response.json()["transcript"]


@pytest.mark.parametrize(
    ("filename", "content_type", "data", "status", "code"),
    [
        ("virus.exe", "application/x-msdownload", b"MZ\x90\x00", 400, "unsupported_media_type"),
        ("scan.pdf", "application/pdf", b"%PDF-1.7", 400, "unsupported_media_type"),
        # An allowed extension contradicted by the content type (D8).
        ("a.wav", "image/png", b"\x89PNG\r\n\x1a\n", 400, "unsupported_media_type"),
        ("empty.wav", "audio/wav", b"", 400, "empty_upload"),
    ],
)
def test_rejects_bad_uploads(
    client: TestClient,
    filename: str,
    content_type: str,
    data: bytes,
    status: int,
    code: str,
) -> None:
    response = client.post(ENDPOINT, files={"file": (filename, data, content_type)})

    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_rejects_an_unsupported_language_value(client: TestClient, audio_bytes: Any) -> None:
    response = client.post(
        ENDPOINT,
        files={"file": ("a.wav", audio_bytes("silence"), "audio/wav")},
        data={"language": "de"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "request_validation_failed"
    assert any("language" in str(field["location"]) for field in body["error"]["details"]["fields"])


def test_rejects_an_oversized_upload_before_reading_the_body(
    settings: Any, audio_bytes: Any
) -> None:
    """Verified against a deliberately tiny limit rather than by allocating 25 MB.

    The Content-Length guard runs in middleware, before the multipart parser has
    consumed anything -- see the MaxBodySizeMiddleware docstring for what it does and
    does not protect against.
    """
    from main import create_app

    tiny = settings.model_copy(update={"max_audio_bytes": 2048, "max_image_bytes": 2048})
    with TestClient(create_app(tiny), raise_server_exceptions=False) as small_client:
        response = small_client.post(
            ENDPOINT, files={"file": ("big.wav", b"x" * 5000, "audio/wav")}
        )

    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "file_too_large"
    assert body["error"]["details"]["max_bytes"] == 2048
    assert body["request_id"]
