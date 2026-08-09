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
    "word_error_rate",
}

#: Expected WER per clinical fixture, pinned so a regression in normalization,
#: alignment or segment filtering fails loudly rather than drifting quietly.
#: (wer, substitutions, deletions, insertions, reference_words)
CLINICAL_WER = {
    "en_clinical_cardiac": (0.0, 0, 0, 0, 26),
    "en_clinical_hypertension": (7 / 26, 3, 4, 0, 26),
    "en_clinical_oncology": (8 / 26, 3, 0, 5, 26),
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


@pytest.mark.parametrize("fixture", sorted(CLINICAL_WER))
def test_reports_word_error_rate_for_clinical_dictation(
    client: TestClient, audio_bytes: Any, transcription_fixture: Any, fixture: str
) -> None:
    response = client.post(
        ENDPOINT,
        files={"file": (f"{fixture}.wav", audio_bytes(fixture), "audio/wav")},
        data={"language": "en"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["speech_detected"] is True
    assert body["detected_language"] == "en"

    expected_wer, subs, dels, ins, reference_words = CLINICAL_WER[fixture]
    score = body["word_error_rate"]
    assert score is not None, "a fixture with a reference_transcript must be scored"
    assert score["wer"] == pytest.approx(expected_wer, abs=5e-5)
    assert score["substitutions"] == subs
    assert score["deletions"] == dels
    assert score["insertions"] == ins
    assert score["reference_words"] == reference_words
    assert score["errors"] == subs + dels + ins
    assert score["exact_match"] is (subs + dels + ins == 0)

    # The score must describe the transcript the caller actually received, so the
    # reference in the fixture has to be the one that was used.
    assert transcription_fixture(fixture)["reference_transcript"]


def test_clean_dictation_scores_a_perfect_wer(client: TestClient, audio_bytes: Any) -> None:
    """The baseline. If normalization or segment filtering starts mangling clean
    input, this is the case that catches it before the noisy ones muddy the signal."""
    response = client.post(
        ENDPOINT,
        files={
            "file": ("en_clinical_cardiac.wav", audio_bytes("en_clinical_cardiac"), "audio/wav")
        },
    )

    score = response.json()["word_error_rate"]
    assert score["wer"] == 0.0
    assert score["exact_match"] is True
    assert score["hits"] == score["reference_words"] == 26


def test_semantically_perfect_numerals_still_score_errors(
    client: TestClient, audio_bytes: Any
) -> None:
    """"180 over 110" for "one hundred eighty over one hundred ten" reads perfectly to
    a human and is still a transcription error. WER measures transcription, not
    comprehension, and this fixture is here to keep that explicit rather than
    surprising."""
    response = client.post(
        ENDPOINT,
        files={
            "file": (
                "en_clinical_hypertension.wav",
                audio_bytes("en_clinical_hypertension"),
                "audio/wav",
            )
        },
    )

    body = response.json()
    assert "180 over 110" in body["transcript"]
    assert body["word_error_rate"]["wer"] > 0.25
    # Every error is a numeral or abbreviation difference; none is a lost word.
    assert body["word_error_rate"]["insertions"] == 0


@pytest.mark.parametrize("fixture", ["bn_prescription", "en_lab_query", "silence"])
def test_word_error_rate_is_null_without_a_reference(
    client: TestClient, audio_bytes: Any, fixture: str
) -> None:
    """The production case. A live provider is asked to produce the transcript, so it
    has nothing to score against -- claiming a WER there would be fabrication."""
    response = client.post(
        ENDPOINT, files={"file": (f"{fixture}.wav", audio_bytes(fixture), "audio/wav")}
    )

    assert response.status_code == 200
    assert response.json()["word_error_rate"] is None


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
