"""Service orchestration: the silence rules and the three extraction outcomes.

Stub ports are used where a fixture cannot produce the situation -- a provider that
reports speech but declines to name a language, for instance, which no recorded
fixture does.
"""

from __future__ import annotations

from typing import Any

import pytest

from config.settings import Settings
from services.document_service import DocumentExtractionService
from services.domain import (
    FilePayload,
    Language,
    LanguageHint,
    RawOcrResult,
    RawTranscription,
    TranscriptSegment,
)
from services.errors import NotALabReportError, UnreadableImageError
from services.transcription_service import TranscriptionService

pytestmark = pytest.mark.unit


class StubTranscriptionPort:
    """Returns a canned RawTranscription so a specific provider shape can be tested."""

    provider_name = "stub-provider"

    def __init__(self, response: RawTranscription) -> None:
        self._response = response
        self.received_language: LanguageHint | None = None

    async def transcribe(self, audio: FilePayload, language: LanguageHint) -> RawTranscription:
        self.received_language = language
        return self._response


def _service(port: Any, settings: Settings) -> TranscriptionService:
    return TranscriptionService(
        port,
        max_bytes=settings.max_audio_bytes,
        allowed_extensions=settings.allowed_audio_extensions,
        allowed_content_types=settings.allowed_audio_content_types,
        silence_rms_threshold=settings.silence_rms_threshold,
        no_speech_probability_threshold=settings.no_speech_probability_threshold,
    )


def _document_service(port: Any, settings: Settings) -> DocumentExtractionService:
    return DocumentExtractionService(
        port,
        max_bytes=settings.max_image_bytes,
        allowed_extensions=settings.allowed_image_extensions,
        allowed_content_types=settings.allowed_image_content_types,
        min_lab_rows=settings.min_lab_rows_for_report,
    )


# ---------------------------------------------------------------- silence
async def test_speech_is_reported_normally(
    client: Any, settings: Settings, audio_payload: Any
) -> None:
    service = _service(client.app.state.transcription_adapter, settings)
    result = await service.transcribe(audio_payload("bn_prescription"), LanguageHint.BN)

    assert result.speech_detected is True
    assert result.detected_language is Language.BN
    assert result.duration_seconds == pytest.approx(8.64)
    assert "জ্বর" in result.transcript
    assert result.warnings == ()


async def test_digital_silence_is_caught_by_amplitude(
    client: Any, settings: Settings, audio_payload: Any
) -> None:
    """silence.wav is normalized to RMS 0.000, below the 0.005 floor."""
    service = _service(client.app.state.transcription_adapter, settings)
    result = await service.transcribe(audio_payload("silence"), LanguageHint.AUTO)

    assert result.speech_detected is False
    assert result.transcript == ""
    assert result.detected_language is Language.UNKNOWN
    # Duration is still reported -- the file exists, it just has no speech.
    assert result.duration_seconds == pytest.approx(4.0)
    assert any("RMS" in warning for warning in result.warnings)


async def test_hallucination_over_noise_is_suppressed(
    client: Any, settings: Settings, audio_payload: Any, transcription_fixture: Any
) -> None:
    """The case that justifies having two independent silence signals.

    ambient_noise.wav is normalized to RMS 0.020 -- deliberately ABOVE the 0.005
    floor -- so the amplitude check cannot catch it. Its recorded transcript is the
    real Whisper failure mode: a fluent, confident, entirely invented sentence over
    room noise. Only the provider's no_speech_probability catches it, and the
    invented text must not reach the caller.
    """
    fixture = transcription_fixture("ambient_noise")
    assert fixture["text"], "fixture must contain hallucinated text for this test to mean anything"

    service = _service(client.app.state.transcription_adapter, settings)
    result = await service.transcribe(audio_payload("ambient_noise"), LanguageHint.AUTO)

    assert result.speech_detected is False
    assert result.transcript == ""
    assert fixture["text"] not in result.transcript
    # It was the provider score that fired, NOT amplitude -- assert both directions,
    # so this cannot pass for the wrong reason.
    assert any("no_speech_probability" in warning for warning in result.warnings)
    assert not any("RMS" in warning for warning in result.warnings)
    assert any("discarded as non-speech" in warning for warning in result.warnings)


# --------------------------------------------------------------- language
@pytest.mark.parametrize(
    ("provider_code", "expected"),
    [
        ("bengali", Language.BN),
        ("bn-BD", Language.BN),
        ("BEN", Language.BN),
        ("en", Language.EN),
        ("english", Language.EN),
        ("klingon", Language.UNKNOWN),
        ("zh", Language.UNKNOWN),
        (None, Language.UNKNOWN),
    ],
)
async def test_maps_provider_language_codes(
    settings: Settings, provider_code: str | None, expected: Language
) -> None:
    port = StubTranscriptionPort(
        RawTranscription(
            text="some speech",
            provider="stub-provider",
            detected_language=provider_code,
            duration_seconds=3.0,
            no_speech_probability=0.01,
        )
    )
    payload = FilePayload(data=b"not-a-wav", filename="a.wav", content_type="audio/wav")
    result = await _service(port, settings).transcribe(payload, LanguageHint.AUTO)

    assert result.detected_language is expected


async def test_echoed_language_is_labelled_as_a_caller_assertion(settings: Settings) -> None:
    """When the provider declines to name a language but the caller asserted one, the
    hint is echoed -- and the response says it was an assertion, not a detection."""
    port = StubTranscriptionPort(
        RawTranscription(
            text="কিছু কথা",
            provider="stub-provider",
            detected_language=None,
            duration_seconds=None,
            no_speech_probability=0.02,
        )
    )
    payload = FilePayload(data=b"not-a-wav", filename="a.wav", content_type="audio/wav")
    result = await _service(port, settings).transcribe(payload, LanguageHint.BN)

    assert result.detected_language is Language.BN
    assert any("caller assertion" in warning for warning in result.warnings)
    # Duration was unavailable and is reported as 0.0 with a warning, not invented.
    assert result.duration_seconds == 0.0
    assert any("Duration is unavailable" in warning for warning in result.warnings)


async def test_partial_segment_filtering_keeps_the_real_speech(settings: Settings) -> None:
    """A clip that is part speech, part noise keeps the speech and drops the noise."""
    port = StubTranscriptionPort(
        RawTranscription(
            text="real speech here AND INVENTED NOISE",
            provider="stub-provider",
            detected_language="en",
            duration_seconds=6.0,
            no_speech_probability=0.05,
            segments=(
                TranscriptSegment(0.0, 3.0, "real speech here", 0.02),
                TranscriptSegment(3.0, 6.0, "AND INVENTED NOISE", 0.95),
            ),
        )
    )
    payload = FilePayload(data=b"not-a-wav", filename="a.wav", content_type="audio/wav")
    result = await _service(port, settings).transcribe(payload, LanguageHint.AUTO)

    assert result.speech_detected is True
    assert result.transcript == "real speech here"
    assert "INVENTED NOISE" not in result.transcript
    assert any("1 segment(s) were discarded" in warning for warning in result.warnings)


# ------------------------------------------------------- document outcomes
async def test_extracts_a_lab_report(
    client: Any, settings: Settings, image_payload: Any
) -> None:
    service = _document_service(client.app.state.ocr_adapter, settings)
    report = await service.extract(image_payload("cbc_report"))

    assert len(report.results) == 11
    assert report.provider == "mock-tesseract"
    assert report.ocr_mean_confidence == pytest.approx(88.7)


async def test_refuses_a_non_lab_image_with_debuggable_detail(
    client: Any, settings: Settings, image_payload: Any
) -> None:
    service = _document_service(client.app.state.ocr_adapter, settings)

    with pytest.raises(NotALabReportError) as caught:
        await service.extract(image_payload("non_lab_receipt"))

    assert caught.value.code == "not_a_lab_report"
    # What was actually seen, so the refusal can be debugged rather than argued with.
    assert caught.value.details["lines_detected"] == 13
    assert caught.value.details["result_rows_detected"] == 0
    assert caught.value.details["rows_required"] == settings.min_lab_rows_for_report


async def test_distinguishes_unreadable_from_not_a_report(
    client: Any, settings: Settings, image_payload: Any
) -> None:
    """Two different problems with two different fixes: retake the photograph, or
    photograph a different document. Collapsing them would misdirect the caller."""
    service = _document_service(client.app.state.ocr_adapter, settings)

    with pytest.raises(UnreadableImageError) as caught:
        await service.extract(image_payload("blank_page"))

    assert caught.value.code == "unreadable_image"
    assert "Retake" in str(caught.value)


async def test_an_empty_ocr_result_is_data_not_a_provider_failure(settings: Settings) -> None:
    """The port contract: an adapter returns zero lines rather than raising, and the
    service is the single place that decides what that means."""

    class EmptyOCRPort:
        provider_name = "stub-ocr"

        async def extract_lines(self, image: FilePayload) -> RawOcrResult:
            return RawOcrResult(lines=(), provider="stub-ocr", mean_confidence=None)

    payload = FilePayload(data=b"\x89PNG", filename="a.png", content_type="image/png")
    with pytest.raises(UnreadableImageError):
        await _document_service(EmptyOCRPort(), settings).extract(payload)
