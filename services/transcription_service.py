"""Transcription orchestration.

Validates the audio payload against domain limits, delegates to a
:class:`~services.ports.TranscriptionPort`, resolves ``auto`` language detection,
and applies the silence / no-speech decision rules.

The no-speech decision uses three independent signals, because each one alone
demonstrably misses a case:

1. **RMS amplitude** below ``silence_rms_threshold`` -- catches genuinely quiet
   recordings. Measurable only for PCM WAV here (see :func:`_probe_wav`); for
   compressed formats it simply does not vote.
2. **The provider's own ``no_speech_probability``** above threshold -- catches the
   dangerous case where a model emits fluent, confident, invented text over room
   noise. Such audio is *not* quiet, so signal 1 cannot see it.
3. **An empty transcript** after noisy segments are discarded.

When no speech is found the result is HTTP-200-worthy: an empty transcript with
``speech_detected=False`` and a warning naming the signal that fired. It is not an
error -- see DECISIONS.md D6.
"""

from __future__ import annotations

import array
import io
import logging
import math
import wave
from typing import Final

from services.domain import (
    FilePayload,
    Language,
    LanguageHint,
    RawTranscription,
    TranscriptionResult,
    language_from_provider_code,
)
from services.ports import TranscriptionPort
from services.upload_policy import validate_upload

logger = logging.getLogger(__name__)

#: Cap on samples inspected when measuring amplitude. A 25 MB 44.1 kHz stereo WAV
#: holds ~6M samples; summing all of them in pure Python would add seconds of
#: latency to every request for no extra accuracy, so a strided subsample is used.
_MAX_RMS_SAMPLES: Final[int] = 200_000

_INT16_FULL_SCALE: Final[float] = 32768.0


class TranscriptionService:
    """Domain logic for ``POST /api/v1/transcribe``."""

    def __init__(
        self,
        port: TranscriptionPort,
        *,
        max_bytes: int,
        allowed_extensions: tuple[str, ...],
        allowed_content_types: tuple[str, ...],
        silence_rms_threshold: float,
        no_speech_probability_threshold: float,
    ) -> None:
        self._port = port
        self._max_bytes = max_bytes
        self._allowed_extensions = allowed_extensions
        self._allowed_content_types = allowed_content_types
        self._silence_rms_threshold = silence_rms_threshold
        self._no_speech_threshold = no_speech_probability_threshold

    async def transcribe(
        self,
        audio: FilePayload,
        language: LanguageHint = LanguageHint.AUTO,
    ) -> TranscriptionResult:
        validate_upload(
            audio,
            max_bytes=self._max_bytes,
            allowed_extensions=self._allowed_extensions,
            allowed_content_types=self._allowed_content_types,
        )

        raw = await self._port.transcribe(audio, language)
        duration_hint, rms = _probe_wav(audio.data)
        return self._interpret(raw, language, duration_hint=duration_hint, rms=rms)

    # ------------------------------------------------------------------ rules
    def _interpret(
        self,
        raw: RawTranscription,
        requested: LanguageHint,
        *,
        duration_hint: float | None,
        rms: float | None,
    ) -> TranscriptionResult:
        warnings: list[str] = []

        kept, dropped = self._filter_segments(raw)
        if dropped:
            warnings.append(
                f"{dropped} segment(s) were discarded as non-speech "
                f"(no_speech_probability above {self._no_speech_threshold})."
            )

        transcript = self._assemble_transcript(raw, kept, dropped)
        speech_detected = self._decide_speech(raw, transcript, rms, warnings)
        if not speech_detected:
            transcript = ""

        duration = self._resolve_duration(raw, duration_hint, warnings)
        detected = self._resolve_language(raw, requested, speech_detected, warnings)

        if not speech_detected:
            logger.info(
                "No speech detected in upload.",
                extra={
                    "provider": raw.provider,
                    "rms": rms,
                    "no_speech_probability": raw.no_speech_probability,
                },
            )

        return TranscriptionResult(
            transcript=transcript,
            detected_language=detected,
            duration_seconds=duration,
            provider=raw.provider,
            speech_detected=speech_detected,
            warnings=tuple(warnings),
        )

    def _filter_segments(self, raw: RawTranscription) -> tuple[list[str], int]:
        """Drop segments the provider itself scored as non-speech."""
        kept: list[str] = []
        dropped = 0
        for segment in raw.segments:
            probability = segment.no_speech_probability
            if probability is not None and probability > self._no_speech_threshold:
                dropped += 1
                continue
            kept.append(segment.text)
        return kept, dropped

    def _assemble_transcript(
        self,
        raw: RawTranscription,
        kept: list[str],
        dropped: int,
    ) -> str:
        """Prefer the provider's own text; rebuild from segments only if needed.

        Re-joining segments is lossy -- it normalizes the whitespace and
        punctuation the provider chose -- so it is done only when segments were
        actually discarded and the full text would therefore include noise.
        """
        if raw.segments and dropped:
            return " ".join(part.strip() for part in kept if part.strip()).strip()
        return raw.text.strip()

    def _decide_speech(
        self,
        raw: RawTranscription,
        transcript: str,
        rms: float | None,
        warnings: list[str],
    ) -> bool:
        signals: list[str] = []

        if rms is not None and rms < self._silence_rms_threshold:
            signals.append(
                f"audio amplitude (RMS {rms:.5f}) is below the silence threshold "
                f"{self._silence_rms_threshold}"
            )
        if (
            raw.no_speech_probability is not None
            and raw.no_speech_probability > self._no_speech_threshold
        ):
            signals.append(
                f"provider reported no_speech_probability {raw.no_speech_probability} "
                f"above the {self._no_speech_threshold} threshold"
            )
        if not transcript:
            signals.append("no transcribable speech remained after filtering")

        if not signals:
            return True

        warnings.append("No speech detected: " + "; ".join(signals) + ".")
        return False

    def _resolve_duration(
        self,
        raw: RawTranscription,
        duration_hint: float | None,
        warnings: list[str],
    ) -> float:
        if raw.duration_seconds is not None:
            return float(raw.duration_seconds)
        if duration_hint is not None:
            warnings.append(
                "Provider did not report a duration; measured it from the audio container."
            )
            return duration_hint
        warnings.append("Duration is unavailable for this upload and is reported as 0.0.")
        return 0.0

    def _resolve_language(
        self,
        raw: RawTranscription,
        requested: LanguageHint,
        speech_detected: bool,
        warnings: list[str],
    ) -> Language:
        if not speech_detected:
            # Claiming a language for audio containing no speech would be
            # fabrication, whatever the provider guessed.
            return Language.UNKNOWN

        detected = language_from_provider_code(raw.detected_language)
        if detected is not Language.UNKNOWN:
            return detected

        if requested is not LanguageHint.AUTO:
            warnings.append(
                "Provider did not report a detected language; echoing the requested "
                f"language {requested.value!r}, which is a caller assertion rather "
                "than a detection."
            )
            return Language(requested.value)

        if raw.detected_language:
            warnings.append(
                f"Provider reported an unsupported language {raw.detected_language!r}; "
                "reported as 'unknown'."
            )
        return Language.UNKNOWN


# ---------------------------------------------------------------------------
# Audio probing (stdlib only -- no provider SDK, no framework)
# ---------------------------------------------------------------------------
def _probe_wav(data: bytes) -> tuple[float | None, float | None]:
    """Return ``(duration_seconds, rms)`` for 16-bit PCM WAV, else ``(None, None)``.

    Only uncompressed 16-bit PCM is handled, on purpose. ``audioop`` would cover
    more widths but was removed in Python 3.13, and pulling in a decoder for MP3
    or WebM would mean a native dependency in the base image -- which would break
    the promise that the default deployment needs no heavy install. For those
    formats the amplitude signal simply abstains and the provider's
    ``no_speech_probability`` carries the decision.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
            if frame_rate <= 0 or frame_count <= 0:
                return None, None
            duration = frame_count / frame_rate
            if sample_width != 2:
                return duration, None
            frames = handle.readframes(frame_count)
    except (wave.Error, EOFError, OSError, ValueError):
        return None, None

    samples = array.array("h")
    usable = len(frames) - (len(frames) % 2)
    if usable <= 0:
        return duration, None
    samples.frombytes(frames[:usable])
    if not samples:
        return duration, None

    stride = max(1, len(samples) // _MAX_RMS_SAMPLES)
    window = samples[::stride] if stride > 1 else samples
    total = math.fsum(float(value) * float(value) for value in window)
    rms = math.sqrt(total / len(window)) / _INT16_FULL_SCALE

    logger.debug(
        "Probed WAV container.",
        extra={
            "duration_seconds": duration,
            "channels": channels,
            "frame_rate": frame_rate,
            "rms": rms,
            "samples_inspected": len(window),
        },
    )
    return duration, rms
