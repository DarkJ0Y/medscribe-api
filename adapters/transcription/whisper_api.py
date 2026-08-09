"""RealTranscriptionAdapter -- OpenAI Whisper API.

Imported lazily by ``adapters.registry``; the ``openai`` SDK is an optional extra,
so a top-level import here is correct -- the ImportError surfaces in the registry's
guard as an actionable ConfigurationError. A self-hosted faster-whisper
implementation drops in as a sibling behind the same port.

What this adapter does *not* do, per the port contract: it does not map language
codes onto :class:`~services.domain.Language`, does not judge whether the audio was
silent, and does not suppress hallucinated text. It reports what the provider said
and lets :class:`~services.transcription_service.TranscriptionService` decide, so
those rules are tested once rather than once per provider.

Two provider realities worth knowing
------------------------------------
**Segment data requires whisper-1.** Only ``whisper-1`` supports
``response_format="verbose_json"``, which is what carries per-segment
``no_speech_prob`` and the clip ``duration``. The newer ``gpt-4o-transcribe``
family returns plain JSON -- text only. Configure one of those and the service
loses its provider-side silence signal, degrading to the amplitude check (WAV
only, per D12). The adapter logs a warning at construction rather than letting
that happen quietly.

**The API's own limit is 25 MB**, which is where the brief's ``MAX_AUDIO_BYTES``
default comes from. Our check rejects an oversized upload before it is sent.
"""

from __future__ import annotations

import logging
from typing import Any, Final

import openai
from openai import AsyncOpenAI

from services.domain import FilePayload, LanguageHint, RawTranscription, TranscriptSegment
from services.errors import ProviderTimeoutError, ProviderUnavailableError

logger = logging.getLogger(__name__)

#: Model families that support verbose_json (and therefore segments + duration).
_VERBOSE_CAPABLE_PREFIXES: Final[tuple[str, ...]] = ("whisper",)

_DEFAULT_CONTENT_TYPE: Final[str] = "application/octet-stream"


class RealTranscriptionAdapter:
    """:class:`~services.ports.TranscriptionPort` backed by the OpenAI audio API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._timeout = timeout_seconds
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._verbose = model.lower().startswith(_VERBOSE_CAPABLE_PREFIXES)
        if not self._verbose:
            logger.warning(
                "Model %r does not support verbose_json: no segment data, no "
                "no_speech_probability and no duration will be available. Silence "
                "detection will rely on audio amplitude alone, which is measurable "
                "only for PCM WAV. Use whisper-1 to keep the provider-side signal.",
                model,
                extra={"model": model},
            )

    @property
    def provider_name(self) -> str:
        # Includes the model so a response can never be mistaken for a different
        # backend, and so a model rollout is visible in stored results.
        return f"openai-{self._model}"

    async def transcribe(
        self,
        audio: FilePayload,
        language: LanguageHint,
    ) -> RawTranscription:
        request: dict[str, Any] = {
            "model": self._model,
            # The SDK accepts (filename, bytes, content_type). The filename matters:
            # the API infers the container format from its extension.
            "file": (
                audio.filename or "audio.wav",
                audio.data,
                audio.content_type or _DEFAULT_CONTENT_TYPE,
            ),
            "timeout": self._timeout,
        }
        if self._verbose:
            request["response_format"] = "verbose_json"
            request["timestamp_granularities"] = ["segment"]
        if language is not LanguageHint.AUTO:
            # ISO-639-1. Omitted entirely for AUTO so the model detects it.
            request["language"] = language.value

        try:
            response = await self._client.audio.transcriptions.create(**request)
        except openai.APITimeoutError as exc:
            # Must precede APIConnectionError -- it is a subclass.
            raise ProviderTimeoutError(
                f"Transcription timed out after {self._timeout}s.",
                provider=self.provider_name,
                details={"timeout_seconds": self._timeout},
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderUnavailableError(
                f"Could not reach the transcription provider: {exc}",
                provider=self.provider_name,
            ) from exc
        except openai.RateLimitError as exc:
            raise ProviderUnavailableError(
                "Transcription provider rate limit exceeded.",
                provider=self.provider_name,
                details={"status_code": getattr(exc, "status_code", None)},
            ) from exc
        except openai.APIStatusError as exc:
            raise ProviderUnavailableError(
                f"Transcription provider returned an error: {exc}",
                provider=self.provider_name,
                details={"status_code": getattr(exc, "status_code", None)},
            ) from exc
        except openai.OpenAIError as exc:
            # Catch-all for the SDK's own errors, so no provider-specific exception
            # type ever escapes into the domain.
            raise ProviderUnavailableError(
                f"Transcription provider failed: {exc}",
                provider=self.provider_name,
            ) from exc

        return to_raw_transcription(response, provider=self.provider_name)

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool.

        Called from the application lifespan on shutdown. Optional protocol --
        ``main.py`` invokes it only if present, so the mocks need no equivalent.
        """
        await self._client.close()


def to_raw_transcription(response: Any, *, provider: str) -> RawTranscription:
    """Map an OpenAI transcription response onto the domain type.

    Written against ``getattr`` rather than a concrete response class so the same
    function handles ``verbose_json`` (text + language + duration + segments) and
    plain ``json`` (text only), and so it can be unit-tested without the SDK.
    """
    segments = tuple(
        TranscriptSegment(
            start_seconds=float(_field(item, "start", 0.0)),
            end_seconds=float(_field(item, "end", 0.0)),
            text=str(_field(item, "text", "")),
            no_speech_probability=_optional_float(_field(item, "no_speech_prob", None)),
        )
        for item in (getattr(response, "segments", None) or [])
    )

    return RawTranscription(
        text=str(getattr(response, "text", "") or ""),
        provider=provider,
        detected_language=getattr(response, "language", None),
        duration_seconds=_optional_float(getattr(response, "duration", None)),
        no_speech_probability=_aggregate_no_speech(segments),
        segments=segments,
    )


def _aggregate_no_speech(segments: tuple[TranscriptSegment, ...]) -> float | None:
    """Collapse per-segment scores into one clip-level score.

    The **minimum** is used, deliberately. The service treats this value as "the
    whole clip is non-speech", and the minimum only crosses the threshold when even
    the *most* speech-like segment scored as non-speech. A mean would let one long
    silent stretch drag a clip containing real speech over the line and blank a
    valid transcript.

    Returns ``None`` when no segment carries a score, so the service knows the
    signal is unavailable rather than reading a fabricated 0.0 as confident speech.
    """
    scores = [s.no_speech_probability for s in segments if s.no_speech_probability is not None]
    return min(scores) if scores else None


def _field(item: Any, name: str, default: Any) -> Any:
    """Read a field from a pydantic model or a plain dict.

    The SDK returns model objects, but ``verbose_json`` segments arrive as dicts in
    some SDK versions. Supporting both avoids a version-pinning trap.
    """
    if isinstance(item, dict):
        return item.get(name, default)
    value = getattr(item, name, default)
    return default if value is None else value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
