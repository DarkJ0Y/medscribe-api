"""MockTranscriptionAdapter -- replays saved responses from testdata/.

No GPU, no network, no model download: this is the adapter that makes
``docker compose up`` work on a clean clone. Fixture selection is handled by
:class:`adapters.replay.FixtureLibrary`; see that module for the resolution order.

Note what this adapter deliberately does *not* do. It does not map
``detected_language`` onto :class:`~services.domain.Language`, does not decide
whether the audio was silent, and does not suppress the hallucinated text in the
``ambient_noise`` fixture. All of that is the service's job (Step 4), and keeping
it there is what lets one set of rules be tested once instead of once per
provider. The mock's only responsibility is to hand back what a provider said.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.replay import FixtureLibrary
from services.domain import FilePayload, LanguageHint, RawTranscription, TranscriptSegment

_PROVIDER = "mock-whisper"


class MockTranscriptionAdapter:
    """Offline :class:`~services.ports.TranscriptionPort` backed by fixtures."""

    def __init__(self, fixtures_dir: Path) -> None:
        self._library = FixtureLibrary(fixtures_dir, provider=_PROVIDER)

    @property
    def provider_name(self) -> str:
        return _PROVIDER

    async def transcribe(
        self,
        audio: FilePayload,
        language: LanguageHint,
    ) -> RawTranscription:
        fixture, payload = await self._library.resolve(
            audio.data,
            audio.filename,
            language_key=language.value,
        )

        return RawTranscription(
            text=self._library.require(payload, "text", fixture),
            # Always this adapter's own name, never payload["provider"]. A fixture
            # must not be able to label itself "openai-whisper-1" and have a
            # replayed response pass for a live model call.
            provider=self.provider_name,
            detected_language=payload.get("detected_language"),
            duration_seconds=payload.get("duration_seconds"),
            no_speech_probability=payload.get("no_speech_probability"),
            segments=self._parse_segments(payload.get("segments") or []),
            # Only a replay adapter can know the correct answer. Forwarding it is
            # what lets the service report a word error rate; the real adapters
            # leave this None, because a live provider has nothing to compare
            # against.
            reference_transcript=payload.get("reference_transcript"),
        )

    def _parse_segments(self, raw: list[dict[str, Any]]) -> tuple[TranscriptSegment, ...]:
        return tuple(
            TranscriptSegment(
                start_seconds=float(item.get("start_seconds", 0.0)),
                end_seconds=float(item.get("end_seconds", 0.0)),
                text=str(item.get("text", "")),
                no_speech_probability=(
                    float(item["no_speech_probability"])
                    if item.get("no_speech_probability") is not None
                    else None
                ),
            )
            for item in raw
        )
