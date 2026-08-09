"""POST /api/v1/transcribe field + response models.

``TranscribeResponse``: transcript, detected_language, duration_seconds, provider
(plus ``speech_detected`` and ``warnings``, which are what make the silence case
readable rather than merely empty).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from services.domain import Language, TranscriptionResult


class TranscribeResponse(BaseModel):
    """Result of transcribing one audio upload."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "transcript": "রোগীর তিন দিন ধরে জ্বর এবং মাথাব্যথা রয়েছে।",
                    "detected_language": "bn",
                    "duration_seconds": 8.64,
                    "provider": "mock-whisper",
                    "speech_detected": True,
                    "warnings": [],
                },
                {
                    "transcript": "",
                    "detected_language": "unknown",
                    "duration_seconds": 6.5,
                    "provider": "mock-whisper",
                    "speech_detected": False,
                    "warnings": [
                        "2 segment(s) were discarded as non-speech "
                        "(no_speech_probability above 0.6).",
                        "No speech detected: provider reported no_speech_probability "
                        "0.883 above the 0.6 threshold; no transcribable speech "
                        "remained after filtering.",
                    ],
                },
            ]
        }
    )

    transcript: str = Field(
        description=(
            "Recognized speech. Empty when no speech was found -- check "
            "`speech_detected` rather than treating an empty string as a failure."
        )
    )
    detected_language: Language = Field(
        description=(
            "Language actually detected: 'bn', 'en', or 'unknown'. Never guessed -- "
            "audio with no speech always reports 'unknown', and a provider language "
            "outside the supported set is reported as 'unknown' rather than coerced."
        )
    )
    duration_seconds: float = Field(
        description=(
            "Audio duration. Falls back to 0.0 with a warning when neither the "
            "provider nor the container reports one."
        )
    )
    provider: str = Field(
        description=(
            "Adapter that produced this result, e.g. 'mock-whisper' or "
            "'openai-whisper-1'. A mock result can never be labelled as a live one."
        )
    )
    speech_detected: bool = Field(
        default=True,
        description=(
            "False for silence or ambient noise. This is a successful 200 response, "
            "not an error: the request was processed correctly and the recording "
            "simply contains no speech."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal notes -- which silence signal fired, how many segments were "
            "discarded as hallucinated, whether the language was echoed from the "
            "request rather than detected."
        ),
    )

    @classmethod
    def from_domain(cls, result: TranscriptionResult) -> TranscribeResponse:
        return cls(
            transcript=result.transcript,
            detected_language=result.detected_language,
            duration_seconds=result.duration_seconds,
            provider=result.provider,
            speech_detected=result.speech_detected,
            warnings=list(result.warnings),
        )
