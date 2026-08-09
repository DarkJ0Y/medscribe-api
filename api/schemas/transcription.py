"""POST /api/v1/transcribe field + response models.

``TranscribeResponse``: transcript, detected_language, duration_seconds, provider
(plus ``speech_detected`` and ``warnings``, which are what make the silence case
readable rather than merely empty).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from services.domain import Language, TranscriptionResult
from services.wer import WordErrorRate


class WordErrorRateModel(BaseModel):
    """Transcription accuracy against a known reference.

    **Present only during fixture replay.** A live transcription has no reference
    to score against -- producing the transcript is the whole point of the request
    -- so this is ``null`` for every real provider call. It exists to
    regression-test the pipeline, not to monitor production accuracy.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "wer": 0.2308,
                "substitutions": 3,
                "deletions": 3,
                "insertions": 0,
                "hits": 20,
                "reference_words": 26,
                "hypothesis_words": 23,
                "errors": 6,
                "exact_match": False,
            }
        }
    )

    wer: float = Field(
        description=(
            "(substitutions + deletions + insertions) / reference_words. "
            "Unbounded above: a hypothesis longer than the reference can exceed 1.0 "
            "through insertions alone, so read it as a rate rather than a percentage "
            "of correctness."
        )
    )
    substitutions: int = Field(description="Reference words replaced by a different word.")
    deletions: int = Field(description="Reference words absent from the transcript.")
    insertions: int = Field(description="Transcript words with no reference counterpart.")
    hits: int = Field(description="Words matched exactly after normalization.")
    reference_words: int = Field(description="Word count of the reference, i.e. the denominator.")
    hypothesis_words: int = Field(description="Word count of the produced transcript.")
    errors: int = Field(description="substitutions + deletions + insertions.")
    exact_match: bool = Field(description="True when there were no errors at all.")

    @classmethod
    def from_domain(cls, score: WordErrorRate) -> WordErrorRateModel:
        return cls(
            wer=round(score.wer, 4),
            substitutions=score.substitutions,
            deletions=score.deletions,
            insertions=score.insertions,
            hits=score.hits,
            reference_words=score.reference_words,
            hypothesis_words=score.hypothesis_words,
            errors=score.errors,
            exact_match=score.is_exact_match,
        )


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
    word_error_rate: WordErrorRateModel | None = Field(
        default=None,
        description=(
            "Accuracy against a known reference. **Null for every live "
            "transcription** -- accuracy cannot be measured without a reference, and "
            "a provider asked to transcribe audio does not have one. Populated only "
            "when replaying a fixture that declares `reference_transcript`, which is "
            "how the pipeline is regression-tested."
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
            word_error_rate=(
                WordErrorRateModel.from_domain(result.word_error_rate)
                if result.word_error_rate is not None
                else None
            ),
        )
