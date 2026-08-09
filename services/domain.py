"""Framework-free domain models shared by services and adapters.

Free of FastAPI and of every HTTP concern, so an adapter can import these types
without dragging the web framework into the provider layer. Deliberately plain
dataclasses rather than Pydantic models: these objects are constructed by code
we control from data that has *already* been validated at the edge, so a second
validation pass would buy nothing and would couple the core to a library.

Two invariants matter more than the rest and are stated here because every
downstream module depends on them:

1. ``LabResult.raw_line`` is the exact, unmodified OCR line. Nothing in the
   pipeline may rewrite, strip or normalize it.
2. ``NumericValue.raw`` is likewise verbatim, and a ``NumericValue`` whose
   ``kind`` is ``UNPARSED`` carries no numbers at all rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# services.wer imports nothing from this module, so this direction is safe and
# keeps WordErrorRate visible as part of the domain vocabulary.
from services.wer import WordErrorRate

# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------


class LanguageHint(StrEnum):
    """What the caller asked for on the way in."""

    BN = "bn"
    EN = "en"
    AUTO = "auto"


class Language(StrEnum):
    """What we are prepared to claim on the way out.

    ``UNKNOWN`` is a first-class answer: a provider that declines to identify
    the language, or identifies one we do not support, is reported as unknown
    rather than defaulted to English.
    """

    BN = "bn"
    EN = "en"
    UNKNOWN = "unknown"


# Provider language codes are not standardized -- Whisper returns "bengali",
# some engines return BCP-47 tags. Map generously, but never fall through to a
# supported language: anything unrecognized becomes UNKNOWN.
_BN_CODES = frozenset({"bn", "ben", "bengali", "bangla", "bn-bd", "bn-in"})
_EN_CODES = frozenset({"en", "eng", "english", "en-us", "en-gb", "en-in"})


def language_from_provider_code(code: str | None) -> Language:
    """Normalize a provider's language code onto :class:`Language`."""
    if not code:
        return Language.UNKNOWN
    normalized = code.strip().lower().replace("_", "-")
    if normalized in _BN_CODES:
        return Language.BN
    if normalized in _EN_CODES:
        return Language.EN
    return Language.UNKNOWN


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilePayload:
    """An uploaded file, already read into memory by the API layer.

    One type serves audio and images: the fields are identical and the size and
    media-type policies that differ between them belong to the services, not to
    the container. ``content_type`` is optional because clients lie about it or
    omit it; ``filename`` may also be absent, so neither is trusted alone.
    """

    data: bytes
    filename: str | None = None
    content_type: str | None = None

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def extension(self) -> str:
        """Lowercased extension including the dot, or ``""`` if there is none."""
        if not self.filename or "." not in self.filename:
            return ""
        return "." + self.filename.rsplit(".", 1)[1].strip().lower()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One timed span of a transcript, when the provider exposes them."""

    start_seconds: float
    end_seconds: float
    text: str
    no_speech_probability: float | None = None


@dataclass(frozen=True, slots=True)
class RawTranscription:
    """What a :class:`~services.ports.TranscriptionPort` returns.

    Intentionally unopinionated -- ``detected_language`` is the provider's own
    string, not a :class:`Language` -- so that the mapping and the no-speech
    decision stay in the service where they can be tested, rather than being
    reimplemented by every adapter.
    """

    text: str
    provider: str
    detected_language: str | None = None
    duration_seconds: float | None = None
    no_speech_probability: float | None = None
    segments: tuple[TranscriptSegment, ...] = ()

    reference_transcript: str | None = None
    """Known-correct text for this audio, when one exists.

    Only a replay adapter can supply this -- a live provider is being asked to
    produce the transcript, so by definition it has nothing to compare against.
    Its presence is what lets the service report a word error rate; its absence is
    the normal production case.
    """


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """The domain answer for ``POST /api/v1/transcribe``."""

    transcript: str
    detected_language: Language
    duration_seconds: float
    provider: str
    speech_detected: bool = True
    warnings: tuple[str, ...] = ()

    word_error_rate: WordErrorRate | None = None
    """Accuracy against a known reference, when the adapter supplied one.

    ``None`` for every live transcription, because accuracy cannot be measured
    without a reference. Populated during fixture replay, which is what makes it
    useful for regression-testing the pipeline rather than for production
    monitoring.
    """


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OcrLine:
    """A single recognized line, exactly as the engine produced it."""

    text: str
    line_number: int
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class RawOcrResult:
    """What an :class:`~services.ports.OCRPort` returns."""

    lines: tuple[OcrLine, ...]
    provider: str
    mean_confidence: float | None = None


# ---------------------------------------------------------------------------
# Normalized values
# ---------------------------------------------------------------------------


class ValueKind(StrEnum):
    """How much structure the normalizer was able to recover."""

    SINGLE = "single"
    """A single quantity: ``5.4``, ``12,500``, ``1.2 x 10^3``."""

    BOUNDED = "bounded"
    """A one-sided bound: ``<0.5``, ``>= 200``."""

    RANGE = "range"
    """A two-sided interval: ``0.8 - 1.2``, ``13.0 to 17.0``."""

    UNPARSED = "unparsed"
    """Recognized as *something*, but not as any known numeric form.

    Carries no numbers. This is the deliberate outcome for ambiguous input --
    see DECISIONS.md D5 -- because a wrong clinical number that looks confident
    is worse than an absent one.
    """


class Comparator(StrEnum):
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="


@dataclass(frozen=True, slots=True)
class NumericValue:
    """A measurement, or an honest admission that it could not be parsed.

    ``raw`` is always the verbatim source text. Which of the numeric fields are
    populated is determined by ``kind``::

        SINGLE    -> value
        BOUNDED   -> comparator + value
        RANGE     -> low + high
        UNPARSED  -> nothing
    """

    raw: str
    kind: ValueKind
    value: float | None = None
    comparator: Comparator | None = None
    low: float | None = None
    high: float | None = None

    @property
    def is_parsed(self) -> bool:
        return self.kind is not ValueKind.UNPARSED

    @classmethod
    def unparsed(cls, raw: str) -> NumericValue:
        """Preserve ``raw`` verbatim without asserting any numeric reading."""
        return cls(raw=raw, kind=ValueKind.UNPARSED)


# ---------------------------------------------------------------------------
# Lab report
# ---------------------------------------------------------------------------


class ResultFlag(StrEnum):
    """Where a value sits relative to its reference range.

    Spelled out rather than the ``H``/``L`` printed on paper reports: the API is
    consumed by software, and ``"high"`` needs no legend. ``UNKNOWN`` covers the
    common cases of an absent reference range or an unparsed value -- it is not
    an error, just the absence of a comparison.
    """

    HIGH = "high"
    LOW = "low"
    NORMAL = "normal"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class LabResult:
    """One row of a lab report.

    ``raw_line`` is the contract with the caller: whatever normalization did or
    failed to do, the original OCR line is here to be audited against.
    """

    test_name: str
    value: NumericValue
    raw_line: str
    unit: str | None = None
    reference_range: NumericValue | None = None
    flag: ResultFlag = ResultFlag.UNKNOWN


@dataclass(frozen=True, slots=True)
class ReportMeta:
    """Header fields of a lab report. Every field is optional -- phone
    photographs crop, and a missing field is reported as missing.

    ``age`` stays a string on purpose: reports write "45 Y", "6 M", "3 Days".
    Coercing that to an integer would silently turn a six-month-old infant into
    a six-year-old child, so the verbatim text is preserved instead.
    """

    patient_name: str | None = None
    age: str | None = None
    sex: str | None = None
    report_date: str | None = None
    lab_name: str | None = None
    reference_no: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedReport:
    """The domain answer for ``POST /api/v1/documents/extract``."""

    meta: ReportMeta
    results: tuple[LabResult, ...] = ()
    provider: str = "unknown"
    ocr_mean_confidence: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
