"""RealOCRAdapter -- Tesseract via pytesseract, line-level output.

Preferred over a VLM on the primary path because ``raw_line`` must be verbatim OCR
text and a generative model may silently rewrite it. See DECISIONS.md D3.

The whitespace problem, and why this adapter is more than a wrapper
------------------------------------------------------------------
``services/report_parser.py`` segments a line into columns on **runs of two or
more spaces**, which is how a lab report's column gaps appear in text. But
``image_to_data`` returns *word boxes* -- text plus pixel coordinates, with no
spacing at all. Joining those words with single spaces would collapse

    Haemoglobin          11.2        g/dl          13.0 - 17.0

into ``Haemoglobin 11.2 g/dl 13.0 - 17.0``, a single cell the parser cannot
segment. Every row would then fail to parse, and every real report would be
rejected as "not a lab report" -- while the mock adapter kept passing, because its
fixtures already contain the spacing.

So :func:`reconstruct_lines` rebuilds the gaps from geometry: it estimates a
character width from the word boxes and converts each inter-word pixel gap into a
proportional number of spaces. That is the load-bearing part of this module, and
it is a pure function precisely so it can be tested without Tesseract installed.

``image_to_string`` would preserve some spacing, but it yields no per-word
confidences and no reliable line grouping, so the geometry is worth reconstructing.
"""

from __future__ import annotations

import asyncio
import io
import logging
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Final

import pytesseract
from PIL import Image, UnidentifiedImageError

from services.domain import FilePayload, OcrLine, RawOcrResult
from services.errors import (
    CorruptUploadError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

#: ``--psm 6`` treats the page as a single uniform block of text, which keeps
#: tabular rows intact. The default (3, full auto page segmentation) tends to split
#: a report's columns into separate blocks, scrambling reading order.
_TESSERACT_CONFIG: Final[str] = "--psm 6"

#: Fallback when no word box is usable for estimating character width.
_FALLBACK_CHAR_WIDTH: Final[float] = 10.0

#: Ceiling on reconstructed spacing, so one wildly misplaced box cannot emit a
#: thousand-space line.
_MAX_GAP_SPACES: Final[int] = 40

_PROVIDER: Final[str] = "tesseract"


class RealOCRAdapter:
    """:class:`~services.ports.OCRPort` backed by a local Tesseract install."""

    def __init__(
        self,
        *,
        languages: str,
        timeout_seconds: float,
        tesseract_cmd: str | None = None,
    ) -> None:
        if tesseract_cmd:
            # Module-level global in pytesseract; set once at construction.
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._languages = languages
        self._timeout = timeout_seconds

    @property
    def provider_name(self) -> str:
        return _PROVIDER

    async def extract_lines(self, image: FilePayload) -> RawOcrResult:
        decoded = _decode_image(image)

        try:
            # Tesseract is a blocking subprocess; running it inline would stall the
            # event loop for the whole recognition.
            data = await asyncio.to_thread(
                pytesseract.image_to_data,
                decoded,
                lang=self._languages,
                output_type=pytesseract.Output.DICT,
                config=_TESSERACT_CONFIG,
                timeout=self._timeout,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise ProviderUnavailableError(
                "The tesseract binary was not found. Install tesseract-ocr, or set "
                "TESSERACT_CMD to its full path.",
                provider=self.provider_name,
            ) from exc
        except RuntimeError as exc:
            # pytesseract signals its own timeout as a bare RuntimeError.
            if "timeout" in str(exc).lower():
                raise ProviderTimeoutError(
                    f"OCR timed out after {self._timeout}s.",
                    provider=self.provider_name,
                    details={"timeout_seconds": self._timeout},
                ) from exc
            raise ProviderUnavailableError(
                f"OCR failed: {exc}", provider=self.provider_name
            ) from exc
        except pytesseract.TesseractError as exc:
            raise ProviderUnavailableError(
                f"Tesseract returned an error: {exc}",
                provider=self.provider_name,
                details={"languages": self._languages},
            ) from exc
        except OSError as exc:
            raise ProviderUnavailableError(
                f"Could not run tesseract: {exc}", provider=self.provider_name
            ) from exc

        lines = reconstruct_lines(data)
        confidences = [line.confidence for line in lines if line.confidence is not None]

        logger.debug(
            "Tesseract recognition complete.",
            extra={"lines": len(lines), "languages": self._languages},
        )

        # An empty result is DATA, not a failure -- the service turns it into
        # UnreadableImageError. See the OCRPort contract.
        return RawOcrResult(
            lines=lines,
            provider=self.provider_name,
            mean_confidence=round(statistics.fmean(confidences), 2) if confidences else None,
        )


def _decode_image(image: FilePayload) -> Image.Image:
    """Decode the upload, converting to greyscale.

    Raises :class:`~services.errors.CorruptUploadError` -- not a provider error --
    because undecodable bytes are the caller's problem. This adapter is the first
    component able to tell: the media-type allowlist (D8) accepts a file on its
    extension and content type, neither of which proves the bytes are a real image.
    """
    try:
        decoded = Image.open(io.BytesIO(image.data))
        decoded.load()
    except UnidentifiedImageError as exc:
        raise CorruptUploadError(
            "The uploaded file could not be decoded as an image. It may be "
            "truncated, or not actually the format its name and content type claim."
        ) from exc
    except (OSError, ValueError) as exc:
        raise CorruptUploadError(f"The uploaded image could not be read: {exc}") from exc

    # Greyscale only. Deliberately no deskew, threshold or upscale: each is a
    # quality/latency trade-off that needs measuring against real photographs
    # rather than guessing, and a bad threshold destroys faint text outright.
    return decoded.convert("L")


def reconstruct_lines(
    data: Mapping[str, Sequence[Any]],
    *,
    max_gap_spaces: int = _MAX_GAP_SPACES,
) -> tuple[OcrLine, ...]:
    """Rebuild spaced text lines from Tesseract word boxes.

    ``data`` is the dict from ``image_to_data(output_type=Output.DICT)``: parallel
    lists keyed ``text``, ``conf``, ``left``, ``width``, ``block_num``, ``par_num``,
    ``line_num``.

    Words are grouped by (block, paragraph, line) in first-appearance order, then
    each inter-word pixel gap is converted into ``round(gap / char_width)`` spaces
    -- at least one -- so column boundaries survive as the multi-space runs the
    parser needs.
    """
    words = _usable_words(data)
    if not words:
        return ()

    char_width = _estimate_char_width(words)

    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for word in words:
        grouped.setdefault(word["key"], []).append(word)

    lines: list[OcrLine] = []
    for index, group in enumerate(grouped.values(), start=1):
        group.sort(key=lambda item: item["left"])
        text = _join_with_gaps(group, char_width, max_gap_spaces)
        if not text.strip():
            continue
        confidences = [item["conf"] for item in group]
        lines.append(
            OcrLine(
                text=text,
                line_number=index,
                confidence=round(statistics.fmean(confidences), 2) if confidences else None,
            )
        )

    # Renumber after dropping blanks so line_number stays contiguous and 1-based,
    # which is what the report parser's adjacency rules assume.
    return tuple(
        OcrLine(text=line.text, line_number=number, confidence=line.confidence)
        for number, line in enumerate(lines, start=1)
    )


def _usable_words(data: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Extract non-blank, positively-scored word boxes.

    Tesseract emits ``conf == -1`` for structural rows that carry no text; keeping
    them would pull the mean confidence down and insert phantom gaps.
    """
    texts = data.get("text") or []
    count = len(texts)
    words: list[dict[str, Any]] = []

    for position in range(count):
        text = str(_at(data, "text", position, ""))
        if not text.strip():
            continue
        conf = _to_float(_at(data, "conf", position, -1.0))
        if conf is None or conf < 0:
            continue
        left = _to_float(_at(data, "left", position, 0.0)) or 0.0
        width = _to_float(_at(data, "width", position, 0.0)) or 0.0
        words.append(
            {
                "text": text,
                "conf": conf,
                "left": left,
                "width": width,
                "key": (
                    int(_to_float(_at(data, "block_num", position, 0)) or 0),
                    int(_to_float(_at(data, "par_num", position, 0)) or 0),
                    int(_to_float(_at(data, "line_num", position, 0)) or 0),
                ),
            }
        )
    return words


def _estimate_char_width(words: Sequence[Mapping[str, Any]]) -> float:
    """Median per-character width across all word boxes.

    The median rather than the mean: a single mis-boxed word -- one glyph assigned a
    200-pixel box -- would otherwise inflate the estimate and silently collapse
    every column gap to one space.
    """
    widths = [
        float(word["width"]) / len(str(word["text"]))
        for word in words
        if word["width"] and str(word["text"])
    ]
    if not widths:
        return _FALLBACK_CHAR_WIDTH
    median = statistics.median(widths)
    return median if median > 0 else _FALLBACK_CHAR_WIDTH


def _join_with_gaps(
    group: Sequence[Mapping[str, Any]],
    char_width: float,
    max_gap_spaces: int,
) -> str:
    parts: list[str] = []
    previous_right: float | None = None

    for word in group:
        left = float(word["left"])
        if previous_right is not None:
            gap_pixels = left - previous_right
            spaces = 1
            if gap_pixels > 0:
                spaces = max(1, min(max_gap_spaces, round(gap_pixels / char_width)))
            parts.append(" " * spaces)
        parts.append(str(word["text"]))
        previous_right = left + float(word["width"])

    return "".join(parts)


def _at(data: Mapping[str, Sequence[Any]], key: str, index: int, default: Any) -> Any:
    values = data.get(key)
    if values is None or index >= len(values):
        return default
    return values[index]


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
