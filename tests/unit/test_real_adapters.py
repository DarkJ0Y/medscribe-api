"""The real adapters' pure logic: OCR whitespace reconstruction and response mapping.

Requires the ``real`` extra (``pip install -e ".[dev,real]"``) because
``adapters/ocr/tesseract.py`` and ``adapters/transcription/whisper_api.py`` import
their provider SDKs at module scope. It skips cleanly without them so a minimal
development install still has a green suite -- but CI installs the extra, because
the reconstruction test below covers a bug that no mock-based test can catch.

No tesseract binary or API key is needed: everything tested here is pure logic.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pytesseract", reason="requires the 'real' extra")
pytest.importorskip("PIL", reason="requires the 'real' extra")
pytest.importorskip("openai", reason="requires the 'real' extra")

from adapters.ocr.tesseract import reconstruct_lines  # noqa: E402
from adapters.transcription.whisper_api import to_raw_transcription  # noqa: E402
from services.domain import OcrLine  # noqa: E402
from services.report_parser import parse_report  # noqa: E402

pytestmark = pytest.mark.unit

# --------------------------------- tesseract whitespace reconstruction
def _word_boxes(rows: list[list[tuple[str, int, int]]]) -> dict[str, list[Any]]:
    """Build an image_to_data DICT from (text, left, width) triples per line."""
    data: dict[str, list[Any]] = {
        key: [] for key in
        ("text", "conf", "left", "width", "block_num", "par_num", "line_num")
    }
    for line_index, row in enumerate(rows, start=1):
        for text, left, width in row:
            data["text"].append(text)
            data["conf"].append("91.5")  # tesseract returns confidences as strings
            data["left"].append(left)
            data["width"].append(width)
            data["block_num"].append(1)
            data["par_num"].append(1)
            data["line_num"].append(line_index)
    return data


CBC_BOXES = [
    [("Haemoglobin", 40, 110), ("11.2", 250, 40), ("g/dl", 370, 40),
     ("13.0", 510, 40), ("-", 560, 10), ("17.0", 580, 40)],
    [("Total", 40, 50), ("WBC", 100, 40), ("Count", 150, 50),
     ("12,500", 250, 60), ("/cumm", 370, 50),
     ("4,000", 510, 50), ("-", 570, 10), ("11,000", 590, 60)],
]


def test_reconstructs_column_gaps_from_pixel_geometry() -> None:
    """Tesseract's word boxes carry no spacing; the parser splits on 2+ spaces.

    Without this reconstruction every real report would parse as a single cell and
    be rejected as "not a lab report" -- while the mock suite kept passing, because
    its fixtures already contain the spacing. See DECISIONS.md D18.
    """
    lines = reconstruct_lines(_word_boxes(CBC_BOXES))

    assert len(lines) == 2
    assert lines[0].text == "Haemoglobin          11.2        g/dl          13.0 - 17.0"
    # Wide gaps became multi-space runs; the narrow ones inside "13.0 - 17.0" did not.
    assert "  " in lines[0].text
    assert "13.0 - 17.0" in lines[0].text
    assert lines[0].confidence == pytest.approx(91.5)
    assert [line.line_number for line in lines] == [1, 2]


def test_reconstructed_lines_parse_correctly_end_to_end() -> None:
    """The reconstruction is only correct if the parser can actually use it."""
    parsed = parse_report(reconstruct_lines(_word_boxes(CBC_BOXES)))

    assert parsed.strong_row_count == 2
    haemoglobin, wbc = parsed.results
    assert haemoglobin.test_name == "Haemoglobin"
    assert haemoglobin.value.value == pytest.approx(11.2)
    assert haemoglobin.unit == "g/dL"
    assert haemoglobin.reference_range is not None
    assert haemoglobin.reference_range.low == pytest.approx(13.0)
    assert haemoglobin.flag.value == "low"
    # A test name split across three word boxes is reassembled.
    assert wbc.test_name == "Total WBC Count"
    assert wbc.value.value == pytest.approx(12500.0)


def test_naive_single_space_joining_would_break_parsing() -> None:
    """The counter-proof. Without it, the test above could pass for the wrong reason."""
    naive = OcrLine(
        text=" ".join(text for text, _, _ in CBC_BOXES[0]), line_number=1, confidence=91.5
    )
    assert parse_report([naive]).strong_row_count == 0


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({}, ()),
        ({"text": [], "conf": []}, ()),
    ],
)
def test_reconstruction_handles_empty_input(data: dict[str, Any], expected: tuple[()]) -> None:
    assert reconstruct_lines(data) == expected


def test_reconstruction_drops_phantom_boxes() -> None:
    """Tesseract emits conf == -1 for structural rows carrying no text."""
    lines = reconstruct_lines(
        {
            "text": ["real", "phantom"],
            "conf": ["88", "-1"],
            "left": [10, 200],
            "width": [40, 40],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
        }
    )

    assert lines[0].text == "real"
    assert lines[0].confidence == pytest.approx(88.0)


def test_one_mis_boxed_glyph_cannot_collapse_every_gap() -> None:
    """Character width is a median, not a mean.

    A single 400-pixel box around one glyph would inflate a mean enough to turn every
    column gap back into a single space, silently reintroducing the D18 bug.
    """
    skewed = _word_boxes([[("x", 0, 400), ("Haemoglobin", 40, 110), ("11.2", 250, 40)]])
    assert "  " in reconstruct_lines(skewed)[0].text


def test_absurd_gaps_are_capped() -> None:
    capped = reconstruct_lines(_word_boxes([[("a", 0, 10), ("b", 100_000, 10)]]), max_gap_spaces=5)
    assert capped[0].text == "a" + " " * 5 + "b"


# ------------------------------------------- whisper response mapping
class _Segment:
    def __init__(self, start: float, end: float, text: str, no_speech_prob: float) -> None:
        self.start, self.end, self.text = start, end, text
        self.no_speech_prob = no_speech_prob


def test_maps_a_verbose_json_response() -> None:
    class Response:
        text = "রোগীর জ্বর আছে"
        language = "bengali"
        duration = 8.64
        segments = [_Segment(0.0, 4.0, "রোগীর জ্বর", 0.02), _Segment(4.0, 8.64, "আছে", 0.05)]

    raw = to_raw_transcription(Response(), provider="openai-whisper-1")

    assert raw.text == "রোগীর জ্বর আছে"
    # Reported verbatim, NOT mapped: the service owns language interpretation.
    assert raw.detected_language == "bengali"
    assert raw.duration_seconds == pytest.approx(8.64)
    assert len(raw.segments) == 2
    # The MINIMUM across segments: the clip is only "no speech" if even its most
    # speech-like segment scored as non-speech. A mean would blank valid transcripts.
    assert raw.no_speech_probability == pytest.approx(0.02)


def test_maps_a_plain_json_response_without_inventing_fields() -> None:
    """Models other than whisper-1 return text only. The absent signals must read as
    unavailable (None), not as a confident 0.0."""

    class Response:
        text = "hello there"

    raw = to_raw_transcription(Response(), provider="openai-gpt-4o-transcribe")

    assert raw.text == "hello there"
    assert raw.detected_language is None
    assert raw.duration_seconds is None
    assert raw.segments == ()
    assert raw.no_speech_probability is None


def test_maps_dict_shaped_segments() -> None:
    """Some SDK versions return verbose_json segments as dicts, not models."""

    class Response:
        text = "x"
        language = "en"
        duration = 2.0
        segments = [{"start": 0.0, "end": 2.0, "text": "x", "no_speech_prob": 0.9}]

    raw = to_raw_transcription(Response(), provider="openai-whisper-1")
    assert raw.segments[0].no_speech_probability == pytest.approx(0.9)
