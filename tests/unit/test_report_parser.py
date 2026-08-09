"""Report parsing against the fixture corpus.

The `raw_line` test is the one that guards the contract the brief calls out: every
emitted result must quote its source OCR line byte for byte.
"""

from __future__ import annotations

from typing import Any

import pytest

from services.domain import OcrLine, ResultFlag, ValueKind
from services.report_parser import parse_report

pytestmark = pytest.mark.unit

ALL_FIXTURES = [
    "cbc_report",
    "lipid_profile",
    "thyroid_panel",
    "partial_crop",
    "non_lab_receipt",
]


@pytest.mark.parametrize(
    ("fixture", "strong_rows", "total_results"),
    [
        ("cbc_report", 11, 11),
        # 6 strong + the Vitamin D row, kept but unparsed
        ("lipid_profile", 6, 7),
        # 4 strong + the "Not Done" FT3 row, vouched for by sitting between them
        ("thyroid_panel", 4, 5),
        # exactly at MIN_LAB_ROWS_FOR_REPORT
        ("partial_crop", 2, 2),
        # the hard negative: structurally receipt-shaped, clinically empty
        ("non_lab_receipt", 0, 0),
    ],
)
def test_counts_rows_per_fixture(
    ocr_lines: Any, fixture: str, strong_rows: int, total_results: int
) -> None:
    parsed = parse_report(ocr_lines(fixture))

    assert parsed.strong_row_count == strong_rows
    assert len(parsed.results) == total_results


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_raw_line_is_always_verbatim(ocr_lines: Any, fixture: str) -> None:
    """The invariant. No stripping, no whitespace collapsing, no reflowing.

    Compared against the exact source strings rather than a normalized form, so an
    accidental ``.strip()`` anywhere in the pipeline fails here.
    """
    lines = ocr_lines(fixture)
    source = {line.text for line in lines}

    parsed = parse_report(lines)
    for result in parsed.results:
        assert result.raw_line in source, f"{result.test_name}: raw_line was altered"
        # Belt and braces: the value's own raw text must appear in its source line.
        assert result.value.raw in result.raw_line


def test_raw_line_preserves_surrounding_whitespace() -> None:
    """Synthetic, because no corpus fixture has leading or trailing whitespace.

    That gap was found by mutation testing: adding ``.strip()`` to ``raw_line``
    passed the whole suite, since stripping is a no-op on every recorded line.
    Real OCR of an indented table can carry a left margin, and the invariant is
    byte-identity with ``OcrLine.text`` -- not "identity modulo whitespace" -- so it
    is asserted against a line that would visibly change under stripping.
    """
    padded = "   Haemoglobin          11.2        g/dl          13.0 - 17.0   "

    parsed = parse_report([OcrLine(text=padded, line_number=1, confidence=90.0)])

    assert len(parsed.results) == 1
    result = parsed.results[0]
    assert result.raw_line == padded, "surrounding whitespace was altered"
    assert result.raw_line.startswith("   ")
    assert result.raw_line.endswith("   ")
    # Cell contents are still trimmed -- only raw_line is sacred.
    assert result.test_name == "Haemoglobin"
    assert result.value.value == pytest.approx(11.2)


def test_extracts_metadata_preferring_specific_labels(ocr_lines: Any) -> None:
    meta = parse_report(ocr_lines("cbc_report")).meta

    assert meta.patient_name == "MD. RAFIQUL ISLAM"
    assert meta.age == "45 Y"
    assert meta.sex == "Male"
    assert meta.lab_name == "POPULAR DIAGNOSTIC CENTRE LTD."
    assert meta.reference_no == "PDC-2024-0098871"
    # The page carries both "Collection Date" and "Report Date"; the specific one wins.
    assert meta.report_date == "12/03/2024"


def test_extracts_metadata_from_split_and_piped_cells(ocr_lines: Any) -> None:
    """"Age : 52 Years    Sex : Female" is two fields on one line; "Dhaka | Ref. No: X"
    is separated by a pipe. Both must be read without being mistaken for a result row."""
    meta = parse_report(ocr_lines("lipid_profile")).meta

    assert meta.age == "52 Years"
    assert meta.sex == "Female"
    assert meta.reference_no == "LAB/2024/44821"
    assert meta.lab_name == "LABAID DIAGNOSTIC"


def test_preserves_infant_age_verbatim(ocr_lines: Any) -> None:
    """"3 M" is three MONTHS. Coercing it to the integer 3 would age the patient
    by three years, which is why ReportMeta.age is a string."""
    assert parse_report(ocr_lines("thyroid_panel")).meta.age == "3 M"


def test_reports_missing_metadata_as_null_rather_than_inventing_it(ocr_lines: Any) -> None:
    """A cropped photograph loses the header. Every field must be None -- in
    particular, a test name must never be promoted to a patient name."""
    meta = parse_report(ocr_lines("partial_crop")).meta

    assert (meta.patient_name, meta.age, meta.sex) == (None, None, None)
    assert (meta.report_date, meta.lab_name, meta.reference_no) == (None, None, None)


@pytest.mark.parametrize(
    ("test_name", "value", "unit", "flag"),
    [
        ("Haemoglobin", 11.2, "g/dL", ResultFlag.LOW),
        ("Total WBC Count", 12500.0, "/µL", ResultFlag.HIGH),
        ("Platelet Count", 180000.0, "/µL", ResultFlag.NORMAL),
        ("ESR", 32.0, "mm/hr", ResultFlag.HIGH),
        ("Neutrophils", 68.0, "%", ResultFlag.NORMAL),
        ("Monocytes", 6.0, "%", ResultFlag.NORMAL),
        ("RBC Count", 4.2, "10^6/µL", ResultFlag.LOW),
        ("MCV", 82.4, "fL", ResultFlag.LOW),
    ],
)
def test_parses_cbc_rows_end_to_end(
    ocr_lines: Any, test_name: str, value: float, unit: str, flag: ResultFlag
) -> None:
    results = {r.test_name: r for r in parse_report(ocr_lines("cbc_report")).results}
    row = results[test_name]

    assert row.value.value == pytest.approx(value)
    assert row.unit == unit
    assert row.flag is flag


def test_keeps_unparseable_values_without_repairing_them(ocr_lines: Any) -> None:
    """The Vitamin D row reads "2S.4" -- a capital S misread for a 5.

    It must survive as a row, with the garble intact and no number invented, while
    the fields that *were* readable are still normalized. Degradation is per-field,
    not per-row.
    """
    results = {r.test_name: r for r in parse_report(ocr_lines("lipid_profile")).results}
    row = results["Vitamin D (25-OH)"]

    assert row.value.kind is ValueKind.UNPARSED
    assert row.value.raw == "2S.4"
    assert row.value.value is None
    assert row.flag is ResultFlag.UNKNOWN
    # Still normalized, because these were readable:
    assert row.unit == "ng/mL"
    assert row.reference_range is not None
    assert row.reference_range.high == pytest.approx(100.0)


def test_keeps_non_numeric_findings_as_rows(ocr_lines: Any) -> None:
    """"Not Done" is a real finding. It must not be merged into the test name, and
    it must not be dropped -- it is vouched for by sitting between two strong rows."""
    results = {r.test_name: r for r in parse_report(ocr_lines("thyroid_panel")).results}

    assert "FT3" in results, "the Not Done row was dropped"
    row = results["FT3"]
    assert row.test_name == "FT3", "the value cell leaked into the test name"
    assert row.value.raw == "Not Done"
    assert row.value.kind is ValueKind.UNPARSED
    assert row.unit is None
    assert row.reference_range is None


def test_rejects_a_receipt_on_structure_not_keywords(ocr_lines: Any) -> None:
    """The hard negative.

    The receipt has a business header, a reference number, a date, a column header
    row and eight "label + number" rows -- everything a naive matcher looks for. What
    disqualifies it is that its numbers are bare singles: no clinical unit and no
    reference *interval* anywhere. So it yields no strong rows, which vouches for
    none of its weak rows either.
    """
    parsed = parse_report(ocr_lines("non_lab_receipt"))

    assert parsed.strong_row_count == 0
    # Checked before the emptiness assertion: `== ()` narrows the tuple to the empty
    # type, after which the comprehension variable has no type left to infer.
    assert not any("Basmati" in result.test_name for result in parsed.results)
    assert parsed.results == ()
