"""OCR line list -> (ReportMeta, [LabResult]).

Separates header/metadata lines from tabular result lines, and guarantees that
every emitted result carries ``raw_line``: the exact, unmodified OCR string it was
derived from.

How rows are found
------------------
Lines are segmented into cells on runs of **two or more spaces** (or a pipe),
which is how OCR renders column gaps. Then each candidate row is classified:

**Strong** -- the value parses, AND the row has either a recognized clinical unit
or a reference cell that is a range/one-sided bound. Strong rows are the evidence
that this document is a lab report at all, and only they are counted against
``MIN_LAB_ROWS_FOR_REPORT``.

**Weak** -- looks like a row but the value did not parse (``Not Done``, an OCR
garble like ``2S.4``). These are kept, with the value preserved verbatim and
UNPARSED, but only when structurally vouched for: either sandwiched *between* two
strong rows, or within two lines of the strong block while carrying a recognized
unit or a real reference range.

That two-tier rule is what rejects a supermarket receipt. A receipt has a
business header, a reference number, a date, a column header row and eight
``label + number`` rows -- everything a naive matcher looks for. What it lacks is
any clinical unit and any reference *interval*: its numbers are bare singles, not
ranges or bounds. So it produces zero strong rows, which vouches for no weak rows
either, and the document is refused rather than returned as a report listing
"Basmati Rice 5kg" with a value of 720.00.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from services import normalizer, units
from services.domain import LabResult, NumericValue, OcrLine, ReportMeta, ValueKind

# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
_CELL_SPLIT: Final[re.Pattern[str]] = re.compile(r"\s{2,}|\s*\|\s*")

_KEY_VALUE: Final[re.Pattern[str]] = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z.#\s/&']{0,30}?)\s*[:\-]\s*(?P<value>.+)"
)

#: Label -> (domain field, specificity). Higher specificity overwrites lower, so
#: "Report Date" wins over a generic "Date" or "Collection Date" on the same page.
_META_LABELS: Final[dict[str, tuple[str, int]]] = {
    "patient name": ("patient_name", 2),
    "patients name": ("patient_name", 2),
    "patient": ("patient_name", 1),
    "name": ("patient_name", 1),
    "age": ("age", 2),
    "age/sex": ("age_sex", 2),
    "sex/age": ("sex_age", 2),
    "sex": ("sex", 2),
    "gender": ("sex", 2),
    "report date": ("report_date", 3),
    "reported on": ("report_date", 3),
    "reporting date": ("report_date", 3),
    "date of report": ("report_date", 3),
    "collection date": ("report_date", 2),
    "sample date": ("report_date", 2),
    "collected on": ("report_date", 2),
    "date": ("report_date", 1),
    "reference no": ("reference_no", 3),
    "reference number": ("reference_no", 3),
    "ref no": ("reference_no", 2),
    "reg no": ("reference_no", 2),
    "registration no": ("reference_no", 2),
    "lab no": ("reference_no", 2),
    "invoice no": ("reference_no", 1),
    "invoice": ("reference_no", 1),
    "id no": ("reference_no", 1),
    "lab name": ("lab_name", 3),
}

#: Words that make up column headings rather than data.
_HEADER_WORDS: Final[frozenset[str]] = frozenset(
    {
        "test", "tests", "name", "result", "results", "value", "values",
        "unit", "units", "reference", "ref", "range", "ranges", "normal",
        "interval", "bio", "biological", "investigation", "investigations",
        "parameter", "parameters", "observation", "method", "flag", "remarks",
        "qty", "price", "item", "items", "amount", "description", "particulars",
    }
)

#: Words that mark a panel title, so it is not mistaken for the lab's name.
_PANEL_WORDS: Final[frozenset[str]] = frozenset(
    {
        "profile", "count", "test", "tests", "panel", "function", "cbc",
        "report", "investigation", "examination", "analysis", "screening",
    }
)

#: Non-numeric results that are real findings, not parse failures. Recognizing
#: them stops "Not Done" from being merged into the test name -- but they are
#: still stored UNPARSED, because they are not numbers.
_NON_NUMERIC_VALUES: Final[frozenset[str]] = frozenset(
    {
        "not done", "notdone", "nd", "nil", "trace", "negative", "positive",
        "absent", "present", "reactive", "non-reactive", "nonreactive",
        "normal", "abnormal", "pending", "awaited", "n/a", "na", "-", "--",
        "insufficient", "haemolysed", "hemolysed", "clotted",
    }
)

_DIGIT: Final[re.Pattern[str]] = re.compile(r"\d")
_WORD: Final[re.Pattern[str]] = re.compile(r"[a-z]+")

#: How far past the strong block a weak row may sit and still be vouched for.
_WEAK_ROW_GAP: Final[int] = 2


@dataclass(frozen=True, slots=True)
class ParsedReport:
    """Outcome of parsing one page of OCR lines.

    ``strong_row_count`` is deliberately separate from ``len(results)``: it is the
    evidence count the lab-report discriminator uses, while ``results`` also
    contains the weak rows that were structurally vouched for.
    """

    meta: ReportMeta
    results: tuple[LabResult, ...]
    strong_row_count: int
    recognized_unit_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Candidate:
    line_number: int
    raw_line: str
    test_name: str
    value: NumericValue
    unit: str | None
    reference: NumericValue | None
    is_strong: bool
    has_supporting_evidence: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cells(text: str) -> list[str]:
    return [part.strip() for part in _CELL_SPLIT.split(text.strip()) if part.strip()]


def _normalize_label(text: str) -> str:
    lowered = text.strip().lower().replace("#", "").replace(".", "")
    collapsed = re.sub(r"\s*/\s*", "/", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()


def _is_header_row(cells: Sequence[str]) -> bool:
    """True when at least two cells consist purely of column-heading words."""
    heading_cells = 0
    for cell in cells:
        words = _WORD.findall(cell.lower())
        if words and all(word in _HEADER_WORDS for word in words):
            heading_cells += 1
    return heading_cells >= 2


def _is_non_numeric_value(cell: str) -> bool:
    return cell.strip().lower() in _NON_NUMERIC_VALUES


def _looks_like_name_fragment(cell: str) -> bool:
    """True when a cell is a spilled-over piece of the test name.

    OCR sometimes puts a wide gap inside a long test name ("Total   WBC   Count"),
    which would otherwise be read as the value column. A fragment has no digits,
    is not a unit, and is not one of the recognized non-numeric results.
    """
    if _DIGIT.search(cell):
        return False
    if _is_non_numeric_value(cell):
        return False
    if units.is_recognized_unit(cell) or units.is_absent(cell):
        return False
    return bool(_WORD.search(cell.lower()))


def _is_interval(value: NumericValue | None) -> bool:
    """A reference cell only counts as evidence if it bounds something.

    A bare number in the reference column ("720.00" on a receipt) states nothing
    about an acceptable interval, so it is not evidence of a lab report.
    """
    return value is not None and value.kind in (ValueKind.RANGE, ValueKind.BOUNDED)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def _extract_meta(lines: Sequence[OcrLine]) -> tuple[ReportMeta, set[int]]:
    """Pull header fields, returning them plus the line numbers they consumed.

    Consumed lines are excluded from result-row detection, which is what stops
    "Age : 52 Years    Sex : Female" -- two cells, no colon-free structure -- from
    being read as a test row named "Age : 52 Years".
    """
    found: dict[str, tuple[str, int]] = {}
    consumed: set[int] = set()

    def offer(field: str, value: str, specificity: int) -> None:
        cleaned = value.strip()
        if not cleaned or units.is_absent(cleaned):
            return
        existing = found.get(field)
        if existing is None or specificity > existing[1]:
            found[field] = (cleaned, specificity)

    for line in lines:
        for cell in _cells(line.text):
            match = _KEY_VALUE.fullmatch(cell)
            if match is None:
                continue
            label = _normalize_label(match.group("label"))
            entry = _META_LABELS.get(label)
            if entry is None:
                continue
            field, specificity = entry
            value = match.group("value").strip()

            if field in ("age_sex", "sex_age"):
                # "45 Y / Male" or "Male / 45 Y"
                parts = [p.strip() for p in value.split("/") if p.strip()]
                order = ("age", "sex") if field == "age_sex" else ("sex", "age")
                # strict=False is deliberate: "Age/Sex : 45 Y" (one part) is common
                # when the sex column is cropped off, and truncating is correct.
                for name, part in zip(order, parts, strict=False):
                    offer(name, part, specificity)
                if len(parts) == 1:
                    offer(order[0], parts[0], specificity)
            else:
                offer(field, value, specificity)
            consumed.add(line.line_number)

    return (
        ReportMeta(
            patient_name=found["patient_name"][0] if "patient_name" in found else None,
            age=found["age"][0] if "age" in found else None,
            sex=found["sex"][0] if "sex" in found else None,
            report_date=found["report_date"][0] if "report_date" in found else None,
            lab_name=found.get("lab_name", (None, 0))[0] or _infer_lab_name(lines),
            reference_no=found["reference_no"][0] if "reference_no" in found else None,
        ),
        consumed,
    )


def _infer_lab_name(lines: Sequence[OcrLine], scan_lines: int = 5) -> str | None:
    """Guess the lab's name from the letterhead.

    Restricted to the first few lines, to single-cell lines without a colon, and
    to predominantly upper-case text -- and panel titles ("COMPLETE BLOOD COUNT
    (CBC)", "THYROID FUNCTION TEST") are excluded by keyword, since they share
    every other property with a letterhead.
    """
    for line in lines[:scan_lines]:
        text = line.text.strip()
        if ":" in text or len(text) < 4:
            continue
        if len(_cells(text)) != 1:
            continue
        letters = [char for char in text if char.isalpha()]
        if not letters:
            continue
        if sum(char.isupper() for char in letters) / len(letters) < 0.6:
            continue
        if set(_WORD.findall(text.lower())) & _PANEL_WORDS:
            continue
        return text
    return None


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------
def _build_candidate(line: OcrLine) -> _Candidate | None:
    cells = _cells(line.text)
    if len(cells) < 2 or _is_header_row(cells):
        return None
    if not _WORD.search(cells[0].lower()):
        return None
    first_words = _WORD.findall(cells[0].lower())
    if first_words and all(word in _HEADER_WORDS for word in first_words):
        return None

    # Absorb name fragments, but never the last cell -- a row must keep a value.
    name_parts = [cells[0]]
    index = 1
    while index < len(cells) - 1 and _looks_like_name_fragment(cells[index]):
        name_parts.append(cells[index])
        index += 1

    value = normalizer.parse_value(cells[index])
    rest = cells[index + 1 :]

    # The reference is the LAST cell that bounds an interval; the unit is the
    # first remaining cell that is not that reference.
    reference: NumericValue | None = None
    reference_position: int | None = None
    for position in range(len(rest) - 1, -1, -1):
        candidate = normalizer.parse_reference_range(rest[position])
        if _is_interval(candidate):
            reference, reference_position = candidate, position
            break

    unit_raw: str | None = None
    for position, cell in enumerate(rest):
        if position == reference_position or units.is_absent(cell):
            continue
        # A cell that is itself a number is not a unit. Without this, a report
        # whose reference column holds a bare target ("5") would report the unit
        # as "5", since normalize_unit passes unrecognized text through.
        if normalizer.parse_value(cell).is_parsed:
            continue
        unit_raw = cell
        break

    unit = units.normalize_unit(unit_raw)
    unit_recognized = units.is_recognized_unit(unit_raw)
    supporting = unit_recognized or _is_interval(reference)

    return _Candidate(
        line_number=line.line_number,
        raw_line=line.text,
        test_name=" ".join(name_parts).strip(),
        value=value,
        unit=unit,
        reference=reference,
        is_strong=value.is_parsed and supporting,
        has_supporting_evidence=supporting,
    )


def _accept_weak(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    """Keep weak rows only where the strong rows vouch for their position."""
    strong_lines = [c.line_number for c in candidates if c.is_strong]
    if not strong_lines:
        return [c for c in candidates if c.is_strong]

    first, last = min(strong_lines), max(strong_lines)
    accepted: list[_Candidate] = []
    for candidate in candidates:
        if candidate.is_strong:
            accepted.append(candidate)
            continue
        inside_block = first < candidate.line_number < last
        adjacent = (
            candidate.line_number > last and candidate.line_number - last <= _WEAK_ROW_GAP
        ) or (candidate.line_number < first and first - candidate.line_number <= _WEAK_ROW_GAP)
        if inside_block or (adjacent and candidate.has_supporting_evidence):
            accepted.append(candidate)
    return accepted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_report(lines: Sequence[OcrLine]) -> ParsedReport:
    """Parse one page of OCR lines into metadata plus lab results."""
    meta, consumed = _extract_meta(lines)

    candidates = [
        candidate
        for line in lines
        if line.line_number not in consumed and (candidate := _build_candidate(line)) is not None
    ]
    accepted = _accept_weak(candidates)
    accepted.sort(key=lambda c: c.line_number)

    results = tuple(
        LabResult(
            test_name=candidate.test_name,
            value=candidate.value,
            # The invariant: whatever parsing did or failed to do, the caller gets
            # the original line back, byte for byte.
            raw_line=candidate.raw_line,
            unit=candidate.unit,
            reference_range=candidate.reference,
            flag=normalizer.derive_flag(candidate.value, candidate.reference),
        )
        for candidate in accepted
    )

    warnings: list[str] = []
    unparsed = sum(1 for result in results if not result.value.is_parsed)
    if unparsed:
        warnings.append(
            f"{unparsed} of {len(results)} result values could not be parsed and are "
            "preserved verbatim without a numeric reading."
        )
    dropped = len(candidates) - len(accepted)
    if dropped:
        warnings.append(
            f"{dropped} row-like line(s) were not recognized as lab results and were omitted."
        )

    return ParsedReport(
        meta=meta,
        results=results,
        strong_row_count=sum(1 for candidate in accepted if candidate.is_strong),
        recognized_unit_count=sum(
            1 for candidate in accepted if units.is_recognized_unit(candidate.unit)
        ),
        warnings=tuple(warnings),
    )
