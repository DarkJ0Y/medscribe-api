"""Value parsing and flag derivation -- the accuracy-critical surface.

Three tests, heavily parameterized. The second one is the important one: it asserts
what the normalizer *refuses* to do. Recall is easy to test and easy to fix; a
parser that confidently invents a clinical number is the failure that reaches a
patient, so every ambiguous form has a case here.
"""

from __future__ import annotations

import pytest

from services import normalizer
from services.domain import Comparator, ResultFlag, ValueKind

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "kind", "expected"),
    [
        # --- the four forms named in the brief ---------------------------------
        ("<0.5", ValueKind.BOUNDED, {"comparator": Comparator.LT, "value": 0.5}),
        ("12,500", ValueKind.SINGLE, {"value": 12500.0}),
        ("1.2 x 10^3", ValueKind.SINGLE, {"value": 1200.0}),
        ("0.8 - 1.2", ValueKind.RANGE, {"low": 0.8, "high": 1.2}),
        # --- exponent notation variants ---------------------------------------
        ("1.8 x 10^5", ValueKind.SINGLE, {"value": 180000.0}),
        ("1.2 x 10^2", ValueKind.SINGLE, {"value": 120.0}),
        ("2.5×10**4", ValueKind.SINGLE, {"value": 25000.0}),
        ("1.2e3", ValueKind.SINGLE, {"value": 1200.0}),
        ("4.5 x 10^-2", ValueKind.SINGLE, {"value": 0.045}),
        # --- comma grouping ----------------------------------------------------
        ("4,000 - 11,000", ValueKind.RANGE, {"low": 4000.0, "high": 11000.0}),
        ("1,234,567", ValueKind.SINGLE, {"value": 1234567.0}),
        # --- a range whose single written exponent belongs to both endpoints ---
        ("1.5 - 4.5 x 10^5", ValueKind.RANGE, {"low": 150000.0, "high": 450000.0}),
        # --- range separators --------------------------------------------------
        ("13.0 to 17.0", ValueKind.RANGE, {"low": 13.0, "high": 17.0}),
        ("0.40 - 4.00", ValueKind.RANGE, {"low": 0.40, "high": 4.00}),
        ("83 – 101", ValueKind.RANGE, {"low": 83.0, "high": 101.0}),  # en dash
        # --- comparators, symbolic and spelled out ----------------------------
        (">1000", ValueKind.BOUNDED, {"comparator": Comparator.GT, "value": 1000.0}),
        (">= 200", ValueKind.BOUNDED, {"comparator": Comparator.GTE, "value": 200.0}),
        ("≥200", ValueKind.BOUNDED, {"comparator": Comparator.GTE, "value": 200.0}),
        ("≤ 0.5", ValueKind.BOUNDED, {"comparator": Comparator.LTE, "value": 0.5}),
        ("< 6.0", ValueKind.BOUNDED, {"comparator": Comparator.LT, "value": 6.0}),
        ("up to 200", ValueKind.BOUNDED, {"comparator": Comparator.LTE, "value": 200.0}),
        ("less than 6", ValueKind.BOUNDED, {"comparator": Comparator.LT, "value": 6.0}),
        ("at least 40", ValueKind.BOUNDED, {"comparator": Comparator.GTE, "value": 40.0}),
        # --- leading zeros as printed on differential counts ------------------
        ("06", ValueKind.SINGLE, {"value": 6.0}),
        ("02", ValueKind.SINGLE, {"value": 2.0}),
        # --- whitespace tolerance ---------------------------------------------
        ("  11.2  ", ValueKind.SINGLE, {"value": 11.2}),
    ],
)
def test_parses_every_documented_value_form(
    raw: str, kind: ValueKind, expected: dict[str, object]
) -> None:
    parsed = normalizer.parse_value(raw)

    assert parsed.kind is kind
    assert parsed.is_parsed is True
    # The raw text survives parsing untouched, always.
    assert parsed.raw == raw
    for field, value in expected.items():
        actual = getattr(parsed, field)
        if isinstance(value, float):
            assert actual == pytest.approx(value), f"{field}: {actual!r} != {value!r}"
        else:
            assert actual == value, f"{field}: {actual!r} != {value!r}"


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("2S.4", "OCR read a capital S for a 5; repairing it would invent a value"),
        ("1,25", "thousands separator or European decimal comma -- unknowable"),
        ("12,50", "invalid thousands grouping"),
        ("1.2 x 10 3", "the caret was lost to OCR, so the exponent would be a guess"),
        ("11..2", "malformed decimal"),
        ("4.5.6", "malformed decimal"),
        ("1 x 10^999", "absurd exponent"),
        ("Not Done", "a real finding, but not a number"),
        ("Trace", "a real finding, but not a number"),
        ("Negative", "a real finding, but not a number"),
        ("", "empty cell"),
        ("   ", "whitespace only"),
        ("abc", "not numeric at all"),
        ("-", "a dash means nothing was printed"),
        ("20 - 10", "inverted range; refuse rather than silently swap"),
    ],
)
def test_refuses_ambiguous_input_without_guessing(raw: str, reason: str) -> None:
    parsed = normalizer.parse_value(raw)

    assert parsed.kind is ValueKind.UNPARSED, reason
    assert parsed.is_parsed is False
    # No number is fabricated on ANY field...
    assert parsed.value is None
    assert parsed.low is None
    assert parsed.high is None
    assert parsed.comparator is None
    # ...and the original text is preserved so a human can adjudicate it.
    assert parsed.raw == raw


@pytest.mark.parametrize(
    ("value", "reference", "flag", "why"),
    [
        # --- plain intervals --------------------------------------------------
        ("11.2", "13.0 - 17.0", ResultFlag.LOW, "below the floor"),
        ("12,500", "4,000 - 11,000", ResultFlag.HIGH, "above the ceiling"),
        ("24", "20 - 45", ResultFlag.NORMAL, "inside"),
        ("20", "20 - 45", ResultFlag.NORMAL, "endpoints are inclusive"),
        ("45", "20 - 45", ResultFlag.NORMAL, "endpoints are inclusive"),
        # --- exponents must be expanded before comparing ----------------------
        ("1.8 x 10^5", "1.5 - 4.5 x 10^5", ResultFlag.NORMAL, "180k inside 150k-450k"),
        ("1.2 x 10^2", "80 - 200", ResultFlag.NORMAL, "120 inside 80-200, not 1.2 below it"),
        # --- upper-bounded references ----------------------------------------
        ("245", "< 200", ResultFlag.HIGH, "over the bound"),
        ("150", "< 150", ResultFlag.HIGH, "the bound is strict"),
        ("149", "< 150", ResultFlag.NORMAL, "under the bound"),
        # --- lower-bounded references: the direction that is easy to invert ---
        ("38", "> 40", ResultFlag.LOW, "HDL below its floor is LOW, not HIGH"),
        ("45", "> 40", ResultFlag.NORMAL, "above its floor"),
        ("40", "> 40", ResultFlag.LOW, "the bound is strict"),
        # --- one-sided values against one-sided references --------------------
        ("<0.5", "< 6.0", ResultFlag.NORMAL, "every value under 0.5 is under 6.0"),
        (">1000", "< 34", ResultFlag.HIGH, "every value over 1000 exceeds 34"),
        ("<10", "> 40", ResultFlag.LOW, "every value under 10 is under 40"),
        # --- everything uncertain must be UNKNOWN, never NORMAL --------------
        ("2S.4", "30 - 100", ResultFlag.UNKNOWN, "unparsed value"),
        ("5.4", None, ResultFlag.UNKNOWN, "no reference range printed"),
        ("5.4", "abc", ResultFlag.UNKNOWN, "unparsed reference"),
        ("<50", "10 - 20", ResultFlag.UNKNOWN, "the bound does not settle it"),
        (">5", "10 - 20", ResultFlag.UNKNOWN, "the bound does not settle it"),
        ("1 - 2", "10 - 20", ResultFlag.UNKNOWN, "a range cannot be positioned"),
        ("5", "7", ResultFlag.UNKNOWN, "a bare reference states a target, not an interval"),
    ],
)
def test_derives_flags_including_inverted_bounds(
    value: str, reference: str | None, flag: ResultFlag, why: str
) -> None:
    derived = normalizer.derive_flag(
        normalizer.parse_value(value),
        normalizer.parse_reference_range(reference),
    )
    assert derived is flag, f"{value} vs {reference}: {why}"


@pytest.mark.parametrize("absent", ["-", "--", "N/A", "n/a", "nil", "none", "", "  "])
def test_absent_reference_is_none_not_unparsed(absent: str) -> None:
    """A range that was never printed differs from one that could not be read."""
    assert normalizer.parse_reference_range(absent) is None
    assert normalizer.parse_reference_range(None) is None
