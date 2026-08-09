"""Numeric value + unit normalization -- the accuracy-critical module.

Structures, without guessing:

    "<0.5"          -> BOUNDED, comparator "<", value 0.5
    "12,500"        -> SINGLE,  value 12500
    "1.2 x 10^3"    -> SINGLE,  value 1200.0
    "0.8 - 1.2"     -> RANGE,   low 0.8, high 1.2
    "1.5 - 4.5 x 10^5" -> RANGE, low 150000, high 450000  (shared exponent)

Anything ambiguous is preserved verbatim behind ``ValueKind.UNPARSED`` rather than
coerced into a confidently wrong number -- see DECISIONS.md D5. Concretely, all of
these come back unparsed with ``raw`` intact:

    "2S.4"      OCR read a capital S for a 5. Repairing it would invent a value.
    "1,25"      Thousands separator or European decimal comma? Unknowable.
    "1.2 x 10 3" The caret was lost; the exponent is a guess.
    "Not Done"  Not a number at all.
    "11..2"     Malformed.

Two implementation notes
------------------------
**Decimal, not float, for intermediate arithmetic.** ``1.2 x 10^3`` computed in
binary floating point risks 1200.0000000000002; ``Decimal("1.2") * 10**3`` is
exactly 1200. The conversion to ``float`` happens once, at the boundary, because
that is what the domain model and JSON declare.

**Anchored patterns only.** Every matcher is a ``fullmatch``. A partial match is
how "11.2 g/dl" silently becomes 11.2 with the unit thrown away, or how "2S.4"
becomes 2. If the whole cell does not fit a known shape, it is unparsed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from typing import Final

from services import units
from services.domain import Comparator, NumericValue, ResultFlag, ValueKind

# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------
# A comma-grouped integer must group in exact thousands. "12,500" qualifies;
# "1,25" does not, and is therefore unparsed rather than being read as either
# 125 or 1.25.
_GROUPED: Final[str] = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
_PLAIN: Final[str] = r"\d+(?:\.\d+)?"
_NUMBER: Final[str] = rf"(?:{_GROUPED}|{_PLAIN})"

#: ``x 10^5``, ``×10**5``, or plain ``e5``. The caret/asterisks are REQUIRED:
#: "1.2 x 10 3" has lost its operator to OCR and stays unparsed.
_QUANTITY_RE: Final[re.Pattern[str]] = re.compile(
    rf"""
    (?P<sign>-)?
    (?P<num>{_NUMBER})
    (?:
        \s*[x×*]\s*10\s*(?:\^|\*\*)\s*(?P<sci>[+-]?\d+)
      | [eE](?P<exp>[+-]?\d+)
    )?
    """,
    re.VERBOSE,
)

_RANGE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<low>.+?)\s*(?:-|–|—|~|\bto\b)\s*(?P<high>.+)",
    re.IGNORECASE,
)

# Longest first: "<=" must win over "<".
_COMPARATORS: Final[tuple[tuple[str, Comparator], ...]] = (
    ("<=", Comparator.LTE),
    ("=<", Comparator.LTE),
    ("≤", Comparator.LTE),
    (">=", Comparator.GTE),
    ("=>", Comparator.GTE),
    ("≥", Comparator.GTE),
    ("<", Comparator.LT),
    (">", Comparator.GT),
)

_WORD_COMPARATORS: Final[tuple[tuple[str, Comparator], ...]] = (
    ("less than or equal to", Comparator.LTE),
    ("greater than or equal to", Comparator.GTE),
    ("less than", Comparator.LT),
    ("greater than", Comparator.GT),
    ("at least", Comparator.GTE),
    ("at most", Comparator.LTE),
    ("up to", Comparator.LTE),
    ("upto", Comparator.LTE),
)

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")

#: Refuse absurd exponents rather than raising or producing inf.
_MAX_ABS_EXPONENT: Final[int] = 30


@dataclass(frozen=True, slots=True)
class _Quantity:
    """A parsed magnitude, keeping mantissa and exponent separable.

    They stay separate so a range can share one written exponent across both
    endpoints -- see :func:`_apply_shared_exponent`.
    """

    mantissa: Decimal
    exponent: int | None

    @property
    def value(self) -> Decimal:
        if self.exponent is None:
            return self.mantissa
        return self.mantissa * (Decimal(10) ** self.exponent)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def _clean(raw: str) -> str:
    """Trim and collapse whitespace runs. Nothing else is altered."""
    return _WHITESPACE.sub(" ", raw.strip())


def _to_decimal(token: str) -> Decimal | None:
    """Convert a number token, validating comma grouping strictly."""
    text = token.strip()
    if "," in text:
        if not re.fullmatch(_GROUPED, text):
            return None
        text = text.replace(",", "")
    try:
        return Decimal(text)
    except (DecimalException, ValueError):
        return None


def _parse_quantity(text: str) -> _Quantity | None:
    """Parse one magnitude, or ``None`` if the whole token is not a magnitude."""
    match = _QUANTITY_RE.fullmatch(text.strip())
    if match is None:
        return None

    mantissa = _to_decimal(match.group("num"))
    if mantissa is None:
        return None
    if match.group("sign"):
        mantissa = -mantissa

    raw_exponent = match.group("sci") or match.group("exp")
    if raw_exponent is None:
        return _Quantity(mantissa=mantissa, exponent=None)

    exponent = int(raw_exponent)
    if abs(exponent) > _MAX_ABS_EXPONENT:
        return None
    return _Quantity(mantissa=mantissa, exponent=exponent)


def _split_comparator(text: str) -> tuple[Comparator, str] | None:
    """Peel a leading comparator, symbolic or spelled out."""
    for symbol, comparator in _COMPARATORS:
        if text.startswith(symbol):
            return comparator, text[len(symbol) :].strip()
    lowered = text.lower()
    for phrase, comparator in _WORD_COMPARATORS:
        if lowered.startswith(phrase):
            return comparator, text[len(phrase) :].strip()
    return None


def _apply_shared_exponent(low: _Quantity, high: _Quantity) -> tuple[_Quantity, _Quantity]:
    """Distribute a single written exponent across both endpoints of a range.

    Lab reports print ``1.5 - 4.5 x 10^5`` meaning 150,000 to 450,000. Reading the
    exponent as belonging to the high endpoint alone gives 1.5 to 450,000, which
    is not a plausible reference interval for anything.

    This is a convention, not a guess: there is no competing reading. It applies
    in one direction only -- a written exponent is never removed, and an exponent
    on the LOW endpoint is never propagated upward, since ``1.5 x 10^5 - 4.5``
    is malformed rather than shorthand.
    """
    if low.exponent is None and high.exponent is not None:
        return _Quantity(low.mantissa, high.exponent), high
    return low, high


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_value(raw: str) -> NumericValue:
    """Structure a measurement cell, preserving ``raw`` verbatim regardless."""
    text = _clean(raw)
    if not text:
        return NumericValue.unparsed(raw)

    # Range before comparator: "0.8 - 1.2" must not be read as a bound.
    if (parsed := _try_range(raw, text)) is not None:
        return parsed
    if (parsed := _try_bounded(raw, text)) is not None:
        return parsed
    if (parsed := _try_single(raw, text)) is not None:
        return parsed
    return NumericValue.unparsed(raw)


def parse_reference_range(raw: str | None) -> NumericValue | None:
    """Structure a reference-range cell.

    Returns ``None`` when the cell is absent (blank, ``-``, ``N/A``) -- the
    reference range was not printed, which is different from being unparseable.
    A cell that is present but unreadable still comes back as UNPARSED so the
    caller can see what was there.
    """
    if raw is None or units.is_absent(raw):
        return None
    return parse_value(raw)


def _try_range(raw: str, text: str) -> NumericValue | None:
    match = _RANGE_RE.fullmatch(text)
    if match is None:
        return None
    low = _parse_quantity(match.group("low"))
    high = _parse_quantity(match.group("high"))
    if low is None or high is None:
        return None

    low, high = _apply_shared_exponent(low, high)
    low_value, high_value = low.value, high.value
    if low_value > high_value:
        # Inverted bounds mean the split landed somewhere it should not have
        # (a hyphenated identifier, a negative number). Refuse rather than
        # silently swapping them.
        return None
    return NumericValue(
        raw=raw,
        kind=ValueKind.RANGE,
        low=float(low_value),
        high=float(high_value),
    )


def _try_bounded(raw: str, text: str) -> NumericValue | None:
    split = _split_comparator(text)
    if split is None:
        return None
    comparator, remainder = split
    quantity = _parse_quantity(remainder)
    if quantity is None:
        return None
    return NumericValue(
        raw=raw,
        kind=ValueKind.BOUNDED,
        comparator=comparator,
        value=float(quantity.value),
    )


def _try_single(raw: str, text: str) -> NumericValue | None:
    quantity = _parse_quantity(text)
    if quantity is None:
        return None
    return NumericValue(raw=raw, kind=ValueKind.SINGLE, value=float(quantity.value))


# ---------------------------------------------------------------------------
# Flag derivation
# ---------------------------------------------------------------------------
def derive_flag(value: NumericValue, reference: NumericValue | None) -> ResultFlag:
    """Position ``value`` against ``reference``, or answer UNKNOWN.

    UNKNOWN is returned whenever the comparison is not certain -- an unparsed
    value, an absent range, or a bound that does not settle the question (a value
    of ``<50`` against a range of ``10 - 20`` could be anywhere). Returning
    NORMAL by default would be the dangerous choice, since "normal" is the answer
    a reader is least likely to double-check.

    Note that the direction of a one-sided reference bound is respected rather
    than assumed: HDL cholesterol against ``> 40`` flags LOW when it falls below,
    which is the opposite of how cholesterol bounds usually read.
    """
    if reference is None or not reference.is_parsed or not value.is_parsed:
        return ResultFlag.UNKNOWN

    if value.kind is ValueKind.SINGLE and value.value is not None:
        return _flag_single(value.value, reference)
    if value.kind is ValueKind.BOUNDED and value.value is not None:
        return _flag_bounded(value.comparator, value.value, reference)
    # A value that is itself a range cannot be positioned against another range.
    return ResultFlag.UNKNOWN


def _flag_single(observed: float, reference: NumericValue) -> ResultFlag:
    if reference.kind is ValueKind.RANGE:
        if reference.low is None or reference.high is None:
            return ResultFlag.UNKNOWN
        if observed < reference.low:
            return ResultFlag.LOW
        if observed > reference.high:
            return ResultFlag.HIGH
        return ResultFlag.NORMAL

    if reference.kind is ValueKind.BOUNDED and reference.value is not None:
        bound = reference.value
        match reference.comparator:
            case Comparator.LT:
                return ResultFlag.HIGH if observed >= bound else ResultFlag.NORMAL
            case Comparator.LTE:
                return ResultFlag.HIGH if observed > bound else ResultFlag.NORMAL
            case Comparator.GT:
                return ResultFlag.LOW if observed <= bound else ResultFlag.NORMAL
            case Comparator.GTE:
                return ResultFlag.LOW if observed < bound else ResultFlag.NORMAL
            case _:
                return ResultFlag.UNKNOWN

    # A reference printed as a single number states a target, not an interval.
    return ResultFlag.UNKNOWN


def _flag_bounded(
    comparator: Comparator | None,
    bound: float,
    reference: NumericValue,
) -> ResultFlag:
    """Flag a one-sided value (``<0.5``, ``>1000``) against a reference.

    Only conclusions that hold for *every* value satisfying the bound are
    returned; everything else is UNKNOWN.
    """
    if comparator is None:
        return ResultFlag.UNKNOWN
    upper_bounded = comparator in (Comparator.LT, Comparator.LTE)

    if reference.kind is ValueKind.RANGE:
        if reference.low is None or reference.high is None:
            return ResultFlag.UNKNOWN
        # "<0.5" is certainly LOW only if 0.5 is at or below the range floor.
        if upper_bounded and bound <= reference.low:
            return ResultFlag.LOW
        if not upper_bounded and bound >= reference.high:
            return ResultFlag.HIGH
        return ResultFlag.UNKNOWN

    if reference.kind is ValueKind.BOUNDED and reference.value is not None:
        reference_bounded_above = reference.comparator in (Comparator.LT, Comparator.LTE)
        reference_bound = reference.value
        if upper_bounded and reference_bounded_above:
            # "<0.5" against "< 6.0": every satisfying value is inside range.
            return ResultFlag.NORMAL if bound <= reference_bound else ResultFlag.UNKNOWN
        if not upper_bounded and not reference_bounded_above:
            return ResultFlag.NORMAL if bound >= reference_bound else ResultFlag.UNKNOWN
        if not upper_bounded and reference_bounded_above:
            # ">1000" against "< 34": every satisfying value exceeds the bound.
            return ResultFlag.HIGH if bound >= reference_bound else ResultFlag.UNKNOWN
        # upper_bounded value against a lower-bounded reference: "<10" vs "> 40"
        return ResultFlag.LOW if bound <= reference_bound else ResultFlag.UNKNOWN

    return ResultFlag.UNKNOWN
