"""Unit standardization, and the factor-1 rule that keeps it safe."""

from __future__ import annotations

import pytest

from services import units

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        # spelling and case
        ("g/dL", "g/dL"),
        ("g/dl", "g/dL"),
        ("G/DL", "g/dL"),
        ("gm/dl", "g/dL"),
        ("gm%", "g/dL"),
        ("mg/dL", "mg/dL"),
        ("mg/dl", "mg/dL"),
        ("(mg/dL)", "mg/dL"),  # brackets stripped
        ("  mg/dl  ", "mg/dL"),
        # cubic millimetre IS a microlitre -- a factor-1 identity
        ("/cumm", "/µL"),
        ("/cu.mm", "/µL"),
        ("cells/cumm", "/µL"),
        ("/mm3", "/µL"),
        ("x10^6/uL", "10^6/µL"),
        ("mill/cumm", "10^6/µL"),
        # every spelling of "millimetres in the first hour"
        ("mm in 1st hr", "mm/hr"),
        ("mm/hour", "mm/hr"),
        # micro sign variants fold together; mIU/L IS µIU/mL
        ("uIU/mL", "µIU/mL"),
        ("µIU/mL", "µIU/mL"),
        ("mIU/L", "µIU/mL"),
        ("fl", "fL"),
        ("percent", "%"),
    ],
)
def test_standardizes_unit_spellings(raw: str, canonical: str) -> None:
    assert units.normalize_unit(raw) == canonical
    assert canonical in units.CANONICAL_UNITS
    assert units.is_recognized_unit(raw) is True


@pytest.mark.parametrize("absent", ["-", "--", "N/A", "nil", "", "   ", None])
def test_absent_unit_is_none(absent: str | None) -> None:
    assert units.normalize_unit(absent) is None
    assert units.is_absent(absent) is True
    assert units.is_recognized_unit(absent) is False


@pytest.mark.parametrize("unknown", ["zorp/L", "widgets", "µSomething/dL"])
def test_unknown_units_pass_through_untouched(unknown: str) -> None:
    """Dropping a unit loses readable information; inventing one fabricates it."""
    assert units.normalize_unit(unknown) == unknown
    assert units.is_recognized_unit(unknown) is False


@pytest.mark.parametrize("numeric", ["720.00", "5", "12,500", "1.2"])
def test_numbers_are_never_treated_as_units(numeric: str) -> None:
    """Guards the report parser: a bare number in the reference column must not be
    picked up as this row's unit."""
    assert units.is_recognized_unit(numeric) is False


def test_only_factor_one_identities_are_ever_mapped() -> None:
    """The rule that keeps a unit bug from becoming a magnitude bug.

    Every alias in the table must be numerically identical to its canonical form,
    so a wrong entry can mislabel a unit but can never misstate a value. These
    pairs differ by a factor of 10 or 100 and must therefore NOT be unified.
    """
    for left, right in [
        ("g/L", "g/dL"),
        ("mg/L", "mg/dL"),
        ("ng/dL", "ng/mL"),
        ("µg/dL", "mg/dL"),
    ]:
        assert units.normalize_unit(left) != units.normalize_unit(right), (
            f"{left} and {right} differ by a conversion factor and must not be "
            "collapsed onto one canonical unit"
        )
