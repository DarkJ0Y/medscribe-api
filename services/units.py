"""Canonical unit vocabulary + alias table for the unit standardizer.

    "gm/dl", "g/dL", "G/DL"   -> "g/dL"
    "10^3/uL", "x10 3/ul"     -> "10^3/µL"

Unknown units pass through untouched and are reported as non-canonical; the
standardizer never invents a unit it does not recognize.

The one rule that matters
------------------------
**An alias may only map units whose conversion factor is exactly 1.** Spelling is
standardized; magnitude is never touched. So ``/cumm -> /µL`` is permitted
(1 mm³ *is* 1 µL) and ``µIU/mL -> µIU/mL`` absorbs ``mIU/L`` (numerically
identical), but nothing here will ever turn ``g/L`` into ``g/dL``, because that
would require rescaling the value — and a rescaled clinical number that looks
authoritative is exactly the failure mode this service is built to avoid. If a
unit needs arithmetic to convert, it does not belong in this table.
"""

from __future__ import annotations

import re
from typing import Final

#: Units this service is willing to assert as canonical output.
CANONICAL_UNITS: Final[frozenset[str]] = frozenset(
    {
        "g/dL",
        "mg/dL",
        "mg/L",
        "µg/dL",
        "ng/dL",
        "ng/mL",
        "pg/mL",
        "pg",
        "fL",
        "/µL",
        "10^3/µL",
        "10^6/µL",
        "%",
        "mm/hr",
        "IU/mL",
        "IU/L",
        "µIU/mL",
        "U/L",
        "mmol/L",
        "mEq/L",
        "mL/min",
        "ratio",
    }
)

#: Cell contents meaning "no unit given". Lab reports print a dash or leave the
#: column blank; both mean absent, not unknown.
ABSENT_MARKERS: Final[frozenset[str]] = frozenset(
    {"", "-", "--", "---", "—", "–", "n/a", "na", "n.a.", "nil", "none", "?"}
)

_MICRO_CHARS: Final[str] = "µμµμ"

# Keys are already in _key() form: lower-cased, micro-folded to "u", whitespace
# and enclosing brackets removed.
_ALIASES: Final[dict[str, str]] = {
    # mass concentration
    "g/dl": "g/dL",
    "gm/dl": "g/dL",
    "gms/dl": "g/dL",
    "gm/di": "g/dL",  # common OCR confusion: l -> i
    "gm%": "g/dL",
    "g%": "g/dL",
    "mg/dl": "mg/dL",
    "mgs/dl": "mg/dL",
    "mg/di": "mg/dL",
    "mg%": "mg/dL",
    "mg/l": "mg/L",
    "ug/dl": "µg/dL",
    "mcg/dl": "µg/dL",
    "ng/dl": "ng/dL",
    "ng/ml": "ng/mL",
    "pg/ml": "pg/mL",
    "pg": "pg",
    # volume
    "fl": "fL",
    # counts per volume -- 1 mm^3 == 1 uL exactly, so these are spelling variants
    "/cumm": "/µL",
    "/cu.mm": "/µL",
    "/cu.m.m": "/µL",
    "/cmm": "/µL",
    "/mm3": "/µL",
    "/mm^3": "/µL",
    "/ul": "/µL",
    "cells/cumm": "/µL",
    "cells/ul": "/µL",
    "count/cumm": "/µL",
    "10^3/ul": "10^3/µL",
    "x10^3/ul": "10^3/µL",
    "10*3/ul": "10^3/µL",
    "10^3/cumm": "10^3/µL",
    "thou/ul": "10^3/µL",
    "thousand/cumm": "10^3/µL",
    "10^6/ul": "10^6/µL",
    "x10^6/ul": "10^6/µL",
    "10*6/ul": "10^6/µL",
    "10^6/cumm": "10^6/µL",
    "mill/cumm": "10^6/µL",
    "million/cumm": "10^6/µL",
    "mil/cumm": "10^6/µL",
    # proportion
    "%": "%",
    "percent": "%",
    # sedimentation rate -- all spellings of "mm in the first hour"
    "mm/hr": "mm/hr",
    "mm/hour": "mm/hr",
    "mmin1sthr": "mm/hr",
    "mmin1sthour": "mm/hr",
    "mm/1sthr": "mm/hr",
    "mminthe1sthour": "mm/hr",
    "mminfirsthour": "mm/hr",
    "mm1sthr": "mm/hr",
    # activity / international units
    "iu/ml": "IU/mL",
    "iu/l": "IU/L",
    "u/l": "U/L",
    "u/ml": "IU/mL",
    # 1 uIU/mL == 1 mIU/L exactly
    "uiu/ml": "µIU/mL",
    "miu/l": "µIU/mL",
    "uiu/l": "µIU/mL",
    # molar
    "mmol/l": "mmol/L",
    "meq/l": "mEq/L",
    "mmol/mol": "mmol/mol",
    # clearance
    "ml/min": "mL/min",
    # dimensionless
    "ratio": "ratio",
    "index": "ratio",
}

_BRACKETS = re.compile(r"[()\[\]{}]")
_WHITESPACE = re.compile(r"\s+")


def _key(raw: str) -> str:
    """Fold a unit string to its lookup key.

    Case, whitespace, enclosing brackets and the several Unicode micro signs are
    all noise from OCR's point of view; folding them here means the alias table
    holds one entry per real unit instead of one per typographic variant.
    """
    text = _BRACKETS.sub("", raw.strip())
    for char in _MICRO_CHARS:
        text = text.replace(char, "u")
    text = _WHITESPACE.sub("", text).lower()
    return text.rstrip(".")


def is_absent(raw: str | None) -> bool:
    """True when the cell means "nothing was printed here"."""
    if raw is None:
        return True
    return raw.strip().lower() in ABSENT_MARKERS


def is_recognized_unit(raw: str | None) -> bool:
    """True only for units in the alias table or already canonical.

    Used by the lab-report discriminator, so a false positive here lets a
    supermarket receipt through. Keep it strict.
    """
    if raw is None or is_absent(raw):
        return False
    return _key(raw) in _ALIASES or raw.strip() in CANONICAL_UNITS


def normalize_unit(raw: str | None) -> str | None:
    """Standardize a unit's spelling, or pass it through unchanged.

    Returns ``None`` for an absent unit, the canonical spelling for a recognized
    one, and the stripped original for anything else. Passing an unknown unit
    through is deliberate: dropping it would lose information the caller can still
    read, and guessing at it would fabricate one.
    """
    if raw is None or is_absent(raw):
        return None
    stripped = raw.strip()
    if stripped in CANONICAL_UNITS:
        return stripped
    return _ALIASES.get(_key(stripped), stripped)
