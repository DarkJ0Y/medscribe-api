"""Re-exports mirroring pytesseract's own ``__init__``.

``from . import pytesseract`` is required, not decorative: the adapter reaches the
mutable global through ``pytesseract.pytesseract.tesseract_cmd``, and mypy only
resolves a submodule as an attribute of its package when the package imports it
explicitly.

Re-export is declared via ``__all__`` rather than redundant ``X as X`` aliases.
Both satisfy mypy's ``no_implicit_reexport`` (which ``--strict`` enables), but
ruff's import sorter rewrites the alias form into one block per name, which is
markedly less readable.
"""

from . import pytesseract
from .pytesseract import (
    ALTONotSupported,
    Output,
    TesseractError,
    TesseractNotFoundError,
    TSVNotSupported,
    get_languages,
    get_tesseract_version,
    image_to_data,
    image_to_string,
)

__all__ = [
    "ALTONotSupported",
    "Output",
    "TSVNotSupported",
    "TesseractError",
    "TesseractNotFoundError",
    "get_languages",
    "get_tesseract_version",
    "image_to_data",
    "image_to_string",
    "pytesseract",
]
