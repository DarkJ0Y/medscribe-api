"""Partial type stubs for the parts of pytesseract this project uses.

Hand-written because pytesseract ships no ``py.typed`` marker and no ``types-``
stub package exists on PyPI. It is the only untyped dependency in the tree --
fastapi, pydantic, starlette, openai and Pillow all ship their own types.

The alternative was ``ignore_missing_imports``, which would turn the whole module
into ``Any`` and silently disable checking of every call into it.

Declaring the real exception hierarchy earns its keep for a second reason: writing
it down is what exposed that ``TesseractError`` derives from ``RuntimeError``, and
therefore that ``adapters/ocr/tesseract.py`` had its ``except RuntimeError`` clause
ordered ahead of ``except TesseractError`` -- making the latter dead code. Note that
mypy does **not** detect that on its own; ``warn_unreachable`` does not model
exception subsumption in ``except`` chains. The bug was found by reading this
hierarchy, and it is guarded by a test rather than by the type checker.

Deliberately partial. Only the symbols the adapter touches are declared, so
reaching for anything else fails type checking and is a prompt to extend this
file rather than a silent fall through to ``Any``.
"""

from typing import Any

# Module-level global that callers mutate to point at a non-PATH binary; the
# adapter sets it from the TESSERACT_CMD setting.
tesseract_cmd: str

class Output:
    BYTES: str
    DATAFRAME: str
    DICT: str
    STRING: str

class TesseractError(RuntimeError):
    """Note the base class: it is RuntimeError, not Exception.

    An ``except RuntimeError`` clause placed before ``except TesseractError``
    therefore shadows it completely.
    """

    status: int
    message: str
    def __init__(self, status: int, message: str) -> None: ...

class TesseractNotFoundError(OSError):
    # Takes no arguments: it builds its own message from tesseract_cmd. Inheriting
    # OSError's permissive __init__ would let a call site pass one and typecheck.
    def __init__(self) -> None: ...
class TSVNotSupported(Exception): ...
class ALTONotSupported(Exception): ...

def image_to_data(
    image: Any,
    lang: str | None = ...,
    config: str = ...,
    nice: int = ...,
    # The return shape follows this argument (a dict for Output.DICT, a string for
    # Output.STRING), so the return type stays Any rather than claiming a precision
    # the real function does not have.
    output_type: str = ...,
    timeout: float = ...,
    pandas_config: dict[str, Any] | None = ...,
) -> Any: ...
def image_to_string(
    image: Any,
    lang: str | None = ...,
    config: str = ...,
    nice: int = ...,
    output_type: str = ...,
    timeout: float = ...,
) -> Any: ...
def get_tesseract_version() -> Any: ...
def get_languages(config: str = ...) -> list[str]: ...
