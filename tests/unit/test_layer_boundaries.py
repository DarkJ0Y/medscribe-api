"""Architecture tests: the three-layer rule, enforced mechanically.

Ruff's TID251 already bans the framework in `services/` and `adapters/` at lint
time. These tests exist because a lint rule that is not run does not hold, and
because they check something ruff cannot: what a module actually *loads* at import
time, including through a transitive `__init__`.

The import scan uses the AST rather than text search. An earlier hand-rolled grep
reported false positives by matching the banned names inside the very docstrings
that describe the rule.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from config.settings import PROJECT_ROOT

pytestmark = pytest.mark.unit

#: Nothing in the domain may name any of these.
FRAMEWORK_MODULES = frozenset({"fastapi", "starlette"})
PROVIDER_SDKS = frozenset({"openai", "pytesseract", "PIL", "easyocr", "cv2", "numpy", "torch"})


def _imported_roots(path: Path) -> set[str]:
    """Top-level package names imported by a module, per its AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize(
    "module_path",
    sorted((PROJECT_ROOT / "services").glob("*.py")),
    ids=lambda path: path.name,
)
def test_services_never_import_framework_or_provider_sdk(module_path: Path) -> None:
    """`services/` is the domain: no HTTP, no provider SDKs, not even pydantic."""
    banned = FRAMEWORK_MODULES | PROVIDER_SDKS
    offending = _imported_roots(module_path) & banned

    assert not offending, (
        f"{module_path.name} imports {sorted(offending)}. The domain must reach the "
        "outside world only through the Protocols in services/ports.py."
    )


@pytest.mark.parametrize(
    "module_path",
    sorted((PROJECT_ROOT / "adapters").rglob("*.py")),
    ids=lambda path: str(path.relative_to(PROJECT_ROOT)),
)
def test_adapters_never_import_the_web_framework(module_path: Path) -> None:
    """Adapters may hold provider SDKs -- that is their job -- but never HTTP types."""
    offending = _imported_roots(module_path) & FRAMEWORK_MODULES

    assert not offending, (
        f"{module_path.relative_to(PROJECT_ROOT)} imports {sorted(offending)}. "
        "Adapters are driven by the domain and must not know about HTTP."
    )


def test_importing_the_domain_loads_nothing_external() -> None:
    """Runs in a subprocess: `sys.modules` in the test process is already polluted by
    the app, so a check here would pass no matter what services actually import."""
    script = """
import sys
before = set(sys.modules)
import services.domain, services.errors, services.ports, services.normalizer
import services.units, services.report_parser, services.upload_policy
import services.transcription_service, services.document_service
banned = {"fastapi", "starlette", "openai", "pytesseract", "PIL", "pydantic"}
leaked = sorted({m.split(".")[0] for m in set(sys.modules) - before} & banned)
print(",".join(leaked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = completed.stdout.strip()

    assert leaked == "", f"importing services/ pulled in {leaked}"


def test_mock_mode_loads_no_provider_sdk() -> None:
    """D4's promise: the default deployment needs no provider SDK installed.

    Checked in a subprocess for the same reason, and it matters more here -- the SDKs
    *are* installed in this dev environment, so an in-process check could not tell
    whether the app loaded them or a test did.
    """
    script = """
import sys
from main import create_app
create_app()
banned = {"openai", "pytesseract", "PIL", "easyocr", "torch"}
leaked = sorted({m.split(".")[0] for m in sys.modules} & banned)
print(",".join(leaked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        # Inherit the environment: a bare env dict leaves Windows Python unable to
        # start. Only the toggle is overridden, so a developer's .env cannot
        # accidentally flip this test into real mode.
        env={**os.environ, "USE_MOCK_ADAPTERS": "true"},
    )
    # The app logs JSON to stdout on startup; the printed result is the last line.
    printed = completed.stdout.strip().splitlines()
    leaked = printed[-1] if printed else ""

    assert leaked == "", f"a mock-mode app loaded provider SDKs: {leaked}"
