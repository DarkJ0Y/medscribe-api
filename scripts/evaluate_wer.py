#!/usr/bin/env python
"""Score every reference-bearing fixture and print a word error rate table.

    python scripts/evaluate_wer.py                    # in-process, no server needed
    python scripts/evaluate_wer.py --base-url http://localhost:8000
    python scripts/evaluate_wer.py --json             # machine-readable
    python scripts/evaluate_wer.py --check            # non-zero exit if WER regresses

Without ``--base-url`` the FastAPI app is driven in-process through Starlette's
test client, so there is nothing to start first. With one, real HTTP requests go to
a running service -- which is how you check a container rather than a code path.

What this measures, and what it does not
----------------------------------------
This is an evaluation of the **pipeline**, not of a speech model. Under the default
mock adapters the transcripts are replayed from ``testdata/transcription/``, so the
WER shown is the error already recorded in each fixture -- it verifies that
normalization, alignment, segment filtering and serialization all preserve the
expected score end to end. It says nothing about how well any real ASR model
performs.

Point it at a service running with ``USE_MOCK_ADAPTERS=false`` and the same
fixtures become a genuine model evaluation, because then the transcript really is
the provider's own output scored against the fixture's reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = PROJECT_ROOT / "testdata" / "transcription"

#: Regression baselines for ``--check``, pinned to the committed fixtures. Kept here
#: rather than derived from the run so that a silent change in scoring is caught
#: instead of being reported as the new normal.
EXPECTED_WER: dict[str, float] = {
    "en_clinical_cardiac": 0.0,
    "en_clinical_hypertension": 7 / 26,
    "en_clinical_oncology": 8 / 26,
}
TOLERANCE = 5e-4


def reference_bearing_fixtures() -> list[tuple[str, dict[str, Any]]]:
    """Every fixture that declares a reference transcript, in manifest order."""
    manifest = json.loads((FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8"))
    found: list[tuple[str, dict[str, Any]]] = []
    for name, entry in manifest["fixtures"].items():
        payload = json.loads(
            (FIXTURES_DIR / entry["response"]).read_text(encoding="utf-8")
        )
        if payload.get("reference_transcript"):
            found.append((name, entry))
    return found


def _post_in_process(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from config.settings import Settings
    from main import create_app

    global _CLIENT
    if _CLIENT is None:
        _CLIENT = TestClient(create_app(Settings(_env_file=None)), raise_server_exceptions=False)
    media = FIXTURES_DIR / entry["media"]
    response = _CLIENT.post(
        "/api/v1/transcribe",
        files={"file": (media.name, media.read_bytes(), "audio/wav")},
        data={"language": "en"},
    )
    return dict(response.json())


_CLIENT: Any = None


def _post_over_http(base_url: str, name: str, entry: dict[str, Any]) -> dict[str, Any]:
    import httpx

    media = FIXTURES_DIR / entry["media"]
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/v1/transcribe",
        files={"file": (media.name, media.read_bytes(), "audio/wav")},
        data={"language": "en"},
        timeout=60.0,
    )
    return dict(response.json())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-url",
        help="score against a running service instead of driving the app in-process",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any WER drifts from its recorded baseline",
    )
    args = parser.parse_args(argv)

    fixtures = reference_bearing_fixtures()
    if not fixtures:
        print("no fixtures declare a reference_transcript", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for name, entry in fixtures:
        body = (
            _post_over_http(args.base_url, name, entry)
            if args.base_url
            else _post_in_process(name, entry)
        )
        score = body.get("word_error_rate")
        if score is None:
            print(f"{name}: the service returned no word_error_rate", file=sys.stderr)
            return 1
        rows.append(
            {
                "fixture": name,
                "provider": body.get("provider"),
                "detected_language": body.get("detected_language"),
                **{
                    key: score[key]
                    for key in (
                        "wer",
                        "substitutions",
                        "deletions",
                        "insertions",
                        "hits",
                        "reference_words",
                        "hypothesis_words",
                        "exact_match",
                    )
                },
                "transcript": body.get("transcript", ""),
            }
        )

    total_errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in rows)
    total_reference = sum(r["reference_words"] for r in rows)
    # Corpus WER is errors over TOTAL reference words, not the mean of per-file rates:
    # averaging rates would weight a six-word clip the same as a six-hundred-word one.
    corpus_wer = total_errors / total_reference if total_reference else None

    if args.json:
        print(
            json.dumps(
                {
                    "source": args.base_url or "in-process",
                    "corpus_wer": corpus_wer,
                    "total_errors": total_errors,
                    "total_reference_words": total_reference,
                    "results": rows,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"\nWord error rate  ({args.base_url or 'in-process ASGI'})")
        print("=" * 88)
        print(f"{'fixture':28} {'WER':>8} {'S':>4} {'D':>4} {'I':>4} {'hits':>5} "
              f"{'ref':>4} {'hyp':>4}  exact")
        print("-" * 88)
        for row in rows:
            print(
                f"{row['fixture']:28} {row['wer']:8.4f} {row['substitutions']:4} "
                f"{row['deletions']:4} {row['insertions']:4} {row['hits']:5} "
                f"{row['reference_words']:4} {row['hypothesis_words']:4}  "
                f"{'yes' if row['exact_match'] else 'no'}"
            )
        print("-" * 88)
        assert corpus_wer is not None
        print(
            f"{'CORPUS (errors / ref words)':28} {corpus_wer:8.4f} "
            f"{'':14} {total_errors:>5} of {total_reference}"
        )
        print("=" * 88)

    if args.check:
        failures = []
        for row in rows:
            baseline = EXPECTED_WER.get(str(row["fixture"]))
            if baseline is None:
                continue
            if abs(float(row["wer"]) - baseline) > TOLERANCE:
                failures.append((row["fixture"], baseline, row["wer"]))
        if failures:
            print("\nWER REGRESSION:", file=sys.stderr)
            for fixture, baseline, actual in failures:
                print(f"  {fixture}: expected {baseline:.4f}, got {actual:.4f}", file=sys.stderr)
            return 1
        print(f"\ncheck: all {len(EXPECTED_WER)} baselines hold")

    return 0


if __name__ == "__main__":
    sys.exit(main())
