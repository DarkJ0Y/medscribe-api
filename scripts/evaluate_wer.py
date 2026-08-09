#!/usr/bin/env python
"""Score every audio sample in the repository and print a word error rate table.

    python scripts/evaluate_wer.py                    # committed fixture corpus
    python scripts/evaluate_wer.py --samples          # + the sampled HF corpus
    python scripts/evaluate_wer.py --base-url http://localhost:8000
    python scripts/evaluate_wer.py --json             # machine-readable
    python scripts/evaluate_wer.py --check            # non-zero exit if WER regresses

Without ``--base-url`` the FastAPI app is driven in-process through Starlette's test
client, so there is nothing to start first. With one, real HTTP requests go to a
running service -- which is how you check a container rather than a code path.

Every sample is listed, including the ones that cannot be scored, because a table
that silently omits them invites the reader to assume the corpus is smaller and
better-referenced than it is. Three situations are distinguished:

**scored** -- the fixture declares ``reference_transcript`` and the replayed
transcript is the ASR output recorded *for that reference*. WER is meaningful as an
end-to-end check of normalization, alignment, segment filtering and serialization.

**no reference** -- nothing to score against. WER is undefined, not zero. This is
also the normal production case for every live request.

**not comparable** -- a reference exists, but the transcript did not come from
transcribing this audio. That is what happens to the sampled corpus under the mock
adapters: the mock replays a fixture chosen by content hash or filename, so the
"transcript" belongs to a different recording entirely. The resulting number is
printed for illustration and excluded from every aggregate, because it measures the
distance between two unrelated texts rather than transcription accuracy.

Run against a service with ``USE_MOCK_ADAPTERS=false`` and the LibriSpeech rows
become genuinely comparable, since the transcript is then the provider's own output
scored against an exact utterance-level reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import wer  # noqa: E402  (import after sys.path setup)

FIXTURES_DIR = PROJECT_ROOT / "testdata" / "transcription"
SAMPLES_DIR = PROJECT_ROOT / "testdata" / "audio"
SAMPLES_MANIFEST = SAMPLES_DIR / "audio_ground_truth.json"

#: Regression baselines for ``--check``, pinned to the committed fixtures. Kept here
#: rather than derived from the run so that a silent change in scoring is caught
#: instead of being reported as the new normal.
EXPECTED_WER: dict[str, float] = {
    "en_clinical_cardiac": 0.0,
    "en_clinical_hypertension": 7 / 26,
    "en_clinical_oncology": 8 / 26,
}
TOLERANCE = 5e-4

SCORED = "scored"
NO_REFERENCE = "no reference"
NOT_COMPARABLE = "not comparable"


@dataclass
class Row:
    """One audio sample and whatever could honestly be said about its accuracy."""

    sample: str
    group: str
    status: str
    provider: str
    language: str
    score: dict[str, Any] | None = None
    note: str = ""

    @property
    def counts_toward_corpus(self) -> bool:
        return self.status == SCORED and self.score is not None


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
class Poster:
    """Posts audio to /api/v1/transcribe, in-process or over HTTP."""

    def __init__(self, base_url: str | None) -> None:
        self._base_url = base_url
        self._client: Any = None

    def post(self, path: Path, language: str) -> dict[str, Any]:
        payload = path.read_bytes()
        files = {"file": (path.name, payload, "audio/wav")}
        data = {"language": language}

        if self._base_url:
            import httpx

            response = httpx.post(
                f"{self._base_url.rstrip('/')}/api/v1/transcribe",
                files=files,
                data=data,
                timeout=120.0,
            )
            body: dict[str, Any] = response.json()
            return body

        if self._client is None:
            from fastapi.testclient import TestClient

            from config.settings import Settings
            from main import create_app

            self._client = TestClient(
                create_app(Settings(_env_file=None)), raise_server_exceptions=False
            )
        result: dict[str, Any] = self._client.post(
            "/api/v1/transcribe", files=files, data=data
        ).json()
        return result


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
def fixture_rows(poster: Poster) -> list[Row]:
    """Every committed transcription fixture, scored where a reference exists."""
    manifest = json.loads((FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8"))
    rows: list[Row] = []

    for name in sorted(manifest["fixtures"]):
        entry = manifest["fixtures"][name]
        recorded = json.loads((FIXTURES_DIR / entry["response"]).read_text(encoding="utf-8"))
        media = FIXTURES_DIR / entry["media"]
        language = "bn" if name.startswith("bn") else "en"

        body = poster.post(media, language)
        provider = str(body.get("provider", "?"))
        detected = str(body.get("detected_language", "?"))

        if not recorded.get("reference_transcript"):
            rows.append(
                Row(
                    sample=name,
                    group="fixture",
                    status=NO_REFERENCE,
                    provider=provider,
                    language=detected,
                    note="fixture declares no reference_transcript",
                )
            )
            continue

        # Taken from the API response, not recomputed here, so the table exercises
        # the same code path a caller would see.
        score = body.get("word_error_rate")
        rows.append(
            Row(
                sample=name,
                group="fixture",
                status=SCORED if score else NO_REFERENCE,
                provider=provider,
                language=detected,
                score=score,
            )
        )
    return rows


def sample_rows(poster: Poster) -> list[Row]:
    """The sampled Hugging Face corpus, if it has been downloaded."""
    if not SAMPLES_MANIFEST.is_file():
        return []

    manifest = json.loads(SAMPLES_MANIFEST.read_text(encoding="utf-8"))
    rows: list[Row] = []

    for sample in manifest["samples"]:
        name = str(sample["file"])
        media = SAMPLES_DIR / name
        if not media.is_file():
            continue
        language = str(sample.get("language", "en"))
        reference = sample.get("transcript")
        alignment = str(sample.get("transcript_alignment", "none"))

        body = poster.post(media, language)
        provider = str(body.get("provider", "?"))
        detected = str(body.get("detected_language", "?"))

        if not reference or alignment != "exact":
            rows.append(
                Row(
                    sample=name,
                    group="sample",
                    status=NO_REFERENCE,
                    provider=provider,
                    language=detected,
                    note=(
                        "long-form source with one whole-recording transcription; "
                        "this clip has no aligned reference"
                    ),
                )
            )
            continue

        # Scored client-side: the service has no reference for these files, so its own
        # word_error_rate is null. Whether the number means anything depends on
        # whether the transcript actually came from this audio.
        computed = wer.compute(str(reference), str(body.get("transcript", "")))
        replayed = provider.startswith("mock-")
        rows.append(
            Row(
                sample=name,
                group="sample",
                status=NOT_COMPARABLE if replayed else SCORED,
                provider=provider,
                language=detected,
                score=None
                if computed is None
                else {
                    "wer": round(computed.wer, 4),
                    "substitutions": computed.substitutions,
                    "deletions": computed.deletions,
                    "insertions": computed.insertions,
                    "hits": computed.hits,
                    "reference_words": computed.reference_words,
                    "hypothesis_words": computed.hypothesis_words,
                    "exact_match": computed.is_exact_match,
                },
                note=(
                    "mock adapter replayed an unrelated fixture; this number is not "
                    "transcription accuracy"
                )
                if replayed
                else "exact utterance-level reference",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
WIDTH = 96


def _render(title: str, rows: list[Row]) -> None:
    print(f"\n{title}")
    print("=" * WIDTH)
    print(
        f"{'sample':28} {'WER':>8} {'S':>4} {'D':>4} {'I':>4} {'hits':>5} "
        f"{'ref':>4} {'hyp':>4}  status"
    )
    print("-" * WIDTH)
    for row in rows:
        if row.score is None:
            print(
                f"{row.sample:28} {'-':>8} {'-':>4} {'-':>4} {'-':>4} {'-':>5} "
                f"{'-':>4} {'-':>4}  {row.status}"
            )
            continue
        s = row.score
        print(
            f"{row.sample:28} {float(s['wer']):8.4f} {s['substitutions']:4} "
            f"{s['deletions']:4} {s['insertions']:4} {s['hits']:5} "
            f"{s['reference_words']:4} {s['hypothesis_words']:4}  {row.status}"
        )
    print("-" * WIDTH)


def _corpus_line(rows: list[Row], label: str) -> float | None:
    scored = [r for r in rows if r.counts_toward_corpus and r.score is not None]
    if not scored:
        print(f"{label:28} {'-':>8}  no scoreable samples in this group")
        return None
    errors = sum(
        int(r.score["substitutions"]) + int(r.score["deletions"]) + int(r.score["insertions"])
        for r in scored
        if r.score is not None
    )
    reference_words = sum(int(r.score["reference_words"]) for r in scored if r.score is not None)
    corpus = errors / reference_words if reference_words else None
    if corpus is None:
        return None
    print(
        f"{label:28} {corpus:8.4f}  {errors} errors / {reference_words} reference words "
        f"over {len(scored)} of {len(rows)} samples"
    )
    return corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Word error rate across every audio sample.")
    parser.add_argument(
        "--base-url",
        help="score against a running service instead of driving the app in-process",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="also score the sampled Hugging Face corpus in testdata/audio/",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any WER drifts from its recorded baseline",
    )
    args = parser.parse_args(argv)

    poster = Poster(args.base_url)
    fixtures = fixture_rows(poster)
    samples = sample_rows(poster) if args.samples else []

    if args.json:
        print(
            json.dumps(
                {
                    "source": args.base_url or "in-process",
                    "groups": {
                        "fixtures": [vars(r) for r in fixtures],
                        "samples": [vars(r) for r in samples],
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        source = args.base_url or "in-process ASGI"
        print(f"\nWord error rate  ({source})")
        _render(f"Committed fixture corpus -- testdata/transcription/  ({len(fixtures)} samples)",
                fixtures)
        _corpus_line(fixtures, "CORPUS (fixtures)")
        print("=" * WIDTH)

        if args.samples:
            if samples:
                _render(
                    "Sampled Hugging Face corpus -- testdata/audio/  "
                    f"({len(samples)} samples)",
                    samples,
                )
                _corpus_line(samples, "CORPUS (samples)")
                print("=" * WIDTH)
                for status in (NOT_COMPARABLE, NO_REFERENCE, SCORED):
                    matching = [r for r in samples if r.status == status]
                    if matching:
                        print(f"  {len(matching):2} {status:16} {matching[0].note}")
            else:
                print(
                    "\nSampled corpus not found. It is gitignored; fetch it with:\n"
                    "    python scripts/download_testdata.py"
                )

    if args.check:
        by_name = {r.sample: r for r in fixtures}
        failures: list[tuple[str, float, float]] = []
        for name, baseline in EXPECTED_WER.items():
            row = by_name.get(name)
            if row is None or row.score is None:
                failures.append((name, baseline, float("nan")))
                continue
            actual = float(row.score["wer"])
            if abs(actual - baseline) > TOLERANCE:
                failures.append((name, baseline, actual))
        if failures:
            print("\nWER REGRESSION:", file=sys.stderr)
            for name, baseline, actual in failures:
                print(f"  {name}: expected {baseline:.4f}, got {actual:.4f}", file=sys.stderr)
            return 1
        print(f"\ncheck: all {len(EXPECTED_WER)} baselines hold")

    return 0


if __name__ == "__main__":
    sys.exit(main())
