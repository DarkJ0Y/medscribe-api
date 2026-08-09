#!/usr/bin/env python
"""Collect a small, reproducible sample corpus from three Hugging Face datasets.

    python scripts/download_testdata.py            # 10 of each, 20-second clips
    python scripts/download_testdata.py --force    # re-download over existing files
    python scripts/download_testdata.py --only bn --clip-seconds 30

Populates, relative to the project root::

    testdata/audio/bn_sample_01.wav .. bn_sample_10.wav
    testdata/audio/en_sample_01.wav .. en_sample_10.wav
    testdata/audio/audio_ground_truth.json
    testdata/documents/lab_report_01.png .. lab_report_10.png
    testdata/documents/doc_ground_truth.json

These are *additive* fixtures for exercising the real adapters and for manual
testing. They are deliberately separate from ``testdata/transcription/`` and
``testdata/ocr/``, which are the mock adapters' replay corpus and are indexed by
sha256 in their own manifests -- nothing here touches those.

Why no `datasets`, `pyarrow` or `pandas`
---------------------------------------
All three sources are reachable without them, and this script is meant to run on a
modest machine:

* Two repos store **raw files** (``.jpg`` / ``.wav``), fetched over plain HTTP.
* LibriSpeech is a single parquet, but the HF datasets-server ``/rows`` endpoint
  returns its text plus per-row audio URLs as JSON, so the parquet reader is never
  needed.

Dependencies are therefore the standard library plus **Pillow** (JPEG -> PNG) and
**soundfile** (FLAC -> WAV). Set ``HF_TOKEN`` in the environment if you hit rate
limits; it is sent as a bearer token and never logged.

The Bengali long-form problem
-----------------------------
``IntisarUddin/Bengali_Long_form_ASR`` holds 310 hours in 382 files: 67-74 MB each,
a median of 42 minutes, up to 5.8 hours. Downloading ten whole files would be
~700 MB, and **every one of them would be rejected by the service's own 25 MiB
upload limit**.

So each file is trimmed to the first ``--clip-seconds`` using an HTTP **Range**
request: parse the RIFF header, then fetch only the first N seconds of PCM and
write a fresh, valid WAV. That is ~0.6 MB instead of 70 MB per file.

The unavoidable consequence, recorded explicitly in the ground truth: the dataset's
transcription covers the **whole recording**, so a 20-second clip has no aligned
reference. Those entries carry ``transcript: null`` and
``transcript_alignment: "none"``, with the full text kept alongside under
``full_recording_transcript``. Do not compute WER against them. LibriSpeech rows are
single utterances, so those transcripts *are* exact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
HF_ROWS = "https://datasets-server.huggingface.co/rows"

DOCS_REPO = "hmnshudhmn24/noisy-medical-document-images-ocr"
BENGALI_REPO = "IntisarUddin/Bengali_Long_form_ASR"
LIBRISPEECH_REPO = "hf-internal-testing/librispeech_asr_dummy"

USER_AGENT = "medscribe-testdata-collector/1.0"
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 2.0

#: Hard ceiling per clip, mirroring the service's MAX_AUDIO_BYTES. A fixture that
#: the endpoint would reject as too large is not a useful fixture.
MAX_CLIP_BYTES = 25 * 1024 * 1024

csv.field_size_limit(10 * 1024 * 1024)  # the ground-truth CSVs embed multi-line JSON


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _request(url: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
    """GET ``url`` with retries, optionally a byte range.

    Retries on transport errors and on 429/5xx, which is what HF returns under
    rate limiting. A 4xx other than 429 is not retried -- it will not fix itself.
    """
    headers = {"User-Agent": USER_AGENT}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"  # never logged
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
        if attempt < MAX_ATTEMPTS:
            delay = BACKOFF_SECONDS * attempt
            print(f"    retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s ({last_error})")
            time.sleep(delay)

    raise RuntimeError(f"giving up on {url}: {last_error}")


def repo_revision(repo: str) -> str:
    """Resolve the repo's current commit sha, recorded for provenance.

    Pinning what was actually downloaded is the difference between a reproducible
    corpus and a pile of files that happened to exist one afternoon.
    """
    info = json.loads(_request(f"{HF_API}/{repo}"))
    return str(info.get("sha") or "main")


def fetch_repo_file(repo: str, revision: str, path: str, *, cache: bool = False) -> bytes:
    url = HF_RESOLVE.format(repo=repo, revision=revision, path=urllib.parse.quote(path))
    if not cache:
        return _request(url)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{repo}@{revision}/{path}".encode()).hexdigest()[:16]
    cached = CACHE_DIR / f"{key}-{Path(path).name}"
    if cached.is_file():
        print(f"    cached: {path}")
        return cached.read_bytes()
    payload = _request(url)
    cached.write_bytes(payload)
    return payload


# ---------------------------------------------------------------------------
# WAV trimming over HTTP Range
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WavLayout:
    """Where the audio lives in a RIFF file, and how to interpret it."""

    audio_format: int
    channels: int
    sample_rate: int
    bits_per_sample: int
    data_offset: int
    data_size: int

    @property
    def frame_bytes(self) -> int:
        return self.channels * self.bits_per_sample // 8

    @property
    def is_pcm(self) -> bool:
        return self.audio_format == 1


def parse_wav_layout(header: bytes) -> WavLayout:
    """Walk the RIFF chunk list to locate ``fmt `` and ``data``.

    Chunks are walked rather than assuming the textbook 44-byte header: these files
    carry a ``LIST`` chunk between ``fmt `` and ``data``, so a fixed offset would
    slice into metadata and produce audible garbage in every clip.
    """
    if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")

    fmt: tuple[int, int, int, int] | None = None
    position = 12
    while position + 8 <= len(header):
        chunk_id = header[position : position + 4]
        chunk_size = struct.unpack("<I", header[position + 4 : position + 8])[0]
        body = position + 8

        if chunk_id == b"fmt " and body + 16 <= len(header):
            audio_format, channels, sample_rate, _byte_rate, _align, bits = struct.unpack(
                "<HHIIHH", header[body : body + 16]
            )
            fmt = (audio_format, channels, sample_rate, bits)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError("data chunk precedes fmt chunk")
            return WavLayout(*fmt, data_offset=body, data_size=chunk_size)

        position = body + chunk_size + (chunk_size % 2)  # chunks are word-aligned

    raise ValueError("no data chunk found in the probed header")


def _rms_int16(pcm: bytes, *, max_samples: int = 50_000) -> float:
    """Normalized RMS of 16-bit PCM, on a strided subsample."""
    import array

    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    stride = max(1, len(samples) // max_samples)
    window = samples[::stride]
    total = sum(float(value) * value for value in window)
    return (total / len(window)) ** 0.5 / 32768


#: Fractions through the audio data at which to try a clip. Long-form recordings
#: routinely open with dead air -- an intro card, a pause before speech -- so
#: "the first N seconds" is not reliably "N seconds of speech".
_CLIP_START_FRACTIONS: tuple[float, ...] = (0.0, 0.10, 0.25, 0.40, 0.55, 0.70)

#: RMS a window must reach to count as speech. Comfortably above the service's
#: SILENCE_RMS_THRESHOLD of 0.005, so an accepted clip cannot be reported as silent.
_MIN_CLIP_RMS = 0.01


@dataclass(frozen=True, slots=True)
class Clip:
    payload: bytes
    layout: WavLayout
    duration: float
    start_seconds: float
    rms: float
    probes: int


def fetch_wav_clip(
    repo: str,
    revision: str,
    path: str,
    seconds: float,
    *,
    probe_bytes: int = 64 * 1024,
) -> Clip:
    """Range-fetch ``seconds`` of a remote WAV, skipping past leading silence.

    Candidate windows are tried at increasing offsets and the first one carrying
    real signal wins. Without this, a recording that opens with dead air yields a
    silent "speech" fixture -- which the transcription endpoint would correctly
    report as ``speech_detected: false``, looking like a service bug rather than a
    corpus bug. The chosen offset is returned so it can be recorded.
    """
    url = HF_RESOLVE.format(repo=repo, revision=revision, path=urllib.parse.quote(path))
    layout = parse_wav_layout(_request(url, byte_range=(0, probe_bytes - 1)))

    if not layout.is_pcm:
        raise ValueError(f"unsupported WAV encoding {layout.audio_format} (only PCM is handled)")

    wanted = int(seconds * layout.sample_rate) * layout.frame_bytes
    wanted = min(wanted, layout.data_size, MAX_CLIP_BYTES - 128)
    wanted -= wanted % layout.frame_bytes  # never end mid-frame
    if wanted <= 0:
        raise ValueError("computed clip length is empty")

    best: tuple[bytes, float, float] | None = None  # (pcm, start_seconds, rms)
    probes = 0
    for fraction in _CLIP_START_FRACTIONS:
        probes += 1
        offset = int(layout.data_size * fraction)
        offset -= offset % layout.frame_bytes
        if offset + wanted > layout.data_size:
            break

        start = layout.data_offset + offset
        pcm = _request(url, byte_range=(start, start + wanted - 1))
        pcm = pcm[: len(pcm) - (len(pcm) % layout.frame_bytes)]
        if not pcm:
            continue

        rms = _rms_int16(pcm)
        start_seconds = offset / layout.frame_bytes / layout.sample_rate
        if best is None or rms > best[2]:
            best = (pcm, start_seconds, rms)
        if rms >= _MIN_CLIP_RMS:
            break
        print(f"      silent at {start_seconds:.0f}s (RMS {rms:.4f}), trying further in")

    if best is None:
        raise ValueError("no readable audio window found")

    pcm, start_seconds, rms = best

    import wave  # stdlib, imported here to keep the module import light

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(layout.channels)
        writer.setsampwidth(layout.bits_per_sample // 8)
        writer.setframerate(layout.sample_rate)
        writer.writeframes(pcm)

    duration = len(pcm) / layout.frame_bytes / layout.sample_rate
    return Clip(
        payload=buffer.getvalue(),
        layout=layout,
        duration=duration,
        start_seconds=start_seconds,
        rms=rms,
        probes=probes,
    )


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------
@dataclass
class Report:
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def _write(path: Path, payload: bytes, *, force: bool, report: Report) -> bool:
    if path.exists() and not force:
        report.skipped.append(path.name)
        print(f"    exists, skipping: {path.name}  (--force to overwrite)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    report.written.append(path.name)
    return True


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def collect_documents(out_dir: Path, count: int, *, force: bool, report: Report) -> dict[str, Any]:
    """Medical document images, converted JPEG -> PNG.

    Half bills and half discharge summaries, so the sample spans both document types
    the repo actually contains rather than over-representing whichever sorts first.
    """
    from PIL import Image

    print(f"\n[documents] {DOCS_REPO}")
    revision = repo_revision(DOCS_REPO)
    print(f"  revision: {revision[:12]}")

    categories = [
        ("bill", "bills", "medical_bills_ground_truth.csv"),
        ("discharge_summary", "discharge_summaries", "discharge_summaries_ground_truth.csv"),
    ]
    per_category = [
        count // 2 + (1 if index < count % 2 else 0) for index in range(len(categories))
    ]

    selected: list[dict[str, Any]] = []
    for (doc_type, folder, csv_name), wanted in zip(categories, per_category, strict=True):
        print(f"  reading {csv_name}")
        text = fetch_repo_file(DOCS_REPO, revision, csv_name, cache=True).decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
        # Sorted for determinism: the same 10 files on every run.
        rows.sort(key=lambda row: row["filename"])
        for row in rows[:wanted]:
            selected.append({"doc_type": doc_type, "folder": folder, "row": row})

    samples: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        source = f"{item['folder']}/{item['row']['filename']}"
        target = out_dir / f"lab_report_{index:02d}.png"
        print(f"  [{index:2d}/{len(selected)}] {source}")
        try:
            raw = fetch_repo_file(DOCS_REPO, revision, source)
            with Image.open(io.BytesIO(raw)) as image:
                # CMYK and palette modes do not round-trip predictably into PNG.
                converted = image.convert("RGB") if image.mode in ("CMYK", "P") else image.copy()
                width, height, mode = converted.width, converted.height, converted.mode
                buffer = io.BytesIO()
                converted.save(buffer, format="PNG", optimize=True)
            payload = buffer.getvalue()
        except Exception as error:  # noqa: BLE001 - one bad file must not abort the run
            report.failed.append((target.name, str(error)))
            print(f"      FAILED: {error}")
            continue

        _write(target, payload, force=force, report=report)
        try:
            ground_truth = json.loads(item["row"]["json_data"])
        except json.JSONDecodeError:
            ground_truth = None

        samples.append(
            {
                "file": target.name,
                "source_file": source,
                "source_format": "jpeg",
                "document_type": item["doc_type"],
                # These are bills and discharge summaries, not lab reports: the
                # extraction endpoint should refuse them. Recorded so a test can
                # assert the refusal instead of expecting results.
                "medscribe_expectation": "not_a_lab_report",
                "image": {"width": width, "height": height, "mode": mode},
                "bytes": len(payload),
                "sha256": _digest(payload),
                "ground_truth": ground_truth,
            }
        )

    return {
        "dataset": DOCS_REPO,
        "dataset_revision": revision,
        "warnings": [
            "THESE FIXTURES REQUIRE USE_MOCK_ADAPTERS=false TO MEAN ANYTHING. Under the "
            "default mock adapters every upload falls back to the cbc_report fixture, so "
            "all ten return HTTP 200 with 11 results that have nothing to do with the "
            "image. That fallback IS logged at WARNING level ('No fixture matched "
            "upload'). Set USE_MOCK_ADAPTERS=false and install tesseract before drawing "
            "any conclusion.",
            "This dataset contains medical BILLS and DISCHARGE SUMMARIES. It contains "
            "no laboratory reports, despite the lab_report_NN.png filenames requested "
            "for this corpus. POST /api/v1/documents/extract should refuse all ten "
            "with HTTP 422 not_a_lab_report -- that is correct behaviour, and it makes "
            "these excellent negative fixtures. Each entry records its true "
            "document_type.",
            "Images are noisy/degraded by design (the dataset simulates scan noise), "
            "so they are a realistic stress test for the OCR adapter rather than a "
            "clean-input baseline.",
            "ground_truth holds the dataset's structured JSON (hospital, patient, "
            "diagnosis, procedures with CPT codes and prices). It is not OCR "
            "line-level text, so it cannot be compared directly against raw_line.",
        ],
        "samples": samples,
    }


def collect_bengali(
    out_dir: Path, count: int, clip_seconds: float, *, force: bool, report: Report
) -> list[dict[str, Any]]:
    print(f"\n[bengali] {BENGALI_REPO}")
    revision = repo_revision(BENGALI_REPO)
    print(f"  revision: {revision[:12]}")

    print("  reading dataset_metadata.json (~18 MB, cached after first run)")
    metadata = json.loads(
        fetch_repo_file(BENGALI_REPO, revision, "dataset_metadata.json", cache=True).decode("utf-8")
    )

    # Shortest recordings first: deterministic, and it maximises the fraction of each
    # recording the clip actually represents (though even the shortest is ~4 minutes).
    usable = [row for row in metadata if row.get("file_name") and row.get("duration_min")]
    usable.sort(key=lambda row: (row["duration_min"], row["file_name"]))

    samples: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    # More candidates than needed: sources are skipped when they duplicate an
    # already-accepted clip. The upstream repo genuinely contains repeated audio --
    # audio_240 and audio_247 produce byte-identical windows -- and two identical
    # fixtures waste a slot while letting one bug pass twice.
    attempts_allowed = max(count * 3, count + 10)

    for row in usable[:attempts_allowed]:
        if len(samples) >= count:
            break

        source = f"audio/{row['file_name']}"
        index = len(samples) + 1
        print(
            f"  [{index:2d}/{count}] {source}  "
            f"({row['duration_min']:.1f} min source -> {clip_seconds:g}s clip)"
        )
        try:
            clip = fetch_wav_clip(BENGALI_REPO, revision, source, clip_seconds)
        except Exception as error:  # noqa: BLE001
            report.failed.append((source, str(error)))
            print(f"      FAILED: {error}")
            continue

        digest = _digest(clip.payload)
        if digest in seen:
            print(f"      identical to {seen[digest]} (duplicate upstream audio), skipping")
            continue
        seen[digest] = f"bn_sample_{index:02d}.wav"
        target = out_dir / f"bn_sample_{index:02d}.wav"

        if clip.rms < _MIN_CLIP_RMS:
            print(
                f"      WARNING: best window is near-silent (RMS {clip.rms:.4f}); "
                "keeping it but the endpoint will report speech_detected=false"
            )
        _write(target, clip.payload, force=force, report=report)
        transcript = str(row.get("transcription") or "")
        samples.append(
            {
                "file": target.name,
                "language": "bn",
                "dataset": BENGALI_REPO,
                "dataset_revision": revision,
                "source_file": source,
                "source_id": row.get("original_id"),
                "source_title": row.get("title"),
                "source_duration_minutes": row["duration_min"],
                # Not necessarily 0: leading dead air is skipped, so record where
                # this clip actually starts in the source recording.
                "clip_start_seconds": round(clip.start_seconds, 3),
                "clip_duration_seconds": round(clip.duration, 3),
                "clip_rms": round(clip.rms, 5),
                "clip_windows_probed": clip.probes,
                "sample_rate": clip.layout.sample_rate,
                "channels": clip.layout.channels,
                "bits_per_sample": clip.layout.bits_per_sample,
                "bytes": len(clip.payload),
                "sha256": _digest(clip.payload),
                # The honest part. The dataset's transcription covers the entire
                # recording; this clip is a 20-second window inside it, so there is
                # no aligned reference and none is invented.
                "transcript": None,
                "transcript_alignment": "none",
                "transcript_note": (
                    f"This clip is {round(clip.duration, 1)}s starting at "
                    f"{round(clip.start_seconds)}s of a {row['duration_min']:.1f}-minute "
                    "recording. The dataset provides one transcription for the whole "
                    "recording, not per segment, so no reference transcript exists for "
                    "this clip. Use it for smoke testing and language detection; do NOT "
                    "compute WER against full_recording_transcript."
                ),
                "full_recording_transcript": transcript,
                "full_recording_transcript_chars": len(transcript),
            }
        )

    return samples


def collect_librispeech(
    out_dir: Path, count: int, *, force: bool, report: Report
) -> list[dict[str, Any]]:
    """English utterances via the datasets-server rows API.

    Used in preference to the parquet file so that no parquet reader is needed. Each
    row is a single utterance, so these transcripts are exactly aligned.
    """
    import soundfile

    print(f"\n[english] {LIBRISPEECH_REPO}")
    revision = repo_revision(LIBRISPEECH_REPO)
    print(f"  revision: {revision[:12]}")

    query = urllib.parse.urlencode(
        {
            "dataset": LIBRISPEECH_REPO,
            "config": "clean",
            "split": "validation",
            "offset": 0,
            "length": count,
        }
    )
    payload = json.loads(_request(f"{HF_ROWS}?{query}"))
    rows = payload.get("rows", [])
    print(f"  rows available: {payload.get('num_rows_total')}, taking {len(rows)}")

    samples: list[dict[str, Any]] = []
    for index, entry in enumerate(rows, start=1):
        row = entry["row"]
        target = out_dir / f"en_sample_{index:02d}.wav"
        audio = row.get("audio") or []
        source_url = audio[0]["src"] if audio else None
        print(f"  [{index:2d}/{len(rows)}] {row.get('id')}")
        if not source_url:
            report.failed.append((target.name, "row carried no audio asset"))
            continue

        try:
            encoded = _request(source_url)
            # LibriSpeech ships FLAC; the corpus is specified as WAV, so decode and
            # re-encode as 16-bit PCM rather than renaming the container.
            data, sample_rate = soundfile.read(io.BytesIO(encoded), dtype="int16")
            buffer = io.BytesIO()
            soundfile.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
            wav = buffer.getvalue()
        except Exception as error:  # noqa: BLE001
            report.failed.append((target.name, str(error)))
            print(f"      FAILED: {error}")
            continue

        _write(target, wav, force=force, report=report)
        channels = 1 if data.ndim == 1 else data.shape[1]
        samples.append(
            {
                "file": target.name,
                "language": "en",
                "dataset": LIBRISPEECH_REPO,
                "dataset_revision": revision,
                "source_format": "flac",
                "utterance_id": row.get("id"),
                "speaker_id": row.get("speaker_id"),
                "chapter_id": row.get("chapter_id"),
                "clip_duration_seconds": round(len(data) / sample_rate, 3),
                "sample_rate": sample_rate,
                "channels": channels,
                "bits_per_sample": 16,
                "bytes": len(wav),
                "sha256": _digest(wav),
                # One utterance per row, so the reference is exact.
                "transcript": row.get("text"),
                "transcript_alignment": "exact",
                "transcript_note": (
                    "LibriSpeech reference transcript, upper-case and without "
                    "punctuation, exactly aligned to this clip."
                ),
            }
        )

    return samples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=10, help="samples per source (default 10)")
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=20.0,
        help="length to trim long-form Bengali audio to (default 20)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument(
        "--only",
        choices=["docs", "bn", "en"],
        action="append",
        help="restrict to one or more sources (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "testdata",
        help="output root (default <project>/testdata)",
    )
    args = parser.parse_args(argv)

    wanted = set(args.only or ["docs", "bn", "en"])
    audio_dir = args.out / "audio"
    docs_dir = args.out / "documents"
    report = Report()
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")

    print(f"output root : {args.out}")
    print(f"count       : {args.count} per source")
    print(f"clip length : {args.clip_seconds:g}s (Bengali long-form only)")
    if os.environ.get("HF_TOKEN"):
        print("auth        : HF_TOKEN present")

    audio_samples: list[dict[str, Any]] = []

    if "docs" in wanted:
        docs_dir.mkdir(parents=True, exist_ok=True)
        doc_truth = collect_documents(docs_dir, args.count, force=args.force, report=report)
        doc_truth["generated_at_utc"] = generated_at
        doc_truth["generator"] = "scripts/download_testdata.py"
        (docs_dir / "doc_ground_truth.json").write_text(
            json.dumps(doc_truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"  wrote {docs_dir / 'doc_ground_truth.json'}")

    if "bn" in wanted:
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_samples += collect_bengali(
            audio_dir, args.count, args.clip_seconds, force=args.force, report=report
        )

    if "en" in wanted:
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_samples += collect_librispeech(audio_dir, args.count, force=args.force, report=report)

    if audio_samples:
        manifest_path = audio_dir / "audio_ground_truth.json"
        existing: dict[str, Any] = {}
        if manifest_path.is_file() and args.only:
            # A partial run must not delete the other language's entries.
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged = {
            sample["file"]: sample for sample in existing.get("samples", []) + audio_samples
        }
        manifest_path.write_text(
            json.dumps(
                {
                    "generated_at_utc": generated_at,
                    "generator": "scripts/download_testdata.py",
                    "clip_seconds": args.clip_seconds,
                    "warnings": [
                        "THESE FIXTURES REQUIRE USE_MOCK_ADAPTERS=false TO MEAN "
                        "ANYTHING. Under the default mock adapters every upload is "
                        "resolved to a canned fixture, so the response describes the "
                        "fixture and not this file -- you will see duration 8.64s for "
                        "every bn_* clip and 11.20s for every en_* clip regardless of "
                        "their real length. Worse, the filenames collide with the mock "
                        "manifest's keyword index ('bn_sample_01' contains the token "
                        "'bn', matching the bn_prescription fixture), so the fallback "
                        "is not even logged as a warning. Set USE_MOCK_ADAPTERS=false "
                        "and configure a real provider before drawing any conclusion.",
                        "Bengali entries have transcript=null and "
                        "transcript_alignment='none'. Their source recordings are 4 "
                        "minutes to 5.8 hours long with a single whole-recording "
                        "transcription, so a trimmed clip has no aligned reference. "
                        "full_recording_transcript is provided for context only -- "
                        "computing WER against it would be meaningless.",
                        "LibriSpeech entries are single utterances with exact "
                        "transcripts, upper-case and unpunctuated.",
                    ],
                    "samples": sorted(merged.values(), key=lambda s: str(s["file"])),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n  wrote {manifest_path}")

    print("\n" + "=" * 62)
    print(f"written: {len(report.written)}   skipped: {len(report.skipped)}   "
          f"failed: {len(report.failed)}")
    for name, error in report.failed:
        print(f"  FAILED {name}: {error}")
    print("=" * 62)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
