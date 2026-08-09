"""Regenerate the synthetic audio/image fixtures and refresh manifest checksums.

Run from anywhere:

    python testdata/generate_media.py

Why this script exists instead of committed mystery binaries: a reviewer can see
exactly what the fixtures contain, and they can be rebuilt byte-for-byte on any
machine. It uses the standard library only -- no numpy, no Pillow, no ffmpeg --
so it runs in the same bare environment as the mock adapters.

What is real and what is not
----------------------------
The **audio** is acoustically real and its properties are load-bearing. Each WAV
is normalized to an exact target RMS, which is what makes the silence-detection
pair meaningful:

    silence.wav        RMS 0.000  -> below SILENCE_RMS_THRESHOLD (0.005), so the
                                    amplitude check alone catches it.
    ambient_noise.wav  RMS 0.020  -> ABOVE that floor, so the amplitude check
                                    does NOT catch it. It is caught only by
                                    no_speech_probability (0.883 in the fixture).

That is the whole argument for having two independent silence signals, expressed
as two files you can measure.

The **images** are valid, decodable 8-bit greyscale PNGs that look like
documents -- one dark bar per line of the corresponding OCR fixture -- but they
contain no actual glyphs. Nothing needs them to: MockOCRAdapter replays the text
from ``responses/*.json``. They exist so HTTP-level tests can post real image
bytes, and so content-addressed replay has something to hash.

JPEG note: writing a baseline JPEG encoder from scratch is not worth it, so every
generated image is a PNG. JPEG acceptance is covered by unit tests against the
media-type policy in ``services/``, which is where that logic actually lives.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import wave
import zlib
from collections.abc import Iterator
from pathlib import Path

TESTDATA = Path(__file__).resolve().parent

SAMPLE_RATE = 8_000  # 8 kHz mono keeps the whole corpus under ~700 KB
SAMPLE_WIDTH = 2  # 16-bit signed PCM


# ---------------------------------------------------------------------------
# Deterministic pseudo-randomness
# ---------------------------------------------------------------------------
def _lcg(seed: int) -> Iterator[float]:
    """Numerical Recipes LCG -- fixed output across Python versions.

    ``random.Random`` would also be seedable, but its algorithm is an
    implementation detail; this one guarantees the fixtures hash identically on
    every interpreter, which matters because the manifest stores their sha256.
    """
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state / 0xFFFFFFFF  # in [0, 1)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def _speech_like(duration: float, seed: int) -> list[float]:
    """A voiced-sounding signal: three formant-ish tones under a syllable
    envelope, with short pauses so the waveform is not uniformly loud."""
    rng = _lcg(seed)
    n = int(duration * SAMPLE_RATE)
    out: list[float] = []
    for i in range(n):
        t = i / SAMPLE_RATE
        # ~3.5 syllables/second, never fully closing except in the pauses below
        envelope = 0.35 + 0.65 * abs(math.sin(2 * math.pi * 1.75 * t))
        # inter-phrase pauses every ~2.4 s
        if (t % 2.4) > 2.05:
            envelope *= 0.04
        sample = (
            0.55 * math.sin(2 * math.pi * 180 * t)
            + 0.30 * math.sin(2 * math.pi * 320 * t)
            + 0.15 * math.sin(2 * math.pi * 640 * t)
            + 0.05 * (next(rng) * 2 - 1)
        )
        out.append(envelope * sample)
    return out


def _noise(duration: float, seed: int) -> list[float]:
    rng = _lcg(seed)
    n = int(duration * SAMPLE_RATE)
    return [next(rng) * 2 - 1 for _ in range(n)]


def _silence(duration: float, _seed: int) -> list[float]:
    return [0.0] * int(duration * SAMPLE_RATE)


def _rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _normalize_to_rms(samples: list[float], target: float) -> list[float]:
    """Scale so the signal has exactly ``target`` RMS.

    Exactness is the point -- these values are asserted against the configured
    silence threshold, so an approximate amplitude would make the test vacuous.
    """
    if target <= 0.0:
        return [0.0] * len(samples)
    current = _rms(samples)
    if current == 0.0:
        return samples
    factor = target / current
    # Clamp rather than clip: a hard clip would distort the RMS we just set.
    peak = max(abs(s) for s in samples) * factor
    if peak > 0.98:
        factor *= 0.98 / peak
    return [s * factor for s in samples]


def write_wav(path: Path, samples: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for s in samples:
        clamped = max(-1.0, min(1.0, s))
        frames += struct.pack("<h", int(clamped * 32767))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(frames))


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def write_png(path: Path, width: int, height: int, rows: list[bytearray]) -> None:
    """Minimal 8-bit greyscale PNG writer (spec section 4, no interlacing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + bytes(row) for row in rows)  # filter type 0 per row

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def render_document(width: int, height: int, line_count: int, seed: int) -> list[bytearray]:
    """Lay out one dark bar per OCR line on a paper-coloured page."""
    rng = _lcg(seed)
    paper, ink, margin = 244, 38, 40
    rows = [bytearray([paper] * width) for _ in range(height)]

    if line_count == 0:  # unreadable page: flat, near-black, no structure
        return [bytearray([26] * width) for _ in range(height)]

    usable = height - 2 * margin
    pitch = max(6, usable // max(line_count, 1))
    bar_height = max(2, min(9, pitch - 4))

    for index in range(line_count):
        top = margin + index * pitch
        if top + bar_height >= height - margin:
            break
        # Headers are short and centred-ish; body lines run wide.
        is_header = index < 3
        length_fraction = (0.30 + 0.22 * next(rng)) if is_header else (0.62 + 0.33 * next(rng))
        bar_width = int((width - 2 * margin) * length_fraction)
        left = margin + (int((width - 2 * margin - bar_width) * next(rng)) if is_header else 0)
        shade = ink + int(next(rng) * 40)  # slight per-line contrast variation
        for y in range(top, top + bar_height):
            row = rows[y]
            for x in range(left, min(left + bar_width, width - margin)):
                row[x] = shade

    return rows


# ---------------------------------------------------------------------------
# Fixture definitions -- durations must match responses/*.json duration_seconds
# ---------------------------------------------------------------------------
AUDIO: tuple[tuple[str, float, str, float, int], ...] = (
    # (name, duration_s, kind, target_rms, seed)
    ("bn_prescription", 8.64, "speech", 0.150, 1001),
    ("en_lab_query", 11.20, "speech", 0.150, 1002),
    ("bn_en_code_switch", 9.41, "speech", 0.150, 1003),
    ("silence", 4.00, "silence", 0.000, 1004),
    ("ambient_noise", 6.50, "noise", 0.020, 1005),
    # Clinical dictation fixtures. Durations mirror duration_seconds in the
    # matching responses/*.json, so a probe of the container agrees with what
    # the fixture claims.
    ("en_clinical_cardiac", 13.40, "speech", 0.150, 1006),
    ("en_clinical_hypertension", 12.80, "speech", 0.150, 1007),
    ("en_clinical_oncology", 13.10, "speech", 0.150, 1008),
)

_GENERATORS = {"speech": _speech_like, "noise": _noise, "silence": _silence}

IMAGES: tuple[tuple[str, int, int, int], ...] = (
    # (name, width, height, seed) -- line_count is read from the OCR fixture
    ("cbc_report", 600, 800, 2001),
    ("lipid_profile", 600, 760, 2002),
    ("thyroid_panel", 600, 720, 2003),
    ("partial_crop", 600, 240, 2004),
    ("non_lab_receipt", 420, 640, 2005),
    ("blank_page", 480, 640, 2006),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_manifest(manifest_path: Path) -> int:
    """Write the sha256 of each fixture's media file back into the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated = 0
    for name, entry in manifest["fixtures"].items():
        media = manifest_path.parent / entry["media"]
        if not media.is_file():
            print(f"  ! {name}: media missing at {media}")
            continue
        digest = _sha256(media)
        if entry.get("sha256") != digest:
            entry["sha256"] = digest
            updated += 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return updated


def main() -> None:
    audio_dir = TESTDATA / "transcription" / "audio"
    print("audio:")
    for name, duration, kind, target_rms, seed in AUDIO:
        samples = _normalize_to_rms(_GENERATORS[kind](duration, seed), target_rms)
        path = audio_dir / f"{name}.wav"
        write_wav(path, samples)
        print(
            f"  {name+'.wav':26} {duration:5.2f}s  rms={_rms(samples):.4f}  "
            f"{path.stat().st_size / 1024:6.1f} KB"
        )

    image_dir = TESTDATA / "ocr" / "images"
    response_dir = TESTDATA / "ocr" / "responses"
    print("images:")
    for name, width, height, seed in IMAGES:
        fixture = json.loads((response_dir / f"{name}.json").read_text(encoding="utf-8"))
        line_count = len(fixture["lines"])
        path = image_dir / f"{name}.png"
        write_png(path, width, height, render_document(width, height, line_count, seed))
        print(
            f"  {name+'.png':26} {width}x{height}  {line_count:2d} lines  "
            f"{path.stat().st_size / 1024:6.1f} KB"
        )

    print("manifests:")
    for manifest in (
        TESTDATA / "transcription" / "manifest.json",
        TESTDATA / "ocr" / "manifest.json",
    ):
        count = _refresh_manifest(manifest)
        print(f"  {manifest.parent.name}/manifest.json: {count} checksum(s) updated")


if __name__ == "__main__":
    main()
