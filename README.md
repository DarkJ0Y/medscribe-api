# medscribe

FastAPI service for Bengali/English speech transcription and structured medical
lab-report extraction, built on a strict three-layer architecture.

Two design commitments run through the whole codebase:

1. **Nothing is guessed.** An OCR garble like `2S.4` comes back as
   `{"raw": "2S.4", "parsed": false}` — never silently repaired to `25.4`. Every
   result carries `raw_line`, the exact OCR text it came from, so any parsed value
   can be audited against the paper report.
2. **Silence is an answer, not an error.** Ambient noise returns `200` with an
   empty transcript and `speech_detected: false`, because a model asked to
   transcribe room noise will happily invent a fluent sentence.

---

## Quick start

```bash
docker compose up
```

That is the whole thing. No `.env`, no API key, no model downloads, no outbound
network access at runtime. Then:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/api/v1/transcribe \
  -F "file=@testdata/transcription/audio/bn_prescription.wav;type=audio/wav" \
  -F "language=bn"
curl -s -X POST localhost:8000/api/v1/documents/extract \
  -F "file=@testdata/ocr/images/cbc_report.png;type=image/png"
```

Interactive docs at <http://localhost:8000/docs>.

### Locally, without Docker

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell; bash: source .venv/bin/activate
pip install -e ".[dev,real]"         # see note below
pytest                               # 256 tests, ~2.5s
uvicorn main:app --reload
```

`[real]` is included **for the test suite, not for running the service**. The
provider-adapter tests import `adapters/ocr/tesseract.py` and
`adapters/transcription/whisper_api.py`, which import their SDKs at module scope.
With only `[dev]` installed, `tests/unit/test_real_adapters.py` skips and the other
255 tests still pass — but you lose the coverage of the OCR whitespace
reconstruction (D18), which is the one thing mocks cannot check. No API key and no
`tesseract` binary are needed either way; those tests exercise pure logic.

---

## Verified behaviour

Measured on the built container, not asserted:

| Claim | Result |
| --- | --- |
| Image size | **279 MB** |
| Healthy after `docker compose up` | **~6 s** |
| Provider SDKs in the default image | `openai`, `pytesseract`, `PIL`, `numpy`, `torch` — **all absent** |
| `tesseract` binary in the default image | **absent** |
| Runs as | `uid=1001(app)`, non-root |
| Root filesystem | **read-only** (`/app` writes denied; `/tmp` is tmpfs) |
| Works with **no network** | outbound unreachable → both endpoints still return correct results |
| 4.6 MB upload under a read-only rootfs | **200** (spools to tmpfs past Starlette's 1 MB threshold) |
| Test suite | **256 passed**, ruff clean |

---

## Endpoints

### `POST /api/v1/transcribe`

Multipart: `file` (wav, mp3, m4a, mp4, ogg, oga, flac, webm — up to 25 MiB) and
`language` (`bn` \| `en` \| `auto`, default `auto`).

```json
{
  "transcript": "রোগীর তিন দিন ধরে জ্বর এবং মাথাব্যথা রয়েছে। ...",
  "detected_language": "bn",
  "duration_seconds": 8.64,
  "provider": "mock-whisper",
  "speech_detected": true,
  "warnings": []
}
```

Silence or ambient noise → still `200`:

```json
{
  "transcript": "",
  "detected_language": "unknown",
  "duration_seconds": 6.5,
  "provider": "mock-whisper",
  "speech_detected": false,
  "warnings": [
    "2 segment(s) were discarded as non-speech (no_speech_probability above 0.6).",
    "No speech detected: provider reported no_speech_probability 0.883 above the 0.6 threshold; ..."
  ]
}
```

**Check `speech_detected`, not the emptiness of `transcript`.** Three independent
signals decide it — see [Silence detection](#silence-detection).

### `POST /api/v1/documents/extract`

Multipart: `file` (JPEG or PNG photograph of a lab report, up to 25 MiB).

```json
{
  "meta": {
    "patient_name": "MD. RAFIQUL ISLAM",
    "age": "45 Y",
    "sex": "Male",
    "report_date": "12/03/2024",
    "lab_name": "POPULAR DIAGNOSTIC CENTRE LTD.",
    "reference_no": "PDC-2024-0098871"
  },
  "results": [
    {
      "test_name": "Haemoglobin",
      "value": { "raw": "11.2", "kind": "single", "parsed": true, "value": 11.2 },
      "unit": "g/dL",
      "reference_range": { "raw": "13.0 - 17.0", "kind": "range", "parsed": true,
                           "low": 13.0, "high": 17.0 },
      "flag": "low",
      "raw_line": "Haemoglobin          11.2        g/dl          13.0 - 17.0"
    }
  ],
  "provider": "mock-tesseract",
  "ocr_mean_confidence": 88.7,
  "warnings": []
}
```

`value` and `reference_range` are **objects, not scalars**. Flattening to a float
would erase the difference between `0.5` and `<0.5`:

| Printed | `kind` | Structured as |
| --- | --- | --- |
| `11.2` | `single` | `value: 11.2` |
| `<0.5` | `bounded` | `comparator: "<"`, `value: 0.5` |
| `12,500` | `single` | `value: 12500` |
| `1.2 x 10^3` | `single` | `value: 1200` |
| `0.8 - 1.2` | `range` | `low: 0.8`, `high: 1.2` |
| `1.5 - 4.5 x 10^5` | `range` | `low: 150000`, `high: 450000` — the exponent distributes |
| `2S.4`, `Not Done`, `1,25` | `unparsed` | nothing; `raw` preserved |

`age` and `report_date` stay **strings**, verbatim. `"3 M"` is a three-month-old
infant — coercing it to `3` would age the patient by three years. `05-04-2024` is
ambiguous between day-first and month-first.

**Two distinct refusals**, both `422`, because the fix differs:

| `error.code` | Meaning | What to do |
| --- | --- | --- |
| `unreadable_image` | No text recognized at all | Retake the photograph |
| `not_a_lab_report` | Text read, but too few clinical rows | Photograph a different document |

### Errors

Every failure — domain error, validation, 404, or an unexpected crash — uses one
envelope:

```json
{
  "error": {
    "code": "not_a_lab_report",
    "message": "The uploaded image contains text but does not appear to be ...",
    "details": { "lines_detected": 13, "result_rows_detected": 0, "rows_required": 2 }
  },
  "request_id": "7b0538557cd94d0caf1872c88e01aeb6"
}
```

Branch on `error.code`, not the message or the status. `request_id` is echoed in
`X-Request-ID` and attached to every log line — including on a 500.

| Status | Codes |
| --- | --- |
| 400 | `empty_upload`, `unsupported_media_type`, `corrupt_upload` |
| 413 | `file_too_large` |
| 422 | `not_a_lab_report`, `unreadable_image`, `request_validation_failed` |
| 500 | `internal_error`, `configuration_error` |
| 502 / 504 | `provider_unavailable` / `provider_timeout` |

---

## Architecture

```
        HTTP                    domain                   outside world
   ┌──────────┐            ┌──────────────┐          ┌──────────────────┐
   │   api/   │ ─────────► │  services/   │ ◄──────── │    adapters/     │
   └──────────┘            └──────────────┘          └──────────────────┘
   routing, Pydantic       business logic,           provider SDKs,
   schemas, validation,    normalizers, ports        mock replay adapters
   error handlers          (Protocols)
```

Both arrows point **at** `services/`. The domain declares the interfaces it needs
in [services/ports.py](services/ports.py) and adapters conform to them, so the core
depends on nothing — no framework, no SDK, not even Pydantic.

**Hard rules**

- `services/` never imports `fastapi`/`starlette` or any provider SDK.
- Provider SDKs and model clients exist only under `adapters/`.
- `api/` holds no business logic.

Enforced twice: statically by ruff `TID251`, and at runtime by
[tests/unit/test_layer_boundaries.py](tests/unit/test_layer_boundaries.py), which
parses the AST of every module and additionally checks in a **subprocess** what the
domain actually loads — an in-process check is worthless when the SDKs are installed
for development.

```
main.py                     composition root
api/    deps, errors, middleware, uploads, schemas/, v1/
services/  ports, domain, errors, normalizer, units, report_parser,
           upload_policy, transcription_service, document_service
adapters/  registry, replay, transcription/{mock,whisper_api}, ocr/{mock,tesseract}
config/    settings, logging
testdata/  11 fixtures + generate_media.py   (727 KB)
tests/     unit/ (10 modules), integration/ (3 modules)
```

---

## Silence detection

Three independent signals, because each alone misses a case the fixture corpus
demonstrates:

| Signal | Catches | Misses |
| --- | --- | --- |
| RMS amplitude below `SILENCE_RMS_THRESHOLD` | `silence.wav` (RMS 0.000) | `ambient_noise.wav` — RMS 0.020 is **above** the floor |
| Provider `no_speech_probability` above threshold | `ambient_noise.wav` (0.883) | providers that do not report one |
| Empty transcript after dropping noisy segments | both | a confident hallucination |

`ambient_noise.wav` is the one that matters. It records the real Whisper failure
mode — *"Thank you. Thanks for watching!"* invented over room noise — and its
amplitude is **deliberately above** the silence floor, so only the provider score
can catch it. A service checking only for an empty transcript would return that
hallucination as a clinical dictation.

Amplitude is measured for 16-bit PCM WAV only. `audioop` was removed in Python
3.13, and adding an MP3/WebM decoder would mean a native dependency in the base
image. For compressed audio that signal abstains rather than guessing.

---

## Configuration

All settings load from the environment or `.env` via `pydantic-settings`. See
[.env.example](.env.example) for the annotated list; every value has a working
default, so no `.env` is required.

| Variable | Default | Notes |
| --- | --- | --- |
| `USE_MOCK_ADAPTERS` | `true` | The master switch |
| `MAX_AUDIO_BYTES` / `MAX_IMAGE_BYTES` | `26214400` | 25 MiB |
| `SILENCE_RMS_THRESHOLD` | `0.005` | Amplitude floor |
| `NO_SPEECH_PROBABILITY_THRESHOLD` | `0.6` | Provider score ceiling |
| `MIN_LAB_ROWS_FOR_REPORT` | `2` | Below this → `not_a_lab_report` |
| `OPENAI_API_KEY`, `WHISPER_MODEL` | — / `whisper-1` | Real mode only |
| `TESSERACT_CMD`, `TESSERACT_LANGS` | — / `eng+ben` | Real mode only |

### Switching to real providers

```bash
docker compose build --build-arg INSTALL_REAL_ADAPTERS=true
USE_MOCK_ADAPTERS=false OPENAI_API_KEY=sk-... docker compose up
```

Locally: `pip install -e ".[real]"`, install `tesseract-ocr`, then set the same two
variables. Misconfiguration **fails at startup**, listing every missing
prerequisite at once — the API key, each absent package, and the `tesseract`
binary — rather than surfacing as an ImportError on the first live request.

Keep `WHISPER_MODEL=whisper-1`. It is the only model supporting `verbose_json`,
which carries the per-segment `no_speech_prob` that the anti-hallucination check
depends on. Configure another and the adapter logs a warning at startup saying so.

---

## Testing

```bash
pytest                      # 256 tests  (255 + 1 skip without the [real] extra)
pytest -m unit              # domain + adapters
pytest -m integration       # full ASGI stack
ruff check api services adapters config main.py tests testdata scripts
```

Every push runs the same commands on GitHub Actions across Python 3.11 and 3.12,
then builds the image, starts the container and smoke-tests the live endpoints —
see [.github/workflows/ci.yml](.github/workflows/ci.yml).

76 test functions across 11 modules, expanding to 256 parametrized cases. The
suite is **mutation-checked**: invariants are verified by breaking the
implementation and confirming the tests fail. That found a real hole — adding
`.strip()` to `raw_line` passed all 255 tests, because no corpus fixture has
surrounding whitespace. See DECISIONS.md D21.

The fixture corpus is generated, not committed as opaque binaries:

```bash
python testdata/generate_media.py     # stdlib only; refreshes manifest sha256s
```

Replay is content-addressed by sha256 of the uploaded bytes, so a renamed upload
cannot silently replay the wrong fixture. Digest freshness is asserted by the
suite rather than trusted.

---

## Known limitations

Stated plainly, because the gaps matter more than the features:

- **The real adapters have never run against live providers.** No call has been
  made to the OpenAI API and no real `tesseract` binary has executed. Their logic,
  error translation and response mapping are tested against the actual SDKs, but
  real response shapes, Bengali script accuracy, and behaviour on skewed phone
  photographs are unverified. This is the single largest gap.
- **OCR whitespace is reconstructed, not read.** Tesseract returns word boxes with
  no spacing, so column gaps are rebuilt from pixel geometry (DECISIONS.md D18).
  An unusual layout can misplace a column boundary.
- **`MIN_LAB_ROWS_FOR_REPORT` will reject genuine sparse reports** — a valid
  single-analyte result, or one whose units are all unrecognized. The unit alias
  table in [services/units.py](services/units.py) is therefore load-bearing for
  recall, not just tidiness.
- **No authentication, authorization or rate limiting.** This service assumes it
  sits behind something that provides them.
- **The `Content-Length` size guard is a convenience, not a defence.** A chunked
  request without that header passes it, and Starlette's multipart parser consumes
  the whole body before a handler runs. Set `client_max_body_size` in your ingress.
- **No JPEG fixture.** Generated images are PNG; JPEG acceptance is tested at the
  media-type policy layer instead (DECISIONS.md D11).
- **Dates and ages are not parsed** — deliberately, as they are ambiguous — so
  callers needing structured dates must parse them with local context.
- **Single uvicorn worker, no concurrency tuning.** Fine for the mock adapters;
  real OCR is CPU-bound and would need worker sizing and a queue.

---

## Design rationale

Twenty-one recorded decisions, each with its cost, in
[DECISIONS.md](DECISIONS.md). The ones worth reading first:

- **D1** — why the ports live in `services/`, not `adapters/`
- **D3** — Tesseract over a VLM, because `raw_line` must be verbatim
- **D5 / D14** — never guess a value; `UNKNOWN` over a false `normal`
- **D12** — three silence signals, and why amplitude abstains for compressed audio
- **D18** — the OCR whitespace bug that mocks structurally could not catch
- **D21** — mutation testing, and the hole it found in the suite
