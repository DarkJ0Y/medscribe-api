# Design decisions

A running log. Each entry records the decision, why it was taken, and what it
costs — including the ones that trade elegance for a guarantee the spec demands.

Read these first if you are short of time: **D1** (where the ports live),
**D5**/**D14** (never guess a clinical value), **D12** (the two silence signals),
**D18** (the OCR whitespace bug mocks could not catch), **D21** (mutation testing,
and the hole it found).

---

## D1 — Ports are declared in `services/`, not `adapters/`

**Decision.** The `TranscriptionPort` and `OCRPort` Protocols live in
[services/ports.py](services/ports.py). Adapters import them; services never
import adapters.

**Why.** The brief requires that `services/` contain no provider SDKs. If the
interface lived in `adapters/`, then `services/` would have to
`from adapters.protocols import ...` — the dependency arrow would point outward,
and importing that package would eventually pull a provider SDK into the domain
through `__init__` side effects. Declaring the port in the core inverts the
dependency: `api/ -> services/ <- adapters/`. The layer rule becomes a
structural fact rather than a naming convention.

**Cost.** Domain models used in port signatures must also live in `services/`
(hence [services/domain.py](services/domain.py)), so adapters import from the
domain. That is the correct direction, but it does mean `services/domain.py` is
shared vocabulary and changes to it ripple outward into every adapter.

---

## D2 — Flat top-level packages (`api/`, `services/`, …) rather than a `src/app/` package

**Decision.** Packages sit at the repository root, matching the layout named in
the brief, with `main.py` as the composition root.

**Why.** The requested layout was explicit. Import paths read as
`from services.normalizer import ...`, which mirrors the architecture diagram
one-to-one and keeps the layer of every import visible at a glance.

**Cost.** `config` is a generic top-level name and could collide with another
distribution in a shared environment; `pyproject.toml` therefore pins the
package list explicitly instead of auto-discovering. A `src/medscribe/` layout
would have been safer for distribution, but this service is deployed as a
container, not published to an index, so the collision risk is contained.

---

## D3 — Tesseract, not a VLM, on the OCR primary path

**Decision.** `RealOCRAdapter` wraps Tesseract (`pytesseract`) for line-level
recognition. A VLM adapter remains possible behind the same port but is not the
default.

**Why.** The spec requires every result item to carry `raw_line` holding *the
exact OCR line string*. A generative model asked to read a report will happily
normalize spelling, fix units and re-order columns — producing better-looking
output while destroying the verbatim guarantee, undetectably. A classical OCR
engine returns the characters it actually saw, so `raw_line` stays audit-grade,
and normalization then happens in code we can test line by line.

**Cost.** Lower raw accuracy on skewed phone photographs and on Bengali script
than a strong VLM would achieve. Accepted: for a medical value, a wrong number
that looks confident is worse than a value we decline to parse.

---

## D4 — No ML/provider dependency in the base dependency set

**Decision.** `openai`, `pytesseract` and `pillow` sit in an optional `real`
extra. Real adapter modules are imported lazily inside
`adapters/registry.py` rather than at package import time.

**Why.** `docker compose up` from a clean clone must work with no API keys, no
network and no heavy model downloads. That is only true if the default image
never installs a provider SDK, and if importing the adapter registry does not
transitively import one.

**Cost.** A misconfiguration (`USE_MOCK_ADAPTERS=false` without the extra
installed) surfaces as an ImportError at first use rather than at startup.
Mitigated by validating the combination in `config/settings.py` and failing
fast with an actionable message.

---

## D5 — Uncertain numeric parses are preserved, never guessed

**Decision.** `services/normalizer.py` returns a structured value only when the
input matches a known form (`<0.5`, `12,500`, `1.2 x 10^3`, `0.8 - 1.2`, …).
Anything else is passed through verbatim with an explicit "not parsed" marker,
and `raw_line` is retained regardless.

**Why.** These are clinical values. A silently mis-scaled result is a patient
safety issue, whereas an unparsed one is a visible gap a caller can handle.

**Cost.** Downstream consumers must handle a nullable structured value, and
recall is lower than an aggressive parser would report.

---

## D6 — Silence is a result, not an error

**Decision.** There is no `NoSpeechDetectedError`. Silent or ambient-noise-only
audio returns HTTP 200 with an empty `transcript`, `speech_detected=false`, a
warning string, and `detected_language: "unknown"`.

**Why.** The brief asks for silence to be handled *gracefully*. A 4xx would be
wrong — the client did nothing incorrect and the request was processed
successfully; the recording simply contains no speech. Modelling it as an error
would also force every caller to implement exception handling for a routine
outcome.

Two independent signals drive the decision, because either alone misfires:
`SILENCE_RMS_THRESHOLD` catches genuinely quiet audio, and
`NO_SPEECH_PROBABILITY_THRESHOLD` catches the more dangerous case — Whisper
emitting a confident, fluent, entirely invented sentence over ambient noise.

**Cost.** Callers must check `speech_detected` rather than relying on a status
code, so an empty transcript can be mistaken for a successful transcription of
nothing if they ignore the field.

---

## D7 — Plain dataclasses in the domain, Pydantic only at the edge

**Decision.** `services/domain.py` uses frozen `@dataclass(slots=True)`.
Pydantic models appear in `api/schemas/` and `config/settings.py` only.

**Why.** Domain objects are constructed by code we control, from data already
validated at the HTTP boundary; a second validation pass would cost latency and
buy nothing. Keeping the core on stdlib types also means the layer rule survives
a future migration off Pydantic, and `frozen=True` makes the `raw_line` and
`raw` verbatim invariants structurally enforced rather than merely documented.

**Cost.** Two representations of the same concepts, so `api/schemas/` must map
domain objects to response models by hand. That mapping is the price of the
boundary — and it is also the seam that lets the wire format change without
touching business logic.

---

## D8 — Both an extension and a content-type allowlist

**Decision.** Uploads are checked against `ALLOWED_*_EXTENSIONS` *and*
`ALLOWED_*_CONTENT_TYPES`. An upload is accepted if it satisfies one and
contradicts neither.

**Why.** Neither signal is trustworthy alone. Browsers send
`application/octet-stream` for perfectly valid `.wav` files, `curl` sends
whatever it is told, and a filename extension is attacker-controlled. Rejecting
on content type alone produces false rejections of real audio; trusting the
extension alone accepts a renamed PDF.

**Cost.** A file with a correct extension and a genuinely wrong content type is
accepted and fails later in the adapter, surfacing as `corrupt_upload` rather
than `unsupported_media_type`. Accepted deliberately: content sniffing at the
edge would mean decoding the upload twice.

---

## D9 — Fixtures are content-addressed, and generated rather than committed opaquely

**Decision.** Mock adapters resolve an upload to a fixture by **sha256 of the
uploaded bytes** first, then filename stem, then keyword tokens, then a default.
The audio and images themselves are produced by
[testdata/generate_media.py](testdata/generate_media.py) from the standard
library alone, and their digests are written back into the manifests.

**Why.** Filename-only replay is brittle — a test that renames its upload
silently starts replaying the wrong fixture, and the test still passes. Hashing
the bytes makes replay exact: post the fixture and you get its recorded response
regardless of what the multipart part was called. Generating the media rather
than committing binaries means a reviewer can see what is in them, and they
rebuild identically on any machine (hence the hand-rolled LCG instead of
`random`, whose algorithm is an implementation detail).

The unknown-upload case falls through to `default_fixture` and logs a WARNING
with the unmatched digest. It does **not** raise: a mock that errored on unknown
input would turn every ad-hoc `curl` against a dev server into a failure instead
of a demo. The warning is what stops that convenience from hiding a missing
fixture.

**Cost.** Regenerating the media invalidates every stored digest, so
`generate_media.py` must be re-run and the manifests re-committed together. If
they drift, replay silently degrades to filename matching — which is why Step 7
asserts digest freshness rather than trusting it.

---

## D10 — The audio fixtures' acoustic properties are load-bearing

**Decision.** Each generated WAV is normalized to an exact target RMS:
`silence.wav` to 0.000, `ambient_noise.wav` to 0.020, speech to 0.150. The
default `SILENCE_RMS_THRESHOLD` is 0.005.

**Why.** This is what makes the two-signal silence design falsifiable rather than
merely asserted. `ambient_noise.wav` sits deliberately **above** the RMS floor,
so the amplitude check cannot catch it; it is caught only by
`no_speech_probability` (0.883 in the fixture) — and its recorded transcript is
the real Whisper failure mode, a fluent *"Thank you. Thanks for watching!"*
hallucinated over room noise. A service that only tested for an empty transcript
would return that as a clinical dictation. The fixture pair proves both checks
are necessary, because each one alone demonstrably misses a case.

**Cost.** 632 KB of WAV in the repository, since PCM is uncompressed and the
declared durations are honest (8 kHz mono keeps it to that). Encoding to a
compressed format would need ffmpeg, which would break the no-heavy-dependency
guarantee.

---

## D11 — Generated images are PNG only; JPEG support is tested at the policy layer

**Decision.** `generate_media.py` emits 8-bit greyscale PNGs. There is no
generated JPEG fixture.

**Why.** A baseline JPEG encoder is a substantial piece of code to write and
verify with no decoder available to check it against, and the fixture would add
nothing: `MockOCRAdapter` replays text from JSON, so image *content* is
irrelevant to it. The images exist only so HTTP-level tests can post real,
decodable bytes and so content-addressed replay has something to hash. JPEG
acceptance is a media-type *policy* question, and it is tested directly against
that policy in `services/`, which is where the logic lives.

**Cost.** No end-to-end test posts real JPEG bytes. If `RealOCRAdapter` ever
develops JPEG-specific decoding behaviour, `generate_media.py` will need a
Pillow-backed branch under the `real` extra to cover it.

---

## D12 — Three independent no-speech signals, and RMS abstains rather than guesses

**Decision.** No-speech is decided by amplitude RMS below threshold, **or** the
provider's `no_speech_probability` above threshold, **or** an empty transcript
after noisy segments are dropped. Amplitude is measured only for 16-bit PCM WAV;
for every other format that signal does not vote at all.

**Why.** Each signal alone misses a case that the fixtures demonstrate.
`silence.wav` (RMS 0.000) is caught by amplitude. `ambient_noise.wav` is **not** —
at RMS 0.020 it sits above the 0.005 floor — and is caught only by the provider
score, which is what stops its hallucinated *"Thank you. Thanks for watching!"*
from being returned as a clinical dictation. A service checking only for an empty
transcript would return it verbatim.

Restricting amplitude to PCM WAV is the honest option: `audioop` would have
covered more sample widths but **was removed in Python 3.13**, and adding an MP3
or WebM decoder means a native dependency in the base image, which breaks the
no-heavy-install guarantee (D4). Abstaining is better than measuring the wrong
thing — for compressed uploads the provider score carries the decision alone.

**Cost.** Silence detection is weaker for compressed audio than for WAV, and that
asymmetry is invisible to the caller. A future adapter that reports duration and
amplitude itself would close the gap without changing this rule.

---

## D13 — Strong/weak row classification, not a single row matcher

**Decision.** A **strong** row has a parsed value *and* either a recognized
clinical unit or a reference cell that is a range/one-sided bound. Only strong
rows count toward `MIN_LAB_ROWS_FOR_REPORT`. **Weak** rows (value unparseable)
are kept only when structurally vouched for: sandwiched between two strong rows,
or within two lines of the strong block while carrying a recognized unit or real
reference range.

**Why.** A single "label followed by number" matcher cannot tell a lab report
from a receipt, and the `non_lab_receipt` fixture is built to prove it — business
header, reference number, date, column header row, eight `label + number` rows.
What it lacks is any clinical unit and any reference *interval*: its numbers are
bare singles. Requiring the interval is the discriminator, and it yields zero
strong rows, which in turn vouches for none of its weak rows.

The two tiers also solve the opposite problem. `Not Done` (thyroid FT3) and the
OCR garble `2S.4` (lipid Vitamin D) are real rows whose values cannot be parsed;
dropping them would silently shrink the report, and accepting them
unconditionally would readmit the receipt. Position relative to the strong block
is what separates the two cases.

**Cost.** A genuine lab report with fewer than two parseable rows, or one whose
units are all unrecognized, is refused as "not a lab report". That is the
intended direction of failure, but it means the unit alias table in
`services/units.py` is load-bearing for *recall*, not just for tidiness.

---

## D14 — UNKNOWN is the answer whenever a comparison is not certain

**Decision.** `derive_flag` returns `UNKNOWN` for an unparsed value, an absent
reference, or a bound that does not settle the question — `<50` against a range
of `10 - 20` could be anywhere, so it is UNKNOWN, not NORMAL.

**Why.** `NORMAL` is the one answer a reader is least likely to double-check.
Defaulting to it when uncertain converts an unknown into a false reassurance,
which is the worst available failure for a clinical value. The one-sided cases
that *are* certain are still resolved: `<0.5` against `< 6.0` is NORMAL because
every value satisfying the bound is inside the range, and `>1000` against `< 34`
is HIGH for the same reason.

Direction is respected rather than assumed: HDL `38` against `> 40` flags **LOW**,
the opposite of how cholesterol bounds usually read. A naive "above the bound
means high" rule inverts the clinical meaning of the one lipid value where it
matters most.

**Cost.** More `UNKNOWN` flags than a confident implementation would report, so
callers must handle a fourth state rather than three.

---

## D15 — Unit aliases may only map units whose conversion factor is exactly 1

**Decision.** `services/units.py` standardizes *spelling*, never magnitude.
`/cumm → /µL` is permitted (1 mm³ **is** 1 µL) and `mIU/L → µIU/mL` likewise.
Nothing in that table will ever turn `g/L` into `g/dL`.

**Why.** A unit conversion is a value conversion, and a rescaled clinical number
carrying an authoritative-looking unit is precisely the failure this service is
built to prevent. Confining the table to factor-1 identities means a bug in it can
mislabel a unit but can never misstate a magnitude. Unrecognized units pass
through untouched and are reported as non-canonical, because dropping one loses
information the caller can still read while guessing at one fabricates it.

**Cost.** Reports using genuinely different units for the same analyte come back
un-unified, and any real conversion must be done by the caller with clinical
context this service does not have.

---

## D16 — Status codes follow the brief (400/413), with 415 and 422 noted

**Decision.** Rejected uploads return **400** (empty upload, unsupported media
type) and **413** (too large), as the brief specifies. Content that is
well-formed but unprocessable returns **422**: `not_a_lab_report`,
`unreadable_image`, and FastAPI's own field-validation failures.

**Why.** The brief named 400/413 explicitly, so that is what the service does.
Worth recording that **415 Unsupported Media Type** is the more precise code for a
rejected format, and it is a one-line change in `_STATUS_BY_TYPE` — the error body
is byte-identical either way. Clients are expected to branch on `error.code`
rather than the status precisely so that this choice stays an implementation
detail.

422 for the two extraction refusals is not a deviation: the request itself was
valid, and 400 would wrongly tell a caller their request was malformed when the
real message is "this photograph is not a lab report".

**Cost.** A strict reader of RFC 9110 would expect 415 for a bad media type. The
`error.code` field is what insulates clients from the difference.

---

## D17 — Unhandled exceptions are converted in the middleware, not only in the handler

**Decision.** `RequestIdMiddleware` catches unhandled exceptions and emits the
error envelope itself. The `@app.exception_handler(Exception)` registration
remains as a fallback for an app assembled without that middleware. The
correlation id is additionally stashed in the ASGI `scope`, and `api/errors.py`
prefers the scope over the `ContextVar`.

**Why.** This was a real defect found during verification, not a theoretical
concern. Starlette's `ServerErrorMiddleware` sits **outside** all user middleware,
which has two consequences that only show up on a 500:

1. By the time the `Exception` handler runs, `RequestIdMiddleware`'s `finally`
   block has already reset the `ContextVar`, so the response body carried
   `request_id: null` — on the one response where a caller most needs it.
2. The response it produces never passes through the middleware's `send` wrapper,
   so it could not carry the `X-Request-ID` header at all.

The scope stash fixes (1). Only catching the exception inside the middleware fixes
(2), because that is the innermost layer still able to add a response header.
Every response the service emits now carries the id in both places, verified to
match.

`BaseHTTPMiddleware` was rejected for the same family of reasons: it runs the
downstream app in a separate task, which breaks `ContextVar` propagation outright.
Both middlewares are plain ASGI.

---

## D18 — `RealOCRAdapter` reconstructs column whitespace from pixel geometry

**Decision.** The Tesseract adapter calls `image_to_data` for word boxes and then
rebuilds each line's spacing in `reconstruct_lines`, converting inter-word pixel
gaps into a proportional number of spaces using a **median** per-character width.

**Why.** This is the integration bug that mocks cannot catch, and it would have
been severe. `services/report_parser.py` segments columns on runs of two or more
spaces — that is how a lab report's columns appear as text. But `image_to_data`
returns coordinates and text with *no spacing at all*. Joining words with single
spaces turns

```
Haemoglobin          11.2        g/dl          13.0 - 17.0
```

into `Haemoglobin 11.2 g/dl 13.0 - 17.0`: one cell, unsegmentable. Every row of
every real report would fail to parse, so every real report would be rejected as
`not_a_lab_report` — **while the entire mock test suite kept passing**, because the
fixtures already contain the spacing.

It is verified as a matched pair: synthetic word boxes reconstruct to exactly the
expected line and parse into two strong rows with correct values, units and flags,
while the same boxes joined with single spaces yield **zero** strong rows.

The median rather than the mean per-character width is also load-bearing: one
mis-boxed glyph given a 400-pixel box would inflate a mean enough to collapse every
column gap back to a single space, silently reintroducing the same bug.

**Cost.** Spacing is inferred, so an unusual layout can misplace a column boundary
where `image_to_string` might have got it right. That was the trade: `image_to_string`
preserves some spacing but yields no per-word confidences and no reliable line
grouping. Gaps are capped at 40 spaces so one bad box cannot emit a pathological
line.

---

## D19 — Segment data requires `whisper-1`, and the adapter says so out loud

**Decision.** `RealTranscriptionAdapter` requests `response_format="verbose_json"`
only for `whisper-*` models. For anything else it requests plain JSON and logs a
**warning at construction**. The clip-level `no_speech_probability` is the
**minimum** across segment scores, and `None` when no segment carries one.

**Why.** Only `whisper-1` supports `verbose_json`, which is what carries
per-segment `no_speech_prob` and the clip `duration`. Configure
`gpt-4o-transcribe` and the service silently loses its provider-side silence
signal, falling back to amplitude alone — which per D12 is measurable only for PCM
WAV. That is a real degradation in the safety property that stops a hallucinated
transcript being returned, so it is announced at startup rather than discovered
from a support ticket.

The minimum is the right aggregate because the service reads this value as "the
whole clip is non-speech": the minimum crosses the threshold only when even the
*most* speech-like segment scored as non-speech. A mean would let one long silent
stretch drag a clip containing real speech over the line and blank a valid
transcript. Returning `None` rather than `0.0` when unavailable matters for the
same reason — `0.0` would read as confident speech.

**Cost.** Two response shapes to map, handled by reading fields with `getattr`
rather than against a concrete SDK response class. That also tolerates SDK versions
that return `verbose_json` segments as dicts instead of models, which is verified.

---

## D20 — `aclose()` is duck-typed, not part of the port

**Decision.** `RealTranscriptionAdapter` exposes `async aclose()` to release its
HTTP connection pool. The lifespan calls it via `getattr(adapter, "aclose", None)`.
It is **not** in the `TranscriptionPort`/`OCRPort` Protocols.

**Why.** Only one adapter holds a resource worth releasing. Putting `aclose()` in
the Protocol would force both mocks and the Tesseract adapter to implement no-op
methods purely to satisfy a type, which makes the interface describe the
implementation rather than the domain's need.

**Cost.** A resource leak in a future adapter that forgets the method is not caught
by the type checker. Cleanup failures are caught and logged rather than raised, so
a misbehaving adapter cannot prevent shutdown.

---

## D21 — The suite is mutation-checked, and that found a hole in it

**Decision.** Key invariants are verified by deliberately breaking the
implementation and confirming the suite fails, not just by confirming it passes.
Three mutations were run against the finished suite.

**Why.** A green suite proves nothing about the tests. Two of the three mutations
were caught immediately — collapsing the OCR column reconstruction to single spaces
failed 4 tests, and making the normalizer "repair" unparseable values by stripping
non-digits failed 8, including the `2S.4` case.

The third **survived**, and it was the most important invariant in the project.
Adding `.strip()` to `raw_line` passed all 255 tests, because **not one line in the
fixture corpus has leading or trailing whitespace** — stripping is a no-op on every
recorded fixture, so the "verbatim" test could not possibly detect it. The
invariant is byte-identity with `OcrLine.text`, not identity-modulo-whitespace, and
real OCR of an indented table can carry a left margin.

Fixed by `test_raw_line_preserves_surrounding_whitespace`, which is synthetic
rather than fixture-driven precisely because the corpus cannot express the case.
The mutation is now caught.

**Cost.** Mutation runs are manual, not automated — there is no `mutmut` in the
dependency set. A CI mutation stage would be the next step if this service grows;
for now the three checked invariants are the ones whose failure would be
clinically significant.

---

## D23 — A hand-written stub for pytesseract, not `ignore_missing_imports`

**Decision.** `mypy --strict` gates the whole tree in CI. `pytesseract` — the only
untyped dependency — is typed by hand in
[stubs/pytesseract/](stubs/pytesseract/) rather than silenced with
`ignore_missing_imports`. The pydantic mypy plugin is enabled; `soundfile` *is*
allowed to be `Any`.

**Why.** `ignore_missing_imports` turns a module into `Any`, which disables checking
of *every* call into it — the opposite of what a strict gate is for. Writing the
types down instead had an immediate payoff: declaring the real exception hierarchy
revealed that `pytesseract.TesseractError` derives from **`RuntimeError`**, and that
`adapters/ocr/tesseract.py` had `except RuntimeError` ordered *ahead* of
`except TesseractError` — making the latter **dead code**. Real Tesseract failures
were reporting the generic `"OCR failed"` message and silently dropping the
`languages` detail.

Both branches raise `ProviderUnavailableError`, so the existing test — which
asserted on `exc.code` — passed either way. That is why the dead branch survived. It
is now covered by `test_each_tesseract_failure_reaches_its_own_handler`, which
asserts the *message* and the `languages` detail, plus a test pinning the hierarchy
itself so a future re-parenting upstream fails loudly.

Be precise about the credit here: **mypy did not find this bug.**
`warn_unreachable` does not model exception subsumption in `except` chains, and the
stub does not make it flag the redundant clause. Writing the stub is what surfaced
it, by forcing the base classes to be stated explicitly. The guard is a test, not
the type checker.

`soundfile` is the one exception to the stub rule: it is used by a single call site
in a developer script, and a faithful stub would have to describe numpy array
returns. That is disproportionate, so it carries a scoped
`ignore_missing_imports` override with the reasoning recorded in `pyproject.toml`.

Enabling `plugins = ["pydantic.mypy"]` was needed for a real reason too: without it,
mypy synthesizes `Settings.__init__` from the model fields and rejects
`Settings(_env_file=None)`, which pydantic-settings accepts at runtime and which the
test suite relies on to ignore a developer's local `.env`.

**Cost.** The stub is deliberately partial — only the symbols the adapter touches —
so reaching for anything else in `pytesseract` fails type checking until the stub is
extended. That is the intended behaviour, but it is a small tax on future work, and
the stub must be kept in step if pytesseract changes a signature.

---

## D22 — The container ships `testdata/`, runs read-only, and needs no network

**Decision.** The runtime image contains the fixture corpus, runs as `uid 1001`
with `read_only: true` and `cap_drop: ALL`, mounts `/tmp` as a 64 MB tmpfs, and
contains **no** provider SDK and no `tesseract` binary unless built with
`--build-arg INSTALL_REAL_ADAPTERS=true`.

**Why.** Shipping `testdata/` is what makes the offline guarantee real rather than
aspirational: the mock adapters replay it from disk, so the container needs no
outbound connection at all. Verified by detaching the container from its Docker
network — outbound became unreachable while both endpoints continued to return
correct results.

`/tmp` must stay writable even though the service writes nothing itself: Starlette
spools multipart uploads over 1 MB to a temporary file. A read-only rootfs with no
tmpfs would fail on exactly the large uploads the 25 MiB limit is meant to permit —
so that path is tested with a 4.6 MB upload rather than assumed.

Dependencies are resolved by reading `[project].dependencies` out of
`pyproject.toml` with stdlib `tomllib` inside the build, rather than duplicating
them into a `requirements.txt` that could drift. One source of truth, and the
dependency layer still caches independently of source changes.

**Cost.** The **build** does need network — for the base image and five pip
dependencies. Only the runtime is network-free, and the README says so explicitly
rather than letting "no network access" be read as covering both. The image is
279 MB, most of it the `python:3.11-slim` base; a distroless or Alpine base would
cut it further at the cost of a harder debugging story and, for Alpine, musl wheel
availability.

**Cost.** Because the middleware handles the exception, `TestClient(app,
raise_server_exceptions=True)` no longer re-raises — a test asserting on a crash
must assert on the 500 response rather than expect a Python exception. The full
traceback is still logged at ERROR under the same `request_id`. The registered
`Exception` handler is now dead code on the normal path, kept deliberately so a
differently-assembled app cannot leak an unstructured 500.
