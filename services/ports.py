"""Outbound ports -- what the domain requires of the outside world.

Declared here in the domain rather than in ``adapters/``: the core owns the
interface it needs and concrete adapters conform to it. Without this inversion
``services/`` would have to import ``adapters/``, and the "no provider SDKs in
services" rule would be a naming convention rather than a structural fact. See
DECISIONS.md D1.

    api/  ------>  services/  <------  adapters/
                   (ports live here)

Contract every implementation must honour
-----------------------------------------
* **Async.** Implementations do network or subprocess I/O. Even the mocks are
  ``async`` so swapping one in never changes a call site.
* **Translate your exceptions.** An adapter must not let a provider-specific
  exception escape. Wrap failures in
  :class:`~services.errors.ProviderUnavailableError` or
  :class:`~services.errors.ProviderTimeoutError`; those are the only failures a
  service is written to expect.
* **Report, do not decide.** Return what the provider said and let the service
  interpret it. Adapters do not map language codes onto
  :class:`~services.domain.Language`, do not judge whether audio was silent, and
  do not decide whether an image is a lab report -- those rules belong in one
  testable place, not duplicated per provider.
* **Preserve text verbatim.** :attr:`~services.domain.OcrLine.text` is the
  characters the engine actually produced. No stripping, case folding, spelling
  correction or reflowing -- ``LabResult.raw_line`` is derived from it and is an
  audit trail.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from services.domain import FilePayload, LanguageHint, RawOcrResult, RawTranscription

# ``runtime_checkable`` enables ``isinstance`` checks in the tests, but note it
# verifies only that members *exist* -- not their signatures. mypy in strict mode
# is what actually checks conformance; the isinstance check is a cheap backstop
# against an adapter that forgets a method entirely.


@runtime_checkable
class TranscriptionPort(Protocol):
    """Speech-to-text."""

    @property
    def provider_name(self) -> str:
        """Stable identifier surfaced to clients as ``provider``.

        Distinguish mock from real (``"mock-whisper"`` vs ``"openai-whisper-1"``)
        so a response can never be mistaken for having come from a live model.
        """
        ...

    async def transcribe(
        self,
        audio: FilePayload,
        language: LanguageHint,
    ) -> RawTranscription:
        """Transcribe ``audio``, optionally constrained to a language.

        ``language`` is a hint, not an assertion: with
        :attr:`~services.domain.LanguageHint.AUTO` the provider detects the
        language itself and reports it in
        :attr:`~services.domain.RawTranscription.detected_language`.

        Populate ``duration_seconds`` when the provider or container exposes it;
        the adapter holds both the bytes and the SDK, so it is the cheapest place
        to learn the duration. Returning ``None`` is permitted and the service
        degrades to ``0.0`` with a warning rather than inventing a number.

        Raises:
            services.errors.ProviderUnavailableError: upstream failed.
            services.errors.ProviderTimeoutError: upstream exceeded its deadline.
        """
        ...


@runtime_checkable
class OCRPort(Protocol):
    """Optical character recognition, at line granularity."""

    @property
    def provider_name(self) -> str:
        """Stable identifier surfaced to clients as ``provider``."""
        ...

    async def extract_lines(self, image: FilePayload) -> RawOcrResult:
        """Recognize ``image`` and return its lines in reading order.

        Line granularity is required, not incidental: the lab-report parser works
        line by line and every emitted result must quote its source line
        verbatim. An engine that only returns a single blob of text cannot back
        this port without first being taught to segment.

        Return an empty ``lines`` tuple for an image containing no recognizable
        text; the service turns that into
        :class:`~services.errors.UnreadableImageError`. Do not raise for "no text
        found" -- that is data, not a provider failure.

        Raises:
            services.errors.ProviderUnavailableError: engine missing or crashed.
            services.errors.ProviderTimeoutError: engine exceeded its deadline.
            services.errors.CorruptUploadError: the bytes are not a decodable
                image. This is the one caller-fault error an adapter may raise,
                because an adapter is the first component able to tell: the
                media-type allowlist admits a file on its extension and content
                type (D8), neither of which proves the bytes are really an image.
        """
        ...
