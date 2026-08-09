"""MockOCRAdapter -- replays saved OCR line output from testdata/.

Ships fixtures for the happy path, a cropped page, a hard non-lab-report negative
(a supermarket receipt) and an unreadable page, so every branch of the document
service is reachable with no OCR engine installed.

Two contract points this adapter honours literally:

* **Verbatim text.** ``OcrLine.text`` is passed through exactly as the fixture
  records it -- no stripping, case folding or whitespace collapsing. The awkward
  column alignment in the fixtures is load-bearing, since ``LabResult.raw_line``
  is derived from it and is meant to be auditable against the paper report.
* **Empty is data, not failure.** The ``blank_page`` fixture returns zero lines
  rather than raising. Turning that into an error is the service's decision, made
  once, instead of each OCR engine inventing its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.replay import FixtureLibrary
from services.domain import FilePayload, OcrLine, RawOcrResult

_PROVIDER = "mock-tesseract"


class MockOCRAdapter:
    """Offline :class:`~services.ports.OCRPort` backed by fixtures."""

    def __init__(self, fixtures_dir: Path) -> None:
        self._library = FixtureLibrary(fixtures_dir, provider=_PROVIDER)

    @property
    def provider_name(self) -> str:
        return _PROVIDER

    async def extract_lines(self, image: FilePayload) -> RawOcrResult:
        fixture, payload = await self._library.resolve(image.data, image.filename)
        raw_lines: list[dict[str, Any]] = self._library.require(payload, "lines", fixture)

        mean_confidence = payload.get("mean_confidence")
        return RawOcrResult(
            lines=self._parse_lines(raw_lines),
            provider=self.provider_name,
            mean_confidence=float(mean_confidence) if mean_confidence is not None else None,
        )

    def _parse_lines(self, raw: list[dict[str, Any]]) -> tuple[OcrLine, ...]:
        return tuple(
            OcrLine(
                # str() but no .strip(): the alignment IS the data.
                text=str(item.get("text", "")),
                line_number=int(item.get("line_number", index + 1)),
                confidence=(
                    float(item["confidence"]) if item.get("confidence") is not None else None
                ),
            )
            for index, item in enumerate(raw)
        )
