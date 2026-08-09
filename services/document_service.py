"""Lab-report extraction orchestration.

Delegates OCR to an :class:`~services.ports.OCRPort`, runs
:func:`services.report_parser.parse_report` over the returned lines, and decides
whether the image was a lab report at all.

Three distinct outcomes, kept distinct on purpose
-------------------------------------------------
* **Unreadable** (:class:`~services.errors.UnreadableImageError`) -- OCR returned
  no lines. Blank, black, or hopelessly blurred. Actionable advice: retake it.
* **Not a lab report** (:class:`~services.errors.NotALabReportError`) -- text was
  read, but too few rows carry the evidence of a clinical result table. Actionable
  advice: photograph a different document. The error carries what *was* seen, so
  the response is debuggable rather than a bare refusal.
* **A report** -- returned, possibly with null metadata and possibly with values
  preserved verbatim as unparsed.

Collapsing the first two into one "bad image" error would tell a caller to retake
a photograph of a receipt, which is not the problem.
"""

from __future__ import annotations

import logging

from services.domain import ExtractedReport, FilePayload
from services.errors import NotALabReportError, UnreadableImageError
from services.ports import OCRPort
from services.report_parser import parse_report
from services.upload_policy import validate_upload

logger = logging.getLogger(__name__)


class DocumentExtractionService:
    """Domain logic for ``POST /api/v1/documents/extract``."""

    def __init__(
        self,
        port: OCRPort,
        *,
        max_bytes: int,
        allowed_extensions: tuple[str, ...],
        allowed_content_types: tuple[str, ...],
        min_lab_rows: int,
    ) -> None:
        self._port = port
        self._max_bytes = max_bytes
        self._allowed_extensions = allowed_extensions
        self._allowed_content_types = allowed_content_types
        self._min_lab_rows = min_lab_rows

    async def extract(self, image: FilePayload) -> ExtractedReport:
        validate_upload(
            image,
            max_bytes=self._max_bytes,
            allowed_extensions=self._allowed_extensions,
            allowed_content_types=self._allowed_content_types,
        )

        raw = await self._port.extract_lines(image)

        if not raw.lines:
            raise UnreadableImageError(
                "No text could be recognized in the uploaded image. "
                "Retake the photograph with the report flat, in focus and evenly lit."
            )

        parsed = parse_report(raw.lines)

        if parsed.strong_row_count < self._min_lab_rows:
            logger.info(
                "Rejected upload as not a lab report.",
                extra={
                    "provider": raw.provider,
                    "lines_detected": len(raw.lines),
                    "strong_rows": parsed.strong_row_count,
                    "rows_required": self._min_lab_rows,
                },
            )
            raise NotALabReportError(
                "The uploaded image contains text but does not appear to be a medical "
                f"lab report: {parsed.strong_row_count} recognizable test row(s) were "
                f"found and at least {self._min_lab_rows} are required. A recognizable "
                "row needs a test name, a value, and either a clinical unit or a "
                "reference range.",
                lines_detected=len(raw.lines),
                result_rows_detected=parsed.strong_row_count,
                rows_required=self._min_lab_rows,
            )

        warnings = list(parsed.warnings)
        if parsed.meta.patient_name is None:
            warnings.append(
                "No patient name was found in the image; metadata fields are reported "
                "as null rather than inferred."
            )

        logger.info(
            "Extracted lab report.",
            extra={
                "provider": raw.provider,
                "lines_detected": len(raw.lines),
                "results": len(parsed.results),
                "strong_rows": parsed.strong_row_count,
                "unparsed_values": sum(1 for r in parsed.results if not r.value.is_parsed),
            },
        )

        return ExtractedReport(
            meta=parsed.meta,
            results=parsed.results,
            provider=raw.provider,
            ocr_mean_confidence=raw.mean_confidence,
            warnings=tuple(warnings),
        )
