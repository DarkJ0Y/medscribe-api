"""POST /api/v1/documents/extract -- multipart image upload -> lab report.

HTTP responsibilities only: read the ``UploadFile`` under a size ceiling, delegate
to :class:`~services.document_service.DocumentExtractionService`, and shape the
response model. Every rule about what counts as a lab report lives in
``services/``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from api.deps import DocumentServiceDep, SettingsDep
from api.errors import openapi_errors
from api.schemas.documents import ExtractResponse
from api.uploads import read_upload

router = APIRouter(tags=["documents"])


@router.post(
    "/documents/extract",
    response_model=ExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract structured results from a lab-report photograph",
    responses=openapi_errors(400, 413, 422, 502, 504),
    response_description=(
        "Report metadata plus one item per test row, each carrying the verbatim "
        "OCR line it was derived from."
    ),
)
async def extract_document(
    service: DocumentServiceDep,
    settings: SettingsDep,
    file: Annotated[
        UploadFile,
        File(description="Photograph of a medical lab report. JPEG or PNG, up to 25 MiB."),
    ],
) -> ExtractResponse:
    """Extract a lab report from a photograph.

    Every returned item carries ``raw_line``: the exact OCR text it came from, so
    any parsed value can be audited against the original document.

    Two distinct refusals, both **422**, kept separate because the fix differs:

    * ``unreadable_image`` -- no text was recognized at all. Retake the photograph.
    * ``not_a_lab_report`` -- text was read, but too few rows carry a test name, a
      value, and either a clinical unit or a reference range. Photograph a
      different document. The ``details`` report what was seen.
    """
    payload = await read_upload(file, max_bytes=settings.max_image_bytes)
    report = await service.extract(payload)
    return ExtractResponse.from_domain(report)
