"""POST /api/v1/transcribe -- multipart audio upload -> transcript.

HTTP responsibilities only: read the ``UploadFile`` under a size ceiling, coerce
the ``language`` field to the domain :class:`LanguageHint`, delegate to
:class:`~services.transcription_service.TranscriptionService`, and shape the
response model.

No validation logic lives here. Size, emptiness and media-type rules are the
service's (via ``services/upload_policy.py``), so they hold no matter what calls
the service, and the errors they raise are translated by ``api/errors.py``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from api.deps import SettingsDep, TranscriptionServiceDep
from api.errors import openapi_errors
from api.schemas.transcription import TranscribeResponse
from api.uploads import read_upload
from services.domain import LanguageHint

router = APIRouter(tags=["transcription"])


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe Bengali or English speech",
    responses=openapi_errors(400, 413, 422, 502, 504),
    response_description="Transcript with the detected language and audio duration.",
)
async def transcribe(
    service: TranscriptionServiceDep,
    settings: SettingsDep,
    file: Annotated[
        UploadFile,
        File(description="Audio file. wav, mp3, m4a, mp4, ogg, oga, flac or webm, up to 25 MiB."),
    ],
    language: Annotated[
        LanguageHint,
        Form(
            description=(
                "'bn' or 'en' to constrain recognition, 'auto' to let the provider "
                "detect it. A hint, not an assertion: the response reports what was "
                "actually detected."
            )
        ),
    ] = LanguageHint.AUTO,
) -> TranscribeResponse:
    """Transcribe an audio upload.

    Returns **200 with an empty transcript and ``speech_detected: false``** for
    silence or ambient noise. That is a successful outcome, not an error: the
    request was processed correctly and the recording contains no speech. The
    ``warnings`` array names the signal that fired.
    """
    payload = await read_upload(file, max_bytes=settings.max_audio_bytes)
    result = await service.transcribe(payload, language)
    return TranscribeResponse.from_domain(result)
