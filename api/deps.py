"""FastAPI dependency providers.

Services and their adapters are constructed **once**, during application startup
(see :func:`main.lifespan`), and stored on ``app.state``. These dependencies only
hand them out.

That matters for more than tidiness: :class:`adapters.replay.FixtureLibrary`
caches the manifest and every response it has read, so a per-request adapter would
re-read the corpus from disk on every call and throw the cache away. Building at
startup also means a misconfiguration fails when the process boots rather than on
the first request to reach production.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from config.settings import Settings
from services.document_service import DocumentExtractionService
from services.transcription_service import TranscriptionService


def get_settings(request: Request) -> Settings:
    """The Settings instance resolved at startup."""
    settings: Settings = request.app.state.settings
    return settings


def get_transcription_service(request: Request) -> TranscriptionService:
    service: TranscriptionService = request.app.state.transcription_service
    return service


def get_document_service(request: Request) -> DocumentExtractionService:
    service: DocumentExtractionService = request.app.state.document_service
    return service


SettingsDep = Annotated[Settings, Depends(get_settings)]
TranscriptionServiceDep = Annotated[TranscriptionService, Depends(get_transcription_service)]
DocumentServiceDep = Annotated[DocumentExtractionService, Depends(get_document_service)]
