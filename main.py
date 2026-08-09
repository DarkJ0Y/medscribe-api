"""ASGI entrypoint: application factory and composition root.

Builds the FastAPI app, wires adapters into services, installs logging, middleware
and exception handlers, and mounts the v1 router.

    uvicorn main:app --host 0.0.0.0 --port 8000

This is the only module besides ``api/`` allowed to import the web framework, and
the only place where the three layers are joined:

    settings -> adapters.registry -> services -> api routers

Everything is constructed in :func:`create_app` rather than in the lifespan hook.
Adapter construction is lazy (a fixture library reads nothing until its first
call), so there is no I/O to defer, and doing it eagerly means a misconfiguration
raises at import time instead of on whichever request happens to arrive first.
It also keeps ``TestClient(app)`` usable without a lifespan context.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from adapters.registry import build_ocr_adapter, build_transcription_adapter
from api.deps import SettingsDep
from api.errors import install_exception_handlers
from api.middleware import MaxBodySizeMiddleware, RequestIdMiddleware
from api.schemas.common import HealthResponse
from api.v1.router import router as v1_router
from config.logging import configure_logging
from config.settings import APP_VERSION, Settings, get_settings
from services.document_service import DocumentExtractionService
from services.transcription_service import TranscriptionService

logger = logging.getLogger(__name__)

DESCRIPTION = """
Speech transcription and medical lab-report extraction, on a strict three-layer
architecture (`api/` -> `services/` <- `adapters/`).

**Two things worth knowing before you call it.**

Every result item from `/documents/extract` carries `raw_line`: the exact OCR text
it was derived from. Parsed values are structured objects that keep their `raw`
text and set `parsed: false` rather than guessing -- an OCR garble like `2S.4` is
never silently repaired into `25.4`.

`/transcribe` answers **200 with an empty transcript** for silence or ambient
noise, with `speech_detected: false` and a warning naming the signal that fired.
Check that field rather than treating an empty transcript as a failure.

Errors always arrive in the same envelope, with a stable `error.code` to branch on
and a `request_id` that matches the server logs.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Pass ``settings`` to override configuration in tests."""
    resolved = settings or get_settings()
    configure_logging(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Application startup complete.",
            extra={
                "app": resolved.app_name,
                "version": APP_VERSION,
                "adapter_mode": resolved.adapter_mode,
                "api_prefix": resolved.api_v1_prefix,
                "max_audio_bytes": resolved.max_audio_bytes,
                "max_image_bytes": resolved.max_image_bytes,
            },
        )
        yield

        # Optional protocol: a real adapter may hold an HTTP connection pool or a
        # subprocess that should be released on shutdown. The mocks have nothing to
        # close, so this is duck-typed rather than added to the port -- putting
        # aclose() in the Protocol would force every adapter to implement a no-op.
        for adapter in (app.state.transcription_adapter, app.state.ocr_adapter):
            closer = getattr(adapter, "aclose", None)
            if closer is None:
                continue
            try:
                await closer()
            except Exception:  # noqa: BLE001 - shutdown must not fail on cleanup
                logger.warning(
                    "Adapter did not close cleanly.",
                    exc_info=True,
                    extra={"adapter": type(adapter).__name__},
                )
        logger.info("Application shutdown.")

    app = FastAPI(
        title=resolved.app_name,
        version=APP_VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "transcription", "description": "Speech-to-text for Bengali and English."},
            {"name": "documents", "description": "Structured extraction from lab reports."},
            {"name": "system", "description": "Liveness and service metadata."},
        ],
    )

    # --- composition root -------------------------------------------------
    app.state.settings = resolved
    # Adapters are kept on state as well as inside the services, so the lifespan
    # can close the ones that hold resources without reaching into a service's
    # private attributes.
    app.state.transcription_adapter = build_transcription_adapter(resolved)
    app.state.ocr_adapter = build_ocr_adapter(resolved)

    app.state.transcription_service = TranscriptionService(
        app.state.transcription_adapter,
        max_bytes=resolved.max_audio_bytes,
        allowed_extensions=resolved.allowed_audio_extensions,
        allowed_content_types=resolved.allowed_audio_content_types,
        silence_rms_threshold=resolved.silence_rms_threshold,
        no_speech_probability_threshold=resolved.no_speech_probability_threshold,
    )
    app.state.document_service = DocumentExtractionService(
        app.state.ocr_adapter,
        max_bytes=resolved.max_image_bytes,
        allowed_extensions=resolved.allowed_image_extensions,
        allowed_content_types=resolved.allowed_image_content_types,
        min_lab_rows=resolved.min_lab_rows_for_report,
    )

    install_exception_handlers(app)

    # Middleware is applied outermost-last, so RequestIdMiddleware is added second
    # and therefore wraps the size guard. That ordering matters: it is what lets a
    # 413 emitted by the guard carry a request_id and appear in the logs under the
    # same correlation id as everything else.
    app.add_middleware(
        MaxBodySizeMiddleware,
        max_bytes=max(resolved.max_audio_bytes, resolved.max_image_bytes),
    )
    app.add_middleware(RequestIdMiddleware, header_name=resolved.request_id_header)

    app.include_router(v1_router, prefix=resolved.api_v1_prefix)

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Liveness probe and service metadata",
    )
    async def health(settings: SettingsDep) -> HealthResponse:
        """Used by the container healthcheck.

        Reports ``adapter_mode`` on purpose: it must be impossible to mistake a
        mock deployment for one backed by live models.
        """
        return HealthResponse(
            status="ok",
            app=settings.app_name,
            version=APP_VERSION,
            adapter_mode=settings.adapter_mode,
        )

    return app


app = create_app()
