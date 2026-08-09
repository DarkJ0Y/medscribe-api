"""Aggregates the v1 endpoint routers under the /api/v1 prefix.

The prefix itself is applied in :func:`main.create_app` from
``settings.api_v1_prefix``, so this module stays unaware of where it is mounted.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.v1 import documents, transcribe

router = APIRouter()
router.include_router(transcribe.router)
router.include_router(documents.router)
