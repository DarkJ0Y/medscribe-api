"""Turning a multipart ``UploadFile`` into a domain :class:`FilePayload`.

Shared by both routers so the two endpoints read uploads identically.

The read is chunked with a hard ceiling. That ceiling is *not* the primary size
guard -- Starlette's multipart parser has already consumed the whole body by the
time a handler runs (see :mod:`api.middleware`) -- but it still earns its place:
it bounds the memory this process allocates when materializing a spooled upload,
and it catches a request that slipped past the ``Content-Length`` check because no
such header was sent.
"""

from __future__ import annotations

from typing import Final

from fastapi import UploadFile

from services.domain import FilePayload
from services.errors import FileTooLargeError

_CHUNK_BYTES: Final[int] = 64 * 1024


async def read_upload(upload: UploadFile, *, max_bytes: int) -> FilePayload:
    """Materialize ``upload`` into a domain payload, aborting past ``max_bytes``.

    Raises:
        FileTooLargeError: the upload exceeds ``max_bytes``. ``size_bytes`` in the
            error is what had been read when the read was abandoned, so it is a
            lower bound on the true size rather than the exact figure -- the exact
            figure is reported by the Content-Length guard when one is available.
    """
    chunks: list[bytes] = []
    total = 0

    while chunk := await upload.read(_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise FileTooLargeError(size_bytes=total, max_bytes=max_bytes)
        chunks.append(chunk)

    # An empty filename arrives as "" from some clients; the domain treats absent
    # and empty identically, so normalize here rather than in three places later.
    filename = upload.filename or None

    return FilePayload(
        data=b"".join(chunks),
        filename=filename,
        content_type=upload.content_type,
    )
