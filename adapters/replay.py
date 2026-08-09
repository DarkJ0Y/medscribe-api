"""Shared fixture-replay machinery for the mock adapters.

Both mocks need the same four things -- load a manifest, resolve an upload to a
fixture, read the canned response, cache it -- so the logic lives here once
rather than being reimplemented (and independently mis-implemented) per adapter.

Resolution order, most to least specific:

1. **sha256 of the uploaded bytes.** True content-addressed replay: post the
   exact fixture file and you get the exact recorded response, whatever the
   upload was named.
2. **Filename stem**, case-insensitive (``CBC_Report.PNG`` -> ``cbc_report``).
3. **Keyword tokens** in the stem, scored by number of hits so
   ``patient_lipid_profile_scan`` beats a single-token coincidence. Ties break on
   manifest order, so the outcome is always deterministic.
4. **Per-language default**, where the manifest defines one (transcription only).
5. **``default_fixture``**, logged at WARNING with the unmatched digest so a
   developer can add the file to the manifest.

Step 5 never fails, by design: a mock that raised on unknown input would make
every ad-hoc curl against a dev server an error instead of a demo. The warning is
what keeps that convenience from hiding a missing fixture.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from services.errors import ProviderUnavailableError

logger = logging.getLogger(__name__)

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


class FixtureLibrary:
    """Manifest-backed store of canned provider responses."""

    def __init__(self, fixtures_dir: Path, *, provider: str) -> None:
        self._dir = fixtures_dir
        self._provider = provider
        self._manifest: dict[str, Any] | None = None
        self._responses: dict[str, dict[str, Any]] = {}
        self._by_sha: dict[str, str] = {}

    # -- loading ------------------------------------------------------------
    async def _load_manifest(self) -> dict[str, Any]:
        """Read and index the manifest once.

        A concurrent double-load is possible and harmless (both callers produce
        the same dict), so this stays lock-free rather than paying for an
        asyncio.Lock on every request.
        """
        if self._manifest is not None:
            return self._manifest

        path = self._dir / "manifest.json"
        raw = await asyncio.to_thread(self._read_json, path)

        fixtures = raw.get("fixtures")
        if not isinstance(fixtures, dict) or not fixtures:
            raise ProviderUnavailableError(
                f"Fixture manifest {path} defines no fixtures.",
                provider=self._provider,
            )

        self._by_sha = {
            entry["sha256"]: name
            for name, entry in fixtures.items()
            if isinstance(entry, dict) and entry.get("sha256")
        }
        self._manifest = raw
        return raw

    def _read_json(self, path: Path) -> dict[str, Any]:
        """Blocking read, always called via ``asyncio.to_thread``.

        Translates every filesystem and syntax failure into
        ProviderUnavailableError: per the port contract an adapter must not let a
        FileNotFoundError or JSONDecodeError escape into the domain.
        """
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProviderUnavailableError(
                f"Fixture file is missing: {path}. "
                "Run `python testdata/generate_media.py` if the corpus is incomplete.",
                provider=self._provider,
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderUnavailableError(
                f"Fixture file {path} is not valid JSON: {exc}",
                provider=self._provider,
            ) from exc
        except OSError as exc:
            raise ProviderUnavailableError(
                f"Fixture file {path} could not be read: {exc}",
                provider=self._provider,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderUnavailableError(
                f"Fixture file {path} must contain a JSON object, got {type(data).__name__}.",
                provider=self._provider,
            )
        return data

    # -- resolution ---------------------------------------------------------
    async def resolve(
        self,
        data: bytes,
        filename: str | None,
        *,
        language_key: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Return ``(fixture_name, response_payload)`` for an upload."""
        manifest = await self._load_manifest()
        fixtures: dict[str, Any] = manifest["fixtures"]

        digest = hashlib.sha256(data).hexdigest()
        name = self._by_sha.get(digest)
        matched_by = "sha256"

        if name is None and filename:
            stem = Path(filename).stem.strip().lower()
            if stem in fixtures:
                name, matched_by = stem, "filename"
            else:
                name = self._match_keywords(stem, fixtures)
                matched_by = "keyword"

        if name is None and language_key:
            defaults = manifest.get("default_by_language") or {}
            candidate = defaults.get(language_key)
            if candidate in fixtures:
                name, matched_by = candidate, "language-default"

        if name is None:
            name = manifest.get("default_fixture")
            matched_by = "default"
            if name not in fixtures:
                raise ProviderUnavailableError(
                    f"Fixture manifest in {self._dir} has no usable default_fixture.",
                    provider=self._provider,
                )
            logger.warning(
                "No fixture matched upload; falling back to default.",
                extra={
                    "provider": self._provider,
                    "fixture": name,
                    "upload_sha256": digest,
                    "upload_filename": filename,
                },
            )

        payload = await self._load_response(name, fixtures[name])
        logger.debug(
            "Replaying fixture.",
            extra={"provider": self._provider, "fixture": name, "matched_by": matched_by},
        )
        return name, payload

    def _match_keywords(self, stem: str, fixtures: dict[str, Any]) -> str | None:
        """Highest keyword-hit count wins; manifest order breaks ties."""
        tokens = {t for t in _TOKEN_SPLIT.split(stem) if t}
        if not tokens:
            return None
        best_name: str | None = None
        best_score = 0
        for name, entry in fixtures.items():
            keywords = entry.get("keywords") or []
            score = sum(1 for kw in keywords if kw.lower() in tokens)
            if score > best_score:
                best_name, best_score = name, score
        return best_name

    async def _load_response(self, name: str, entry: dict[str, Any]) -> dict[str, Any]:
        if name in self._responses:
            return self._responses[name]
        relative = entry.get("response")
        if not relative:
            raise ProviderUnavailableError(
                f"Fixture {name!r} declares no response file.",
                provider=self._provider,
            )
        payload = await asyncio.to_thread(self._read_json, self._dir / relative)
        self._responses[name] = payload
        return payload

    # -- helpers for adapters ----------------------------------------------
    def require(self, payload: dict[str, Any], key: str, fixture: str) -> Any:
        """Fetch a required fixture field, or fail attributably."""
        if key not in payload:
            raise ProviderUnavailableError(
                f"Fixture {fixture!r} is missing the required field {key!r}.",
                provider=self._provider,
            )
        return payload[key]
