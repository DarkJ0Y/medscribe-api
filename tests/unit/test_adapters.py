"""Mock adapters: fixture replay and corpus integrity.

Needs no provider SDK, so it runs in a minimal ``pip install -e '.[dev]'``
environment. The real adapters' pure logic lives in test_real_adapters.py,
which requires the ``real`` extra and skips without it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from adapters.ocr.mock import MockOCRAdapter
from adapters.transcription.mock import MockTranscriptionAdapter
from services.domain import FilePayload, LanguageHint
from services.errors import ProviderUnavailableError
from services.ports import OCRPort, TranscriptionPort

pytestmark = pytest.mark.unit


# ------------------------------------------------------------ replay
def test_mock_adapters_satisfy_their_ports(client: Any) -> None:
    assert isinstance(client.app.state.transcription_adapter, TranscriptionPort)
    assert isinstance(client.app.state.ocr_adapter, OCRPort)


@pytest.mark.parametrize(
    "fixture", ["bn_prescription", "en_lab_query", "bn_en_code_switch", "silence", "ambient_noise"]
)
async def test_transcription_replay_is_content_addressed(
    client: Any, audio_bytes: Any, transcription_fixture: Any, fixture: str
) -> None:
    """Uploaded under a name that matches nothing, so only the sha256 can resolve it.

    Filename-only replay is brittle: a test that renames its upload would silently
    start replaying a different fixture and still pass.
    """
    adapter = client.app.state.transcription_adapter
    payload = FilePayload(data=audio_bytes(fixture), filename="anonymous_upload.wav")

    raw = await adapter.transcribe(payload, LanguageHint.AUTO)

    assert raw.text == transcription_fixture(fixture)["text"]
    # The adapter stamps its own name -- a fixture cannot claim to be a live model.
    assert raw.provider == "mock-whisper"


@pytest.mark.parametrize("kind", ["transcription", "ocr"])
def test_manifest_digests_are_current(manifest: Any, testdata_dir: Path, kind: str) -> None:
    """Guards D9's failure mode.

    Regenerating the media without refreshing the manifests would silently downgrade
    content-addressed replay to filename matching, and every other test would keep
    passing. So freshness is asserted rather than assumed.
    """
    index = manifest(kind)
    for name, entry in index["fixtures"].items():
        media = testdata_dir / kind / entry["media"]
        assert media.is_file(), f"{name}: media missing at {media}"

        digest = hashlib.sha256(media.read_bytes()).hexdigest()
        assert entry["sha256"] == digest, (
            f"{name}: manifest digest is stale. Re-run "
            "`python testdata/generate_media.py` and commit both."
        )


def test_every_fixture_has_a_distinct_digest(manifest: Any) -> None:
    """Two fixtures with identical bytes would make content-addressed replay
    ambiguous, and whichever lost the race would be silently unreachable."""
    digests = [
        entry["sha256"]
        for kind in ("transcription", "ocr")
        for entry in manifest(kind)["fixtures"].values()
    ]
    assert len(digests) == len(set(digests))


async def test_adapters_translate_io_failures_into_domain_errors(tmp_path: Path) -> None:
    """The port contract: no FileNotFoundError or JSONDecodeError may escape."""
    missing = MockTranscriptionAdapter(fixtures_dir=tmp_path / "does_not_exist")
    with pytest.raises(ProviderUnavailableError) as caught:
        await missing.transcribe(FilePayload(data=b"x", filename="a.wav"), LanguageHint.AUTO)
    assert caught.value.details["provider"] == "mock-whisper"

    (tmp_path / "manifest.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(ProviderUnavailableError) as caught:
        await MockOCRAdapter(fixtures_dir=tmp_path).extract_lines(
            FilePayload(data=b"x", filename="a.png")
        )
    assert "not valid JSON" in str(caught.value)

    (tmp_path / "manifest.json").write_text('{"fixtures": {}}', encoding="utf-8")
    with pytest.raises(ProviderUnavailableError):
        await MockOCRAdapter(fixtures_dir=tmp_path).extract_lines(
            FilePayload(data=b"x", filename="a.png")
        )


async def test_ocr_text_is_passed_through_without_stripping(
    client: Any, image_payload: Any, testdata_dir: Path
) -> None:
    """The column alignment IS the data: `raw_line` is derived from it."""
    raw = await client.app.state.ocr_adapter.extract_lines(image_payload("cbc_report"))
    expected = json.loads(
        (testdata_dir / "ocr" / "responses" / "cbc_report.json").read_text(encoding="utf-8")
    )

    for line, source in zip(raw.lines, expected["lines"], strict=True):
        assert line.text == source["text"], "OCR text was altered in transit"
