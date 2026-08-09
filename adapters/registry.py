"""Adapter factory: ``Settings`` -> concrete port implementations.

``settings.use_mock_adapters`` selects between the ``Mock*`` and ``Real*``
adapters. The real adapter modules are imported **inside** the factory functions,
never at module scope, so a mock-only deployment never needs the provider SDKs
installed at all -- that is what keeps the default image free of model weights
and lets ``docker compose up`` work with no API keys. See DECISIONS.md D4.

The ImportError guards are not redundant with the startup check in
``config.settings``: that check runs against the settings object, while this one
also covers a partially-installed environment (SDK present but broken) and keeps
the failure attributable to a specific adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.errors import ConfigurationError

if TYPE_CHECKING:
    from config.settings import Settings
    from services.ports import OCRPort, TranscriptionPort

_INSTALL_HINT = (
    "Install the optional provider dependencies with: pip install -e '.[real]' "
    "-- or set USE_MOCK_ADAPTERS=true to run against the mock adapters."
)


def build_transcription_adapter(settings: Settings) -> TranscriptionPort:
    """Construct the configured :class:`~services.ports.TranscriptionPort`."""
    if settings.use_mock_adapters:
        from adapters.transcription.mock import MockTranscriptionAdapter

        return MockTranscriptionAdapter(
            fixtures_dir=settings.testdata_path / "transcription",
        )

    try:
        from adapters.transcription.whisper_api import RealTranscriptionAdapter
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ConfigurationError(
            f"RealTranscriptionAdapter is unavailable: {exc}. {_INSTALL_HINT}"
        ) from exc

    # Guarded by Settings._check_real_adapter_prerequisites, but asserted here so
    # a hand-built Settings in a test cannot slip a None into the SDK.
    if settings.openai_api_key is None:
        raise ConfigurationError(
            "OPENAI_API_KEY is required when USE_MOCK_ADAPTERS=false."
        )

    return RealTranscriptionAdapter(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.whisper_model,
        timeout_seconds=settings.transcription_timeout_seconds,
    )


def build_ocr_adapter(settings: Settings) -> OCRPort:
    """Construct the configured :class:`~services.ports.OCRPort`."""
    if settings.use_mock_adapters:
        from adapters.ocr.mock import MockOCRAdapter

        return MockOCRAdapter(fixtures_dir=settings.testdata_path / "ocr")

    try:
        from adapters.ocr.tesseract import RealOCRAdapter
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ConfigurationError(
            f"RealOCRAdapter is unavailable: {exc}. {_INSTALL_HINT}"
        ) from exc

    return RealOCRAdapter(
        languages=settings.tesseract_langs,
        timeout_seconds=settings.ocr_timeout_seconds,
        tesseract_cmd=settings.tesseract_cmd,
    )
