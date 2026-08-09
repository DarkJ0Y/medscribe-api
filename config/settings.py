"""pydantic-settings ``Settings``, loaded from environment and ``.env``.

Key toggle: ``USE_MOCK_ADAPTERS=true`` (the default, and what docker-compose
sets) selects the mock adapters, so a clean clone boots with no API keys and no
model downloads.

Field names map to upper-case environment variables one-for-one
(``use_mock_adapters`` <- ``USE_MOCK_ADAPTERS``). See ``.env.example``.

A note on the prerequisite check below: it uses ``importlib.util.find_spec`` and
``shutil.which``, which *locate* the provider SDK and the tesseract binary
without importing or executing either. That keeps this module compliant with the
rule that provider SDKs are imported only inside ``adapters/`` -- config is
allowed to know a dependency's *name*, not to load it.
"""

from __future__ import annotations

import importlib.util
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root -- the parent of this ``config/`` package. Relative paths in
#: settings resolve against it, so the service behaves identically whatever the
#: process working directory happens to be.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: Service version, surfaced in OpenAPI and on /health.
#: Keep in sync with ``[project].version`` in pyproject.toml. Not read via
#: importlib.metadata because the service runs from the source tree in the
#: container rather than as an installed distribution.
APP_VERSION: Final[str] = "0.1.0"

_VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
)

#: Packages required only when ``USE_MOCK_ADAPTERS=false``; shipped in the
#: ``real`` extra and absent from the default image on purpose.
_REAL_ADAPTER_PACKAGES: Final[tuple[tuple[str, str], ...]] = (
    ("openai", "RealTranscriptionAdapter (OpenAI Whisper API)"),
    ("pytesseract", "RealOCRAdapter (Tesseract)"),
    ("PIL", "RealOCRAdapter (Pillow, image decoding)"),
)


class Settings(BaseSettings):
    """Runtime configuration. Construct via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # ---------------------------------------------------------------- adapters
    use_mock_adapters: bool = Field(
        default=True,
        description=(
            "true -> mock adapters (offline, keyless, no model weights). "
            "false -> real adapters, which require: pip install -e '.[real]'"
        ),
    )

    # ---------------------------------------------------------------- app/http
    app_name: str = "medscribe"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    request_id_header: str = "X-Request-ID"

    # ----------------------------------------------------------- upload limits
    # 25 MiB per the spec, enforced in the API layer before the body is buffered.
    max_audio_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    max_image_bytes: int = Field(default=25 * 1024 * 1024, gt=0)

    # Both an extension and a content-type allowlist, because neither is
    # trustworthy alone: browsers send "application/octet-stream" for valid
    # audio, and curl sends whatever it is told. The service accepts an upload
    # only if it satisfies one list and contradicts neither.
    #
    # Overridable from the environment as a JSON array, e.g.
    #   ALLOWED_IMAGE_EXTENSIONS='[".jpg", ".png"]'
    allowed_audio_extensions: tuple[str, ...] = (
        ".wav",
        ".mp3",
        ".m4a",
        ".mp4",
        ".ogg",
        ".oga",
        ".flac",
        ".webm",
    )
    allowed_audio_content_types: tuple[str, ...] = (
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
        "audio/ogg",
        "audio/flac",
        "audio/x-flac",
        "audio/webm",
    )
    allowed_image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    allowed_image_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/pjpeg",
        "image/png",
    )

    # --------------------------------------------------- transcription (real)
    openai_api_key: SecretStr | None = None
    whisper_model: str = "whisper-1"
    transcription_timeout_seconds: float = Field(default=120.0, gt=0)

    # Mean-amplitude floor below which audio is treated as containing no speech.
    silence_rms_threshold: float = Field(default=0.005, ge=0.0, le=1.0)
    # Whisper's own no-speech score; above this the segment is discarded even if
    # the model emitted text for it (it hallucinates on ambient noise).
    no_speech_probability_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # ------------------------------------------------------------- ocr (real)
    tesseract_cmd: str | None = None
    tesseract_langs: str = "eng+ben"
    ocr_timeout_seconds: float = Field(default=60.0, gt=0)

    # A page yielding fewer than this many recognizable test rows is reported as
    # "not a lab report" rather than returned as garbage.
    min_lab_rows_for_report: int = Field(default=2, ge=1)

    # --------------------------------------------------------------- testdata
    testdata_dir: Path = Path("testdata")

    # ------------------------------------------------------------- validators
    @field_validator(
        "openai_api_key",
        "tesseract_cmd",
        mode="before",
    )
    @classmethod
    def _blank_is_absent(cls, value: Any) -> Any:
        """Treat ``KEY=`` in a ``.env`` file as unset.

        Without this, the commented-out placeholders in ``.env.example`` would
        arrive as empty strings and read as "configured" everywhere downstream.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> Any:
        if isinstance(value, str):
            upper = value.strip().upper()
            if upper not in _VALID_LOG_LEVELS:
                raise ValueError(
                    f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {value!r}"
                )
            return upper
        return value

    @field_validator("api_v1_prefix")
    @classmethod
    def _normalize_prefix(cls, value: str) -> str:
        prefix = value.strip().rstrip("/")
        if not prefix.startswith("/"):
            raise ValueError(f"API_V1_PREFIX must start with '/', got {value!r}")
        return prefix

    @field_validator(
        "allowed_audio_extensions",
        "allowed_image_extensions",
        mode="after",
    )
    @classmethod
    def _normalize_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Lower-case and dot-prefix, so ``FilePayload.extension`` can compare
        directly rather than every call site re-normalizing."""
        normalized = []
        for ext in value:
            cleaned = ext.strip().lower()
            if cleaned and not cleaned.startswith("."):
                cleaned = f".{cleaned}"
            if cleaned:
                normalized.append(cleaned)
        return tuple(normalized)

    @field_validator(
        "allowed_audio_content_types",
        "allowed_image_content_types",
        mode="after",
    )
    @classmethod
    def _normalize_content_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(ct.strip().lower() for ct in value if ct.strip())

    @model_validator(mode="after")
    def _check_real_adapter_prerequisites(self) -> Settings:
        """Fail at startup, not on the first live request.

        ``USE_MOCK_ADAPTERS=false`` with the ``real`` extra uninstalled would
        otherwise surface as an ImportError from deep inside an adapter, on a
        request, in production. Every missing prerequisite is collected so one
        restart reveals all of them instead of one per attempt.
        """
        if self.use_mock_adapters:
            return self

        problems: list[str] = []

        if self.openai_api_key is None:
            problems.append(
                "OPENAI_API_KEY is not set (required by RealTranscriptionAdapter)"
            )

        for module, used_by in _REAL_ADAPTER_PACKAGES:
            if importlib.util.find_spec(module) is None:
                problems.append(f"Python package {module!r} is not installed (needed by {used_by})")

        binary = self.tesseract_cmd or "tesseract"
        if shutil.which(binary) is None and not Path(binary).is_file():
            problems.append(
                f"tesseract binary not found at {binary!r} "
                "(set TESSERACT_CMD, or install tesseract-ocr)"
            )

        if problems:
            listed = "\n".join(f"  - {p}" for p in problems)
            raise ValueError(
                "USE_MOCK_ADAPTERS=false but the real adapters are not ready:\n"
                f"{listed}\n"
                "Either install the extras (pip install -e '.[real]') and supply "
                "the missing configuration, or set USE_MOCK_ADAPTERS=true."
            )
        return self

    # -------------------------------------------------------------- derived
    @property
    def testdata_path(self) -> Path:
        """``testdata_dir`` resolved against :data:`PROJECT_ROOT` when relative."""
        if self.testdata_dir.is_absolute():
            return self.testdata_dir
        return PROJECT_ROOT / self.testdata_dir

    @property
    def adapter_mode(self) -> str:
        """Human-readable mode, for the health endpoint and startup logs."""
        return "mock" if self.use_mock_adapters else "real"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that a ``.env`` read and the prerequisite probe happen once.
    Tests override by calling ``get_settings.cache_clear()`` or by injecting a
    ``Settings`` instance directly -- the services never call this function, they
    receive what they need as constructor arguments.
    """
    return Settings()
