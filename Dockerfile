# syntax=docker/dockerfile:1
#
# Two stages: dependencies are built into a virtualenv, then only that venv and the
# source are copied into a clean runtime image. Nothing from the build toolchain
# reaches the final layer.
#
# The default image contains NO provider SDK and NO model weights. That is what
# makes `docker compose up` work on a clean clone with no API key and no downloads
# beyond the base image and the five runtime dependencies -- see DECISIONS.md D4.
# Pass --build-arg INSTALL_REAL_ADAPTERS=true to add the openai/pytesseract/Pillow
# extra and the tesseract binary with its Bengali and English language packs.

ARG PYTHON_VERSION=3.11-slim

# ---------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG INSTALL_REAL_ADAPTERS=false
# Promoted to an env var so the resolver script below can read it -- a bare ARG is
# not visible to os.environ inside the process.
ENV INSTALL_REAL_ADAPTERS=${INSTALL_REAL_ADAPTERS}

# Only pyproject.toml is copied first, so the dependency layer is cached and a
# source edit does not trigger a full reinstall. The dependency list is read out of
# pyproject with stdlib tomllib rather than duplicated into a requirements.txt --
# one source of truth, and it cannot drift.
COPY pyproject.toml ./
RUN python - <<'PY' > /tmp/requirements.txt
import os
import pathlib
import tomllib

project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]
requirements = list(project["dependencies"])
if os.environ.get("INSTALL_REAL_ADAPTERS", "false").lower() == "true":
    requirements += project["optional-dependencies"]["real"]
print("\n".join(requirements))
PY
RUN cat /tmp/requirements.txt && pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION} AS runtime

ARG INSTALL_REAL_ADAPTERS=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    USE_MOCK_ADAPTERS=true

# The tesseract binary is a system package, not a pip install, so it belongs here
# rather than in the builder. Skipped entirely by default: it and its language data
# are ~50 MB that a mock deployment has no use for.
RUN if [ "$INSTALL_REAL_ADAPTERS" = "true" ]; then \
        apt-get update \
     && apt-get install -y --no-install-recommends \
            tesseract-ocr tesseract-ocr-eng tesseract-ocr-ben \
     && rm -rf /var/lib/apt/lists/*; \
    fi

# Unprivileged, and with no home directory or login shell it has nothing to offer
# an attacker who reaches it.
RUN groupadd --system --gid 1001 app \
 && useradd --system --uid 1001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# Copied explicitly rather than with `COPY . .` so a stray .env, .venv or pat.txt in
# the build context can never end up in the image. .dockerignore is the second line
# of defence, not the only one.
COPY --chown=app:app main.py pyproject.toml ./
COPY --chown=app:app api ./api
COPY --chown=app:app services ./services
COPY --chown=app:app adapters ./adapters
COPY --chown=app:app config ./config
# testdata is deliberately shipped: the mock adapters replay it at runtime, and
# that is precisely what lets the default image work with no network access.
COPY --chown=app:app testdata ./testdata

USER app
EXPOSE 8000

# No curl or wget in the slim image, and adding one for a healthcheck would mean
# shipping an HTTP client an attacker could use. Python is already here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-c", \
         "import sys,urllib.request;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
