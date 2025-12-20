# Triage Dockerfile - Modernized
# Uses Python 3.12+, uv for fast package management, and current best practices

FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="Triage" \
      org.opencontainers.image.description="Risk modeling and prediction system" \
      org.opencontainers.image.authors="Center for Data Science and Public Policy" \
      org.opencontainers.image.source="https://github.com/dssg/triage"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        libpq-dev \
        liblapack-dev \
        postgresql-client \
        git \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv - fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
ENV USERNAME=triage \
    USERID=1000 \
    SHELL=/bin/bash

RUN useradd \
    --uid "${USERID}" \
    --create-home \
    --shell /bin/bash \
    "${USERNAME}"

WORKDIR /triage

# Copy project files
COPY --chown=triage:triage pyproject.toml README.md LICENSE ./
COPY --chown=triage:triage src/ src/

# Switch to non-root user
USER ${USERNAME}

# Install triage with uv
RUN uv venv /home/triage/.venv && \
    uv pip install --python /home/triage/.venv/bin/python -e ".[dev]" && \
    uv pip install --python /home/triage/.venv/bin/python ipython

# Add venv to PATH
ENV PATH="/home/triage/.venv/bin:$PATH"

# Customize prompt
RUN echo 'export PS1="\[\033[1;34m\][\[\033[1;35m\]triage\[\033[1;34m\]:\w]#\[\033[0m\] "' >> /home/triage/.bashrc

ENTRYPOINT ["bash"]

# Production stage - installs from PyPI
FROM base AS production

LABEL org.opencontainers.image.title="Triage Production"

# Install production version from PyPI
RUN uv pip install --python /home/triage/.venv/bin/python triage

# Override prompt
RUN echo 'export PS1="\[\033[1;34m\][\[\033[1;32m\]triage-prod\[\033[1;34m\]:\w]#\[\033[0m\] "' >> /home/triage/.bashrc

ENTRYPOINT ["triage"]
