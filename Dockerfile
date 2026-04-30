FROM python:3.11-slim

ARG MAXIONBENCH_PIP_EXTRAS=dev,engines,reporting,datasets

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/maxionbench

COPY pyproject.toml README.md ./
COPY AGENTS.md CLAUDE.md document.md project.md prompt.md ./
COPY maxionbench ./maxionbench
COPY configs ./configs
COPY docs ./docs

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install ".[${MAXIONBENCH_PIP_EXTRAS}]"

ENTRYPOINT ["maxionbench"]
CMD ["--help"]
