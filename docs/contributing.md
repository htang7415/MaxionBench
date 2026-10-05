# Contributing

## Setup

```bash
python3 -m venv ~/.venvs/maxionbench
. ~/.venvs/maxionbench/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install -e ".[rag]"   # v0.2/v0.3 RAG and serving tools
```

Keep the virtual environment on an internal disk. On exFAT or other external volumes, macOS writes
`._*` AppleDouble files that break Python imports and pytest collection, and imports run up to 10x
slower. Remove stray metadata before testing or committing:

```bash
find . -name '._*' -not -path './.git/*' -delete
```

## Validate changes

```bash
python -m ruff check maxionbench scripts tests
python -m pytest -q
python scripts/build_package.py
```

Keep changes focused. Add tests for behavior changes. Document schema or benchmark-policy changes under `docs/migrations/`.

Generated files under `artifacts/`, `results/`, and `release/` are not source files and should not be committed.

After changing dependencies, regenerate the lock file:

```bash
python -m piptools compile \
  --extra dev \
  --extra reporting \
  --allow-unsafe \
  --generate-hashes \
  --strip-extras \
  --output-file requirements-dev.lock \
  pyproject.toml
```
