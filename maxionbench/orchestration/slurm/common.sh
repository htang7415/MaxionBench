#!/usr/bin/env bash
set -euo pipefail

# Shared Slurm helpers for MaxionBench jobs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONUNBUFFERED=1
