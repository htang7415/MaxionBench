#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

INPUT_ROOT="${MAXIONBENCH_OUTPUT_ROOT:-${ROOT_DIR}/artifacts/runs/slurm}"
FIGURES_ROOT="${MAXIONBENCH_FIGURES_ROOT:-${ROOT_DIR}/artifacts/figures}"

mkdir -p "$(mb_resolve_host_path "${FIGURES_ROOT}")"
mb_log "validating benchmark outputs under ${INPUT_ROOT}"
mb_python -m maxionbench.cli validate \
  --input "${INPUT_ROOT}" \
  --enforce-protocol \
  --json

mb_log "generating milestone figures under ${FIGURES_ROOT}/milestones/Mx"
mb_python -m maxionbench.cli report \
  --input "${INPUT_ROOT}" \
  --mode milestones \
  --out "${FIGURES_ROOT}/milestones/Mx"

mb_log "generating final figures under ${FIGURES_ROOT}/final"
mb_python -m maxionbench.cli report \
  --input "${INPUT_ROOT}" \
  --mode final \
  --out "${FIGURES_ROOT}/final"
