#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

DATASET_ROOT="${MAXIONBENCH_DATASET_ROOT:-${ROOT_DIR}/dataset}"
CACHE_DIR="${MAXIONBENCH_DATASET_CACHE_DIR:-${ROOT_DIR}/.cache}"
CRAG_EXAMPLES="${MAXIONBENCH_CRAG_EXAMPLES:-500}"
DOWNLOAD_ARGS=()

if [[ "${MAXIONBENCH_SKIP_D1D2_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--skip-d1d2)
fi
if [[ "${MAXIONBENCH_SKIP_D3_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--skip-d3)
fi
if [[ "${MAXIONBENCH_SKIP_D4_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--skip-d4)
fi

mkdir -p "$(mb_resolve_host_path "${DATASET_ROOT}")" "$(mb_resolve_host_path "${CACHE_DIR}")"
mb_log "downloading datasets into ${DATASET_ROOT} with cache ${CACHE_DIR}"
mb_python -m maxionbench.cli download-datasets \
  --root "${DATASET_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --crag-examples "${CRAG_EXAMPLES}" \
  "${DOWNLOAD_ARGS[@]}" \
  --json
