#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

CONFIG_PATH="${MAXIONBENCH_CALIBRATE_CONFIG:-configs/scenarios/calibrate_d3.yaml}"
if [[ -z "${MAXIONBENCH_CALIBRATE_CONFIG:-}" ]]; then
  SCENARIO_CONFIG_DIR="${MAXIONBENCH_SCENARIO_CONFIG_DIR:-}"
  if [[ -n "${SCENARIO_CONFIG_DIR}" ]]; then
    CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml"
    if [[ -f "$(mb_resolve_config "${CANDIDATE_CONFIG_PATH}")" ]]; then
      CONFIG_PATH="${CANDIDATE_CONFIG_PATH}"
    fi
  fi
fi
if [[ ! -f "$(mb_resolve_config "${CONFIG_PATH}")" ]]; then
  mb_die "calibration config does not exist: ${CONFIG_PATH}"
fi

mb_require_tmpdir
mb_allocate_ports
mb_scratch_preflight "${CONFIG_PATH}"
mb_prepare_output_paths "calibrate_d3"

STAGED_CONFIG="$(mb_stage_config_to_tmp "${MB_PREFLIGHT_CONFIG}")"
mb_log "staged config: ${STAGED_CONFIG}"
mb_run_benchmark "${STAGED_CONFIG}" --seed "${MAXIONBENCH_SEED:-42}"

mkdir -p "${ROOT_DIR}/artifacts/calibration"
if [[ -f "${MB_OUTPUT_TMP}/d3_params.yaml" ]]; then
  cp "${MB_OUTPUT_TMP}/d3_params.yaml" "${ROOT_DIR}/artifacts/calibration/d3_params.yaml"
  mb_log "wrote calibration params to ${ROOT_DIR}/artifacts/calibration/d3_params.yaml"
fi

mb_copy_back_output
