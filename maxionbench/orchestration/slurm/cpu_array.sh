#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

SCENARIOS=(
  "configs/scenarios/s1_ann_frontier.yaml"
  "configs/scenarios/s2_filtered_ann.yaml"
  "configs/scenarios/s3_churn_smooth.yaml"
  "configs/scenarios/s3b_churn_bursty.yaml"
  "configs/scenarios/s4_hybrid.yaml"
  "configs/scenarios/s6_fusion.yaml"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#SCENARIOS[@]}" ]]; then
  mb_die "SLURM_ARRAY_TASK_ID=${TASK_ID} is out of range for ${#SCENARIOS[@]} scenarios"
fi

CONFIG_PATH="${SCENARIOS[${TASK_ID}]}"
SCENARIO_NAME="$(basename "${CONFIG_PATH}" .yaml)"

mb_require_tmpdir
mb_allocate_ports
mb_scratch_preflight "${CONFIG_PATH}"
mb_prepare_output_paths "${SCENARIO_NAME}"
STAGED_CONFIG="$(mb_stage_config_to_tmp "${MB_PREFLIGHT_CONFIG}")"

EXTRA_ARGS=()
if [[ "${SCENARIO_NAME}" == "s2_filtered_ann" || "${SCENARIO_NAME}" == "s3_churn_smooth" || "${SCENARIO_NAME}" == "s3b_churn_bursty" ]]; then
  D3_PARAMS_PATH="${MAXIONBENCH_D3_PARAMS_PATH:-${ROOT_DIR}/artifacts/calibration/d3_params.yaml}"
  if [[ ! -f "${D3_PARAMS_PATH}" ]]; then
    mb_die "missing required d3 params file for ${SCENARIO_NAME}: ${D3_PARAMS_PATH}"
  fi
  EXTRA_ARGS+=(--d3-params "${D3_PARAMS_PATH}")
fi

mb_run_benchmark "${STAGED_CONFIG}" --seed "${MAXIONBENCH_SEED:-42}" "${EXTRA_ARGS[@]}"
mb_copy_back_output
