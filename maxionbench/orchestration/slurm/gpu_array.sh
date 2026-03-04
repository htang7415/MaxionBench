#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

SCENARIOS=(
  "configs/scenarios/s1_ann_frontier.yaml"
  "configs/scenarios/s5_rerank.yaml"
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

mb_run_benchmark "${STAGED_CONFIG}" --seed "${MAXIONBENCH_SEED:-42}"
mb_copy_back_output
