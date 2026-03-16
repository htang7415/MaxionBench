#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

SCENARIOS=(
  "configs/scenarios/s5_rerank.yaml"
  "configs/scenarios/s1_ann_frontier_track_b_gpu.yaml"
  "configs/scenarios/s5_rerank_track_c_gpu.yaml"
)
SCENARIO_CONFIG_DIR="${MAXIONBENCH_SCENARIO_CONFIG_DIR:-}"
RUN_MANIFEST_PATH="${MAXIONBENCH_SLURM_RUN_MANIFEST:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "${RUN_MANIFEST_PATH}" ]]; then
  if [[ ! -f "$(mb_resolve_host_path "${RUN_MANIFEST_PATH}")" ]]; then
    mb_die "run manifest does not exist: ${RUN_MANIFEST_PATH}"
  fi
  CONFIG_PATH="$(
    mb_python -m maxionbench.orchestration.slurm.run_manifest resolve \
      --manifest "$(mb_resolve_host_path "${RUN_MANIFEST_PATH}")" \
      --group gpu \
      --task-id "${TASK_ID}" \
      --field config_path
  )"
else
  if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#SCENARIOS[@]}" ]]; then
    mb_die "SLURM_ARRAY_TASK_ID=${TASK_ID} is out of range for ${#SCENARIOS[@]} scenarios"
  fi

  DEFAULT_CONFIG_PATH="${SCENARIOS[${TASK_ID}]}"
  CANDIDATE_CONFIG_PATH=""
  CONFIG_PATH="${DEFAULT_CONFIG_PATH}"
  if [[ -n "${SCENARIO_CONFIG_DIR}" ]]; then
    CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/$(basename "${DEFAULT_CONFIG_PATH}")"
    if [[ -f "$(mb_resolve_config "${CANDIDATE_CONFIG_PATH}")" ]]; then
      CONFIG_PATH="${CANDIDATE_CONFIG_PATH}"
    fi
  fi
fi
if [[ ! -f "$(mb_resolve_config "${CONFIG_PATH}")" ]]; then
  mb_die "scenario config does not exist: ${CONFIG_PATH}"
fi
SCENARIO_KEY="$(mb_read_config_field "${CONFIG_PATH}" "scenario")"
if [[ -z "${SCENARIO_KEY}" ]]; then
  SCENARIO_KEY="$(basename "${CONFIG_PATH}" .yaml)"
fi
SCENARIO_NAME="${SCENARIO_KEY}"

mb_require_tmpdir
mb_source_dataset_env
mb_allocate_ports
mb_scratch_preflight "${CONFIG_PATH}"
mb_prepare_output_paths "${SCENARIO_NAME}"
STAGED_CONFIG="$(mb_stage_config_to_tmp "${MB_PREFLIGHT_CONFIG}")"

mb_run_benchmark "${STAGED_CONFIG}" --seed "${MAXIONBENCH_SEED:-42}"
mb_copy_back_output
