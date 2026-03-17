#!/usr/bin/env bash
set -euo pipefail
SLURM_DIR="${MAXIONBENCH_SLURM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=/dev/null
source "${SLURM_DIR}/common.sh"

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

export MAXIONBENCH_ENABLE_HF_RERANKER=1
mb_require_gpu_fail_fast
mb_require_tmpdir
mb_require_visible_gpu
mb_source_dataset_env
mb_require_dataset_env_contract "${CONFIG_PATH}"
mb_allocate_ports
mb_scratch_preflight "${CONFIG_PATH}"
mb_prepare_output_paths "${SCENARIO_NAME}"
SERVICE_STARTED=0
trap 'status=$?; trap - EXIT; mb_finalize_job "${status}" "${SERVICE_STARTED:-0}"; exit "${status}"' EXIT
STAGED_CONFIG="$(mb_stage_config_to_tmp "${MB_PREFLIGHT_CONFIG}")"
export MB_STAGE_ROOT="$(dirname "${STAGED_CONFIG}")"

if mb_engine_requires_service "${STAGED_CONFIG}"; then
  mb_start_engine_services "${STAGED_CONFIG}"
  mb_wait_engine_health "${STAGED_CONFIG}"
  SERVICE_STARTED=1
fi

mb_run_benchmark "${STAGED_CONFIG}" --seed "${MAXIONBENCH_SEED:-42}"
