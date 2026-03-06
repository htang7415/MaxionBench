#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

SCENARIOS=(
  "configs/scenarios/s1_ann_frontier.yaml"
  "configs/scenarios/s1_ann_frontier_d3.yaml"
  "configs/scenarios/s2_filtered_ann.yaml"
  "configs/scenarios/s3_churn_smooth.yaml"
  "configs/scenarios/s3b_churn_bursty.yaml"
  "configs/scenarios/s4_hybrid.yaml"
  "configs/scenarios/s6_fusion.yaml"
)
SCENARIO_CONFIG_DIR="${MAXIONBENCH_SCENARIO_CONFIG_DIR:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#SCENARIOS[@]}" ]]; then
  mb_die "SLURM_ARRAY_TASK_ID=${TASK_ID} is out of range for ${#SCENARIOS[@]} scenarios"
fi

DEFAULT_CONFIG_PATH="${SCENARIOS[${TASK_ID}]}"
SKIP_S6_RAW="${MAXIONBENCH_SKIP_S6:-0}"
SKIP_S6=0
case "$(echo "${SKIP_S6_RAW}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    SKIP_S6=1
    ;;
esac
if [[ "${SKIP_S6}" -eq 1 && "$(basename "${DEFAULT_CONFIG_PATH}")" == "s6_fusion.yaml" ]]; then
  echo "MAXIONBENCH_SKIP_S6 is enabled; skipping S6 task index ${TASK_ID}."
  exit 0
fi

CANDIDATE_CONFIG_PATH=""
CONFIG_PATH="${DEFAULT_CONFIG_PATH}"
if [[ -n "${SCENARIO_CONFIG_DIR}" ]]; then
  CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/$(basename "${DEFAULT_CONFIG_PATH}")"
  if [[ -f "$(mb_resolve_config "${CANDIDATE_CONFIG_PATH}")" ]]; then
    CONFIG_PATH="${CANDIDATE_CONFIG_PATH}"
  fi
fi
if [[ ! -f "$(mb_resolve_config "${CONFIG_PATH}")" ]]; then
  mb_die "scenario config does not exist: ${CONFIG_PATH}"
fi
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
