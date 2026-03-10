#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SCENARIO_CONFIG_DIR="configs/scenarios_paper"
SEED="42"
SLURM_PROFILE=""
CPU_ONLY=0
SKIP_S6=0
LAUNCH=0
SKIP_PYTEST=0
SKIP_CALIBRATION=0
PREFETCH_DATASETS=0
CONTAINER_RUNTIME=""
CONTAINER_IMAGE=""
HF_CACHE_DIR=""
CONTAINER_BINDS=()
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'EOF'
Usage: ./run_workstation.sh [options]

Runs workstation preflight checks, D3 calibration verification, and Slurm plan validation.
By default this script does NOT submit Slurm jobs.
Each invocation writes a structured run bundle under artifacts/workstation_runs/<run_id>/.

Bundle layout:
  reports/   run logs and summary files
  checks/    JSON validation artifacts
  results/   local and Slurm run outputs
  figures/   reserved figure output folders
  helpers/   helper scripts (for example render_figures.sh)

Options:
  --scenario-config-dir <dir>  Scenario config directory for paper lane (default: configs/scenarios_paper)
  --slurm-profile <name>       Local Slurm profile key from profiles_local.yaml
  --seed <int>                 Seed forwarded to submit-slurm-plan (default: 42)
  --cpu-only                   Use skip-gpu mode when submitting Slurm jobs
  --skip-s6                    Defer S6 by forwarding --skip-s6 to submit-slurm-plan
  --launch                     Submit Slurm jobs after checks pass
  --skip-pytest                Skip pytest -q
  --skip-calibration           Skip calibrate_d3 + verify-d3-calibration
  --prefetch-datasets         Prefetch required paper-lane datasets into the shared repo cache before checks
  --container-runtime <name>   Container runtime for Slurm jobs (currently: apptainer)
  --container-image <path>     Container image for Slurm jobs (for example /shared/maxionbench.sif)
  --container-bind <spec>      Extra container bind spec for Slurm jobs; repeatable host[:container[:opts]]
  --hf-cache-dir <path>        HF cache dir to bind/export inside containerized Slurm jobs
  -h, --help                   Show this help

Private Slurm profile overrides:
  - tracked docs/code do not ship named cluster presets
  - copy maxionbench/orchestration/slurm/profiles_local.example.yaml to
    maxionbench/orchestration/slurm/profiles_local.yaml and edit your local values there

Paper D3 calibration:
  - if ${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml sets `calibration_require_real_data: true`
    and does not already contain a concrete `dataset_path`, export:
      MAXIONBENCH_D3_DATASET_PATH=/abs/path/to/laion_d3_vectors.npy
  - optional checksum pin:
      MAXIONBENCH_D3_DATASET_SHA256=<64-char-lowercase-hex>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario-config-dir)
      SCENARIO_CONFIG_DIR="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --slurm-profile)
      SLURM_PROFILE="${2:-}"
      shift 2
      ;;
    --cpu-only)
      CPU_ONLY=1
      shift
      ;;
    --skip-s6)
      SKIP_S6=1
      shift
      ;;
    --launch)
      LAUNCH=1
      shift
      ;;
    --skip-pytest)
      SKIP_PYTEST=1
      shift
      ;;
    --skip-calibration)
      SKIP_CALIBRATION=1
      shift
      ;;
    --prefetch-datasets)
      PREFETCH_DATASETS=1
      shift
      ;;
    --container-runtime)
      CONTAINER_RUNTIME="${2:-}"
      shift 2
      ;;
    --container-image)
      CONTAINER_IMAGE="${2:-}"
      shift 2
      ;;
    --container-bind)
      CONTAINER_BINDS+=("${2:-}")
      shift 2
      ;;
    --hf-cache-dir)
      HF_CACHE_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v maxionbench >/dev/null 2>&1; then
  echo "error: 'maxionbench' command not found in PATH" >&2
  exit 127
fi

if [[ ! -f "${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml" ]]; then
  echo "error: missing ${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml" >&2
  exit 2
fi

CALIBRATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml"

SLURM_PROFILE_ARGS=()
if [[ -n "${SLURM_PROFILE}" ]]; then
  SLURM_PROFILE_ARGS=(--slurm-profile "${SLURM_PROFILE}")
fi
SLURM_S6_ARGS=()
if [[ "${SKIP_S6}" -eq 1 ]]; then
  SLURM_S6_ARGS=(--skip-s6)
fi
SLURM_PREFETCH_ARGS=()
if [[ "${PREFETCH_DATASETS}" -eq 1 ]]; then
  SLURM_PREFETCH_ARGS=(--prefetch-datasets)
fi
SLURM_CONTAINER_ARGS=()
if [[ -n "${CONTAINER_RUNTIME}" ]]; then
  case "${CONTAINER_RUNTIME}" in
    apptainer)
      ;;
    *)
      echo "error: --container-runtime must be: apptainer" >&2
      exit 2
      ;;
  esac
  if [[ -z "${CONTAINER_IMAGE}" ]]; then
    echo "error: --container-image is required when --container-runtime is set" >&2
    exit 2
  fi
  SLURM_CONTAINER_ARGS+=(--container-runtime "${CONTAINER_RUNTIME}" --container-image "${CONTAINER_IMAGE}")
fi
if [[ -n "${CONTAINER_IMAGE}" && -z "${CONTAINER_RUNTIME}" ]]; then
  echo "error: --container-runtime is required when --container-image is set" >&2
  exit 2
fi
if [[ -n "${HF_CACHE_DIR}" ]]; then
  SLURM_CONTAINER_ARGS+=(--hf-cache-dir "${HF_CACHE_DIR}")
fi
if [[ "${#CONTAINER_BINDS[@]}" -gt 0 ]]; then
  for bind_spec in "${CONTAINER_BINDS[@]}"; do
    if [[ -n "${bind_spec}" ]]; then
      SLURM_CONTAINER_ARGS+=(--container-bind "${bind_spec}")
    fi
  done
fi

run_submit_slurm_plan() {
  local -a cmd=("maxionbench" "submit-slurm-plan")
  if [[ "${#SLURM_PROFILE_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${SLURM_PROFILE_ARGS[@]}")
  fi
  if [[ "${#SLURM_S6_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${SLURM_S6_ARGS[@]}")
  fi
  if [[ "${#SLURM_PREFETCH_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${SLURM_PREFETCH_ARGS[@]}")
  fi
  if [[ "${#SLURM_CONTAINER_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${SLURM_CONTAINER_ARGS[@]}")
  fi
  cmd+=("$@")
  "${cmd[@]}"
}

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="workstation_${RUN_TS}"
RUN_BUNDLE_ROOT="artifacts/workstation_runs/${RUN_ID}"
RUN_REPORT_DIR="${RUN_BUNDLE_ROOT}/reports"
RUN_CHECKS_DIR="${RUN_BUNDLE_ROOT}/checks"
RUN_RESULTS_ROOT="${RUN_BUNDLE_ROOT}/results"
RUN_RESULTS_LOCAL="${RUN_RESULTS_ROOT}/local"
RUN_RESULTS_SLURM="${RUN_RESULTS_ROOT}/slurm"
RUN_FIGURES_ROOT="${RUN_BUNDLE_ROOT}/figures"
RUN_FIGURES_MILESTONES="${RUN_FIGURES_ROOT}/milestones"
RUN_FIGURES_FINAL="${RUN_FIGURES_ROOT}/final"
RUN_HELPERS_DIR="${RUN_BUNDLE_ROOT}/helpers"

mkdir -p \
  "${RUN_REPORT_DIR}" \
  "${RUN_CHECKS_DIR}" \
  "${RUN_RESULTS_LOCAL}" \
  "${RUN_RESULTS_SLURM}" \
  "${RUN_FIGURES_MILESTONES}" \
  "${RUN_FIGURES_FINAL}" \
  "${RUN_HELPERS_DIR}"

REPORT_LOG="${RUN_REPORT_DIR}/run.log"
REPORT_SUMMARY_TXT="${RUN_REPORT_DIR}/summary.txt"
REPORT_SUMMARY_MD="${RUN_REPORT_DIR}/summary.md"
RENDER_FIGURES_HELPER="${RUN_HELPERS_DIR}/render_figures.sh"
START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"
LOG_PIPE="${RUN_REPORT_DIR}/run.pipe"

SLURM_PLAN_VERIFY_JSON="${RUN_CHECKS_DIR}/slurm_plan_verify.json"
SLURM_PLAN_VERIFY_SKIP_GPU_JSON="${RUN_CHECKS_DIR}/slurm_plan_verify_skip_gpu.json"
SLURM_SUBMIT_DRY_RUN_JSON="${RUN_CHECKS_DIR}/slurm_submit_plan_dry_run.json"
SLURM_SUBMIT_SKIP_GPU_DRY_RUN_JSON="${RUN_CHECKS_DIR}/slurm_submit_plan_skip_gpu_dry_run.json"
SLURM_SUBMIT_PAPER_SKIP_GPU_DRY_RUN_JSON="${RUN_CHECKS_DIR}/slurm_submit_plan_paper_skip_gpu_dry_run.json"
SLURM_SNAPSHOT_VALIDATION_JSON="${RUN_CHECKS_DIR}/slurm_snapshot_validation.json"
CI_PROTOCOL_AUDIT_DEFAULT_JSON="${RUN_CHECKS_DIR}/ci_protocol_audit_default.json"
CI_PROTOCOL_AUDIT_PAPER_JSON="${RUN_CHECKS_DIR}/ci_protocol_audit_paper.json"

cat > "${RENDER_FIGURES_HELPER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${ROOT_DIR}"
INPUT_DIR="\${1:-${ROOT_DIR}/${RUN_RESULTS_SLURM}}"
MILESTONES_OUT="\${2:-${ROOT_DIR}/${RUN_FIGURES_MILESTONES}}"
FINAL_OUT="\${3:-${ROOT_DIR}/${RUN_FIGURES_FINAL}}"
cd "\${ROOT_DIR}"
mkdir -p "\${MILESTONES_OUT}" "\${FINAL_OUT}"
maxionbench report --input "\${INPUT_DIR}" --mode milestones --out "\${MILESTONES_OUT}"
maxionbench report --input "\${INPUT_DIR}" --mode final --out "\${FINAL_OUT}"
echo "figures generated:"
echo "- milestones: \${MILESTONES_OUT}"
echo "- final: \${FINAL_OUT}"
EOF
chmod +x "${RENDER_FIGURES_HELPER}"

rm -f "${LOG_PIPE}"
mkfifo "${LOG_PIPE}"
exec 3>&1 4>&2
tee -a "${REPORT_LOG}" < "${LOG_PIPE}" >&3 &
TEE_PID="$!"
exec > "${LOG_PIPE}" 2>&1

finalize_report() {
  local exit_code="$?"
  local end_ts
  end_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local end_epoch
  end_epoch="$(date +%s)"
  local duration_s
  duration_s="$((end_epoch - START_EPOCH))"
  local status="success"
  if [[ "${exit_code}" -ne 0 ]]; then
    status="failure"
  fi

  local args_rendered=""
  if [[ "${#ORIGINAL_ARGS[@]}" -gt 0 ]]; then
    args_rendered="$(printf '%q ' "${ORIGINAL_ARGS[@]}")"
    args_rendered="${args_rendered% }"
  else
    args_rendered="(none)"
  fi

  if [[ -f "artifacts/calibration/d3_params.yaml" ]]; then
    cp -f "artifacts/calibration/d3_params.yaml" "${RUN_RESULTS_LOCAL}/d3_params.yaml"
  fi

  cat > "${REPORT_SUMMARY_TXT}" <<EOF
run_id=${RUN_ID}
status=${status}
exit_code=${exit_code}
start_utc=${START_TS}
end_utc=${end_ts}
duration_s=${duration_s}
repo_root=${ROOT_DIR}
scenario_config_dir=${SCENARIO_CONFIG_DIR}
slurm_profile=${SLURM_PROFILE:-none}
cpu_only=${CPU_ONLY}
launch=${LAUNCH}
skip_pytest=${SKIP_PYTEST}
skip_calibration=${SKIP_CALIBRATION}
prefetch_datasets=${PREFETCH_DATASETS}
skip_s6=${SKIP_S6}
args=${args_rendered}
bundle_root=${RUN_BUNDLE_ROOT}
report_log=${REPORT_LOG}
EOF

  cat > "${REPORT_SUMMARY_MD}" <<EOF
# Workstation Run Report

- run_id: \`${RUN_ID}\`
- status: \`${status}\`
- exit_code: \`${exit_code}\`
- start_utc: \`${START_TS}\`
- end_utc: \`${end_ts}\`
- duration_s: \`${duration_s}\`
- scenario_config_dir: \`${SCENARIO_CONFIG_DIR}\`
- slurm_profile: \`${SLURM_PROFILE:-none}\`
- launch: \`${LAUNCH}\`
- cpu_only: \`${CPU_ONLY}\`
- prefetch_datasets: \`${PREFETCH_DATASETS}\`
- skip_s6: \`${SKIP_S6}\`
- args: \`${args_rendered}\`

## Bundle Paths

- bundle_root: \`${RUN_BUNDLE_ROOT}\`
- reports: \`${RUN_REPORT_DIR}\`
- checks: \`${RUN_CHECKS_DIR}\`
- results_local: \`${RUN_RESULTS_LOCAL}\`
- results_slurm: \`${RUN_RESULTS_SLURM}\`
- figures_milestones: \`${RUN_FIGURES_MILESTONES}\`
- figures_final: \`${RUN_FIGURES_FINAL}\`
- render_figures_helper: \`${RENDER_FIGURES_HELPER}\`

## Check Artifacts

- slurm_plan_verify: \`${SLURM_PLAN_VERIFY_JSON}\`
- slurm_plan_verify_skip_gpu: \`${SLURM_PLAN_VERIFY_SKIP_GPU_JSON}\`
- slurm_submit_plan_default: \`${SLURM_SUBMIT_DRY_RUN_JSON}\`
- slurm_submit_plan_skip_gpu: \`${SLURM_SUBMIT_SKIP_GPU_DRY_RUN_JSON}\`
- slurm_submit_plan_paper_skip_gpu: \`${SLURM_SUBMIT_PAPER_SKIP_GPU_DRY_RUN_JSON}\`
- slurm_snapshot_validation: \`${SLURM_SNAPSHOT_VALIDATION_JSON}\`
- ci_protocol_audit_default: \`${CI_PROTOCOL_AUDIT_DEFAULT_JSON}\`
- ci_protocol_audit_paper: \`${CI_PROTOCOL_AUDIT_PAPER_JSON}\`
EOF

  ln -sfn "${RUN_ID}" "artifacts/workstation_runs/latest"
  echo "Run report saved: ${RUN_BUNDLE_ROOT}"

  exec 1>&3 2>&4
  wait "${TEE_PID}" 2>/dev/null || true
  rm -f "${LOG_PIPE}"
  exec 3>&- 4>&-
}

trap finalize_report EXIT

echo "Run bundle root: ${RUN_BUNDLE_ROOT}"
echo "Run log: ${REPORT_LOG}"

echo "==> Step 1: code and pin sanity"
if [[ "${SKIP_PYTEST}" -eq 0 ]]; then
  pytest -q
else
  echo "Skipping pytest -q (--skip-pytest)"
fi
maxionbench verify-pins --config-dir configs/scenarios --json
maxionbench verify-pins --config-dir "${SCENARIO_CONFIG_DIR}" --strict-d3-scenario-scale --json
maxionbench verify-dataset-manifests --manifest-dir maxionbench/datasets/manifests --json
maxionbench verify-conformance-configs --config-dir configs/conformance --json

echo "==> Step 2: D3 calibration gate"
if [[ "${SKIP_CALIBRATION}" -eq 0 ]]; then
  if [[ "${PREFETCH_DATASETS}" -eq 1 ]]; then
    python -m maxionbench.orchestration.slurm.dataset_prefetch \
      --repo-root "${ROOT_DIR}" \
      --scenario-config-dir "${SCENARIO_CONFIG_DIR}" \
      --env-sh artifacts/prefetch/dataset_env.sh \
      --json
    if [[ -f "${ROOT_DIR}/artifacts/prefetch/dataset_env.sh" ]]; then
      # shellcheck disable=SC1091
      source "${ROOT_DIR}/artifacts/prefetch/dataset_env.sh"
    fi
  fi
  CALIBRATION_PATH_CHECK="$(python - <<'PY' "${CALIBRATE_CONFIG_PATH}"
import os
import pathlib
import re
import sys
import yaml

path = pathlib.Path(sys.argv[1])
payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
if not isinstance(payload, dict):
    raise SystemExit("invalid calibrate_d3 config")

required = bool(payload.get("calibration_require_real_data", False))
raw = payload.get("dataset_path")
resolved = ""
if isinstance(raw, str):
    token = raw.strip()
    match = re.fullmatch(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))", token)
    if match is not None:
        env_name = match.group(1) or match.group(2) or ""
        resolved = str(os.environ.get(env_name, "")).strip()
    else:
        resolved = os.path.expandvars(raw).strip()
elif raw is not None:
    resolved = str(raw).strip()

print("missing_real_dataset_path" if required and not resolved else "ok")
PY
)"
  if [[ "${CALIBRATION_PATH_CHECK}" == "missing_real_dataset_path" ]]; then
    echo "error: ${CALIBRATE_CONFIG_PATH} requires real D3 vectors. Set MAXIONBENCH_D3_DATASET_PATH=/abs/path/to/laion_d3_vectors.npy" >&2
    echo "error: optional checksum pin: MAXIONBENCH_D3_DATASET_SHA256=<64-char-lowercase-hex>" >&2
    echo "error: alternatively, point --scenario-config-dir at a calibrate_d3.yaml with a concrete dataset_path" >&2
    exit 2
  fi
  maxionbench run \
    --config "${CALIBRATE_CONFIG_PATH}" \
    --seed "${SEED}" \
    --repeats 1 \
    --no-retry \
    --output-dir "${RUN_RESULTS_LOCAL}/calibrate_d3"
  maxionbench verify-d3-calibration --d3-params artifacts/calibration/d3_params.yaml --strict --json
else
  echo "Skipping D3 calibration run/check (--skip-calibration)"
fi

echo "==> Step 3: Slurm plan and protocol audit"
maxionbench verify-slurm-plan --json | tee "${SLURM_PLAN_VERIFY_JSON}"
maxionbench verify-slurm-plan --skip-gpu --json | tee "${SLURM_PLAN_VERIFY_SKIP_GPU_JSON}"
run_submit_slurm_plan \
  --output-root "${RUN_RESULTS_SLURM}" \
  --dry-run \
  --json | tee "${SLURM_SUBMIT_DRY_RUN_JSON}"
run_submit_slurm_plan \
  --output-root "${RUN_RESULTS_SLURM}" \
  --skip-gpu \
  --dry-run \
  --json | tee "${SLURM_SUBMIT_SKIP_GPU_DRY_RUN_JSON}"
run_submit_slurm_plan \
  --scenario-config-dir "${SCENARIO_CONFIG_DIR}" \
  --output-root "${RUN_RESULTS_SLURM}" \
  --skip-gpu \
  --dry-run \
  --json | tee "${SLURM_SUBMIT_PAPER_SKIP_GPU_DRY_RUN_JSON}"
maxionbench validate-slurm-snapshots \
  --verify-path "${SLURM_PLAN_VERIFY_JSON}" \
  --verify-path "${SLURM_PLAN_VERIFY_SKIP_GPU_JSON}" \
  --submit-path "${SLURM_SUBMIT_DRY_RUN_JSON}" \
  --submit-path "${SLURM_SUBMIT_SKIP_GPU_DRY_RUN_JSON}" \
  --submit-path "${SLURM_SUBMIT_PAPER_SKIP_GPU_DRY_RUN_JSON}" \
  --required-baseline-scenario configs/scenarios/s1_ann_frontier_d3.yaml \
  --json | tee "${SLURM_SNAPSHOT_VALIDATION_JSON}"
maxionbench ci-protocol-audit \
  --config-dir configs/scenarios \
  --slurm-dir maxionbench/orchestration/slurm \
  --manifest-dir maxionbench/datasets/manifests \
  --verify-path "${SLURM_PLAN_VERIFY_JSON}" \
  --verify-path "${SLURM_PLAN_VERIFY_SKIP_GPU_JSON}" \
  --submit-path "${SLURM_SUBMIT_DRY_RUN_JSON}" \
  --submit-path "${SLURM_SUBMIT_SKIP_GPU_DRY_RUN_JSON}" \
  --submit-path "${SLURM_SUBMIT_PAPER_SKIP_GPU_DRY_RUN_JSON}" \
  --required-baseline-scenario configs/scenarios/s1_ann_frontier_d3.yaml \
  --output "${CI_PROTOCOL_AUDIT_DEFAULT_JSON}" \
  --strict \
  --json
maxionbench ci-protocol-audit \
  --config-dir "${SCENARIO_CONFIG_DIR}" \
  --slurm-dir maxionbench/orchestration/slurm \
  --manifest-dir maxionbench/datasets/manifests \
  --verify-path "${SLURM_PLAN_VERIFY_JSON}" \
  --verify-path "${SLURM_PLAN_VERIFY_SKIP_GPU_JSON}" \
  --submit-path "${SLURM_SUBMIT_DRY_RUN_JSON}" \
  --submit-path "${SLURM_SUBMIT_SKIP_GPU_DRY_RUN_JSON}" \
  --submit-path "${SLURM_SUBMIT_PAPER_SKIP_GPU_DRY_RUN_JSON}" \
  --required-baseline-scenario configs/scenarios/s1_ann_frontier_d3.yaml \
  --strict-d3-scenario-scale \
  --output "${CI_PROTOCOL_AUDIT_PAPER_JSON}" \
  --strict \
  --json

echo "==> Step 4: Slurm submit"
if [[ "${LAUNCH}" -eq 1 ]]; then
  if [[ "${CPU_ONLY}" -eq 1 ]]; then
    run_submit_slurm_plan \
      --scenario-config-dir "${SCENARIO_CONFIG_DIR}" \
      --output-root "${RUN_RESULTS_SLURM}" \
      --skip-gpu \
      --seed "${SEED}"
  else
    run_submit_slurm_plan \
      --scenario-config-dir "${SCENARIO_CONFIG_DIR}" \
      --output-root "${RUN_RESULTS_SLURM}" \
      --seed "${SEED}"
  fi
else
  echo "Dry-run only. Re-run with --launch to submit jobs."
fi

echo "Figure helper script: ${RENDER_FIGURES_HELPER}"
