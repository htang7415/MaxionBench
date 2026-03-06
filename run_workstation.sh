#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SCENARIO_CONFIG_DIR="configs/scenarios_paper"
SEED="42"
SLURM_PROFILE=""
CPU_ONLY=0
LAUNCH=0
SKIP_PYTEST=0
SKIP_CALIBRATION=0
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
  --slurm-profile <name>       Slurm profile preset (your_cluster|your_cluster)
  --seed <int>                 Seed forwarded to submit-slurm-plan (default: 42)
  --cpu-only                   Use skip-gpu mode when submitting Slurm jobs
  --launch                     Submit Slurm jobs after checks pass
  --skip-pytest                Skip pytest -q
  --skip-calibration           Skip calibrate_d3 + verify-d3-calibration
  -h, --help                   Show this help
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

SLURM_PROFILE_ARGS=()
if [[ -n "${SLURM_PROFILE}" ]]; then
  case "${SLURM_PROFILE}" in
    your_cluster|your_cluster)
      SLURM_PROFILE_ARGS=(--slurm-profile "${SLURM_PROFILE}")
      ;;
    *)
      echo "error: --slurm-profile must be one of: your_cluster, your_cluster" >&2
      exit 2
      ;;
  esac
fi

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

SLURM_PLAN_VERIFY_JSON="${RUN_CHECKS_DIR}/slurm_plan_verify.json"
SLURM_PLAN_VERIFY_SKIP_GPU_JSON="${RUN_CHECKS_DIR}/slurm_plan_verify_skip_gpu.json"
SLURM_SUBMIT_DRY_RUN_JSON="${RUN_CHECKS_DIR}/slurm_submit_plan_dry_run.json"
SLURM_SUBMIT_SKIP_GPU_DRY_RUN_JSON="${RUN_CHECKS_DIR}/slurm_submit_plan_skip_gpu_dry_run.json"
SLURM_SUBMIT_PAPER_SKIP_GPU_DRY_RUN_JSON="${RUN_CHECKS_DIR}/slurm_submit_plan_paper_skip_gpu_dry_run.json"
SLURM_SNAPSHOT_VALIDATION_JSON="${RUN_CHECKS_DIR}/slurm_snapshot_validation.json"
CI_PROTOCOL_AUDIT_JSON="${RUN_CHECKS_DIR}/ci_protocol_audit.json"

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

exec > >(tee -a "${REPORT_LOG}") 2>&1

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
- ci_protocol_audit: \`${CI_PROTOCOL_AUDIT_JSON}\`
EOF

  ln -sfn "${RUN_ID}" "artifacts/workstation_runs/latest"
  echo "Run report saved: ${RUN_BUNDLE_ROOT}"
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
  maxionbench run \
    --config "${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml" \
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
maxionbench submit-slurm-plan \
  "${SLURM_PROFILE_ARGS[@]}" \
  --output-root "${RUN_RESULTS_SLURM}" \
  --dry-run \
  --json | tee "${SLURM_SUBMIT_DRY_RUN_JSON}"
maxionbench submit-slurm-plan \
  "${SLURM_PROFILE_ARGS[@]}" \
  --output-root "${RUN_RESULTS_SLURM}" \
  --skip-gpu \
  --dry-run \
  --json | tee "${SLURM_SUBMIT_SKIP_GPU_DRY_RUN_JSON}"
maxionbench submit-slurm-plan \
  "${SLURM_PROFILE_ARGS[@]}" \
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
  --output "${CI_PROTOCOL_AUDIT_JSON}" \
  --strict \
  --json

echo "==> Step 4: Slurm submit"
if [[ "${LAUNCH}" -eq 1 ]]; then
  if [[ "${CPU_ONLY}" -eq 1 ]]; then
    maxionbench submit-slurm-plan \
      "${SLURM_PROFILE_ARGS[@]}" \
      --scenario-config-dir "${SCENARIO_CONFIG_DIR}" \
      --output-root "${RUN_RESULTS_SLURM}" \
      --skip-gpu \
      --seed "${SEED}"
  else
    maxionbench submit-slurm-plan \
      "${SLURM_PROFILE_ARGS[@]}" \
      --scenario-config-dir "${SCENARIO_CONFIG_DIR}" \
      --output-root "${RUN_RESULTS_SLURM}" \
      --seed "${SEED}"
  fi
else
  echo "Dry-run only. Re-run with --launch to submit jobs."
fi

echo "Figure helper script: ${RENDER_FIGURES_HELPER}"
