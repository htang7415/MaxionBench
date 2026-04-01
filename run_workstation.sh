#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SCENARIO_CONFIG_DIR="configs/scenarios_paper"
ENGINE_CONFIG_DIR="configs/engines"
LANE="cpu"
SEED="42"
SKIP_S6=0
SKIP_CALIBRATION=0
SCRATCH_DIR="${MAXIONBENCH_WORKSTATION_SCRATCH:-${ROOT_DIR}}"
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'EOF'
Usage: bash run_workstation.sh [options]

Run the canonical local workstation workflow for MaxionBench.
The default full-run lane targets Linux workstations and uses:
  - scenario templates: configs/scenarios_paper
  - engine configs:     configs/engines
  - lane:               cpu

Execution flow:
  verify pins/manifests -> optional D3 calibration -> build local run matrix
  -> preflight D2/D3 storage -> run generated configs sequentially

Each invocation writes a bundle under artifacts/workstation_runs/<run_id>/.

Options:
  --scenario-config-dir <dir>  Scenario config directory (default: configs/scenarios_paper)
  --engine-config-dir <dir>    Engine config directory (default: configs/engines)
  --lane <cpu|gpu|all>         Matrix lane to run (default: cpu)
  --seed <int>                 Seed forwarded to benchmark runs (default: 42)
  --scratch-dir <path>         Filesystem root used for local preflight (default: repo root or MAXIONBENCH_WORKSTATION_SCRATCH)
  --skip-s6                    Exclude S6 from generated run matrix
  --skip-calibration           Skip calibrate_d3 + verify-d3-calibration
  -h, --help                   Show this help

Notes:
  - GPU lane is explicit because the A100 may be shared.
  - Service engines are run through Docker Compose.
  - Local/in-process engines run directly through `maxionbench run`.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario-config-dir)
      SCENARIO_CONFIG_DIR="${2:-}"
      shift 2
      ;;
    --engine-config-dir)
      ENGINE_CONFIG_DIR="${2:-}"
      shift 2
      ;;
    --lane)
      LANE="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --scratch-dir)
      SCRATCH_DIR="${2:-}"
      shift 2
      ;;
    --skip-s6)
      SKIP_S6=1
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

case "${LANE}" in
  cpu|gpu|all)
    ;;
  *)
    echo "error: --lane must be cpu, gpu, or all" >&2
    exit 2
    ;;
esac

if ! command -v maxionbench >/dev/null 2>&1; then
  echo "error: 'maxionbench' command not found in PATH" >&2
  exit 127
fi
if ! command -v python >/dev/null 2>&1; then
  echo "error: 'python' command not found in PATH" >&2
  exit 127
fi
if [[ ! -d "${SCENARIO_CONFIG_DIR}" ]]; then
  echo "error: missing scenario config dir ${SCENARIO_CONFIG_DIR}" >&2
  exit 2
fi
if [[ ! -d "${ENGINE_CONFIG_DIR}" ]]; then
  echo "error: missing engine config dir ${ENGINE_CONFIG_DIR}" >&2
  exit 2
fi
if [[ ! -d "${SCRATCH_DIR}" ]]; then
  echo "error: scratch dir does not exist: ${SCRATCH_DIR}" >&2
  exit 2
fi

CALIBRATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml"
if [[ "${SKIP_CALIBRATION}" -eq 0 && ! -f "${CALIBRATE_CONFIG_PATH}" ]]; then
  echo "error: missing ${CALIBRATE_CONFIG_PATH}" >&2
  exit 2
fi

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="workstation_${RUN_TS}"
RUN_BUNDLE_ROOT="artifacts/workstation_runs/${RUN_ID}"
RUN_REPORT_DIR="${RUN_BUNDLE_ROOT}/reports"
RUN_CHECKS_DIR="${RUN_BUNDLE_ROOT}/checks"
RUN_PREFLIGHT_DIR="${RUN_CHECKS_DIR}/preflight"
RUN_MATRIX_DIR="${RUN_CHECKS_DIR}/run_matrix"
RUN_RESULTS_ROOT="${RUN_BUNDLE_ROOT}/results"
RUN_RESULTS_LOCAL="${RUN_RESULTS_ROOT}/local"
RUN_FIGURES_ROOT="${RUN_BUNDLE_ROOT}/figures"
RUN_FIGURES_MILESTONES="${RUN_FIGURES_ROOT}/milestones"
RUN_FIGURES_FINAL="${RUN_FIGURES_ROOT}/final"
RUN_HELPERS_DIR="${RUN_BUNDLE_ROOT}/helpers"

mkdir -p \
  "${RUN_REPORT_DIR}" \
  "${RUN_CHECKS_DIR}" \
  "${RUN_PREFLIGHT_DIR}" \
  "${RUN_MATRIX_DIR}" \
  "${RUN_RESULTS_LOCAL}" \
  "${RUN_FIGURES_MILESTONES}" \
  "${RUN_FIGURES_FINAL}" \
  "${RUN_HELPERS_DIR}"

REPORT_LOG="${RUN_REPORT_DIR}/run.log"
REPORT_SUMMARY_TXT="${RUN_REPORT_DIR}/summary.txt"
REPORT_SUMMARY_MD="${RUN_REPORT_DIR}/summary.md"
RUN_MATRIX_SUMMARY_JSON="${RUN_CHECKS_DIR}/run_matrix_summary.json"
RUN_MATRIX_JSON="${RUN_MATRIX_DIR}/run_matrix.json"
RENDER_FIGURES_HELPER="${RUN_HELPERS_DIR}/render_figures.sh"
START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"

exec > >(tee -a "${REPORT_LOG}") 2>&1

cat > "${RENDER_FIGURES_HELPER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${ROOT_DIR}"
INPUT_DIR="\${1:-${ROOT_DIR}/${RUN_RESULTS_LOCAL}}"
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

export MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI="${ROOT_DIR}/${RUN_RESULTS_LOCAL}/lancedb/service"

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

  cat > "${REPORT_SUMMARY_TXT}" <<EOF
run_id=${RUN_ID}
status=${status}
exit_code=${exit_code}
start_utc=${START_TS}
end_utc=${end_ts}
duration_s=${duration_s}
repo_root=${ROOT_DIR}
scenario_config_dir=${SCENARIO_CONFIG_DIR}
engine_config_dir=${ENGINE_CONFIG_DIR}
lane=${LANE}
seed=${SEED}
scratch_dir=${SCRATCH_DIR}
skip_s6=${SKIP_S6}
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
- engine_config_dir: \`${ENGINE_CONFIG_DIR}\`
- lane: \`${LANE}\`
- seed: \`${SEED}\`
- scratch_dir: \`${SCRATCH_DIR}\`
- skip_s6: \`${SKIP_S6}\`
- skip_calibration: \`${SKIP_CALIBRATION}\`
- args: \`${args_rendered}\`

## Bundle Paths

- bundle_root: \`${RUN_BUNDLE_ROOT}\`
- reports: \`${RUN_REPORT_DIR}\`
- checks: \`${RUN_CHECKS_DIR}\`
- results_local: \`${RUN_RESULTS_LOCAL}\`
- figures_milestones: \`${RUN_FIGURES_MILESTONES}\`
- figures_final: \`${RUN_FIGURES_FINAL}\`
- render_figures_helper: \`${RENDER_FIGURES_HELPER}\`
EOF

  ln -sfn "${RUN_ID}" "artifacts/workstation_runs/latest"
  echo "Run report saved: ${RUN_BUNDLE_ROOT}"
}

trap finalize_report EXIT

engine_from_config() {
  sed -n 's/^engine:[[:space:]]*//p' "$1" | head -n1 | tr -d '"' | xargs
}

service_engine() {
  case "$1" in
    qdrant|milvus|opensearch|pgvector|weaviate)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

should_preflight() {
  case "$1" in
    D2|D3)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

docker_benchmark_service() {
  case "$1" in
    gpu)
      printf '%s\n' "benchmark-gpu"
      ;;
    *)
      printf '%s\n' "benchmark"
      ;;
  esac
}

run_generated_config() {
  local group="$1"
  local config_path="$2"
  local engine="$3"
  local dataset_bundle="$4"
  local config_stem
  config_stem="$(basename "${config_path}" .yaml)"

  if should_preflight "${dataset_bundle}"; then
    python -m maxionbench.orchestration.local_preflight \
      --config "${config_path}" \
      --scratch-dir "${SCRATCH_DIR}" \
      --json | tee "${RUN_PREFLIGHT_DIR}/${config_stem}.json"
  fi

  if service_engine "${engine}"; then
    if ! command -v docker >/dev/null 2>&1; then
      echo "error: docker is required for service engine ${engine}" >&2
      exit 127
    fi
    bash run_docker_scenario.sh \
      --config "${config_path}" \
      --benchmark-service "$(docker_benchmark_service "${group}")" \
      -- --seed "${SEED}"
    return 0
  fi

  maxionbench run --config "${config_path}" --seed "${SEED}"
}

echo "Run bundle root: ${RUN_BUNDLE_ROOT}"
echo "Run log: ${REPORT_LOG}"

echo "==> Step 1: workstation sanity"
maxionbench verify-pins --config-dir configs/scenarios --json
if [[ "${SCENARIO_CONFIG_DIR}" != "configs/scenarios" ]]; then
  VERIFY_ARGS=(--config-dir "${SCENARIO_CONFIG_DIR}" --json)
  if [[ "${SCENARIO_CONFIG_DIR}" == *"scenarios_paper"* ]]; then
    VERIFY_ARGS+=(--strict-d3-scenario-scale)
  fi
  maxionbench verify-pins "${VERIFY_ARGS[@]}"
fi
maxionbench verify-dataset-manifests --manifest-dir maxionbench/datasets/manifests --json
maxionbench verify-conformance-configs --config-dir configs/conformance --json

echo "==> Step 2: D3 calibration gate"
if [[ "${SKIP_CALIBRATION}" -eq 0 ]]; then
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

def resolve_value(raw):
    if isinstance(raw, str):
        token = raw.strip()
        match = re.fullmatch(
            r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}|([A-Za-z_][A-Za-z0-9_]*))",
            token,
        )
        if match is not None:
            env_name = match.group(1) or match.group(3) or ""
            default_value = match.group(2) if match.group(1) else None
            env_value = str(os.environ.get(env_name, "")).strip()
            if env_value:
                return env_value
            return str(default_value or "").strip()
        return os.path.expandvars(token).strip()
    if raw is None:
        return ""
    return str(raw).strip()

required = bool(payload.get("calibration_require_real_data", False))
processed_root = resolve_value(payload.get("processed_dataset_path"))
dataset_path = resolve_value(payload.get("dataset_path"))

resolved = ""
if processed_root:
    candidate = pathlib.Path(processed_root).expanduser()
    if candidate.suffix.lower() not in {".npy", ".npz"}:
        candidate = candidate / "base.npy"
    resolved = str(candidate)
elif dataset_path:
    resolved = str(pathlib.Path(dataset_path).expanduser())

if required and not resolved:
    print("missing_real_dataset_path")
elif required and not pathlib.Path(resolved).exists():
    print(f"missing_real_dataset_file:{resolved}")
else:
    print("ok")
PY
)"
  if [[ "${CALIBRATION_PATH_CHECK}" == "missing_real_dataset_path" ]]; then
    echo "error: ${CALIBRATE_CONFIG_PATH} requires real D3 vectors." >&2
    echo "error: run bash preprocess_all_datasets.sh first or set MAXIONBENCH_D3_DATASET_PATH=/abs/path/to/d3_vectors.npy" >&2
    exit 2
  fi
  if [[ "${CALIBRATION_PATH_CHECK}" == missing_real_dataset_file:* ]]; then
    echo "error: real D3 vectors not found: ${CALIBRATION_PATH_CHECK#missing_real_dataset_file:}" >&2
    echo "error: run bash preprocess_all_datasets.sh first or point MAXIONBENCH_D3_DATASET_PATH at an existing .npy/.npz file" >&2
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

echo "==> Step 3: build local run matrix"
MATRIX_ARGS=(
  --scenario-config-dir "${SCENARIO_CONFIG_DIR}"
  --engine-config-dir "${ENGINE_CONFIG_DIR}"
  --out-dir "${RUN_MATRIX_DIR}"
  --output-root "${RUN_RESULTS_LOCAL}"
  --lane "${LANE}"
  --json
)
if [[ "${SKIP_S6}" -eq 1 ]]; then
  MATRIX_ARGS+=(--skip-s6)
fi
python -m maxionbench.orchestration.run_matrix "${MATRIX_ARGS[@]}" | tee "${RUN_MATRIX_SUMMARY_JSON}"

mapfile -t MATRIX_ROWS < <(python - <<'PY' "${RUN_MATRIX_JSON}" "${LANE}"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lane = sys.argv[2].strip().lower()
payload = json.loads(path.read_text(encoding="utf-8"))
rows = []
if lane in {"cpu", "all"}:
    rows.extend(payload.get("cpu_rows", []))
if lane in {"gpu", "all"}:
    rows.extend(payload.get("gpu_rows", []))
for row in rows:
    print("\t".join([
        str(row.get("group", "")),
        str(row.get("config_path", "")),
        str(row.get("engine", "")),
        str(row.get("dataset_bundle", "")),
    ]))
PY
)

if [[ "${#MATRIX_ROWS[@]}" -eq 0 ]]; then
  echo "error: generated run matrix is empty for lane ${LANE}" >&2
  exit 2
fi

echo "==> Step 4: execute generated configs (${#MATRIX_ROWS[@]} runs)"
for row in "${MATRIX_ROWS[@]}"; do
  IFS=$'\t' read -r group config_path engine dataset_bundle <<< "${row}"
  echo "--> ${group}: ${engine} ${config_path}"
  run_generated_config "${group}" "${config_path}" "${engine}" "${dataset_bundle}"
done

echo "Figure helper script: ${RENDER_FIGURES_HELPER}"
