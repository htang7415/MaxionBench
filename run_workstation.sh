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
SKIP_COMPLETED=0
CONTINUE_ON_FAILURE=0
PREBUILD_IMAGES=1
RESUME_BUNDLE=""
ENGINE_FILTER=""
SCENARIO_FILTER=""
TEMPLATE_FILTER=""
MAX_RUNS=""
GPU_BENCHMARK_MODE="docker"
SCRATCH_DIR="${MAXIONBENCH_WORKSTATION_SCRATCH:-${ROOT_DIR}}"
ORIGINAL_ARGS=("$@")
COMPLETED_RUNS=0
SKIPPED_RUNS=0
FAILED_RUNS=0
CALIBRATION_STATUS="ran"
FAILED_ROW_LABELS=()

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
  --skip-completed             Skip rows whose output_dir already has run_status=success
  --continue-on-failure        Continue executing later rows after an individual row fails
  --resume-bundle <path|id>    Reuse an existing bundle root under artifacts/workstation_runs
  --engine-filter <csv>        Run only matching engine names from the generated matrix
  --scenario-filter <csv>      Run only matching scenario names from the generated matrix
  --template-filter <csv>      Run only matching template file names from the generated matrix
  --max-runs <int>             Execute at most N selected matrix rows after filtering
  --gpu-benchmark-mode <mode>  GPU benchmark execution: docker or local (default: docker)
  --no-prebuild                Skip one-time docker compose build; service rows build on demand
  -h, --help                   Show this help

Notes:
  - GPU lane is explicit because the A100 may be shared.
  - GPU-lane rows use Docker by default; `--gpu-benchmark-mode local` runs the benchmark from the host env.
  - CPU local/in-process rows run directly through `maxionbench run`.
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
    --skip-completed)
      SKIP_COMPLETED=1
      shift
      ;;
    --continue-on-failure)
      CONTINUE_ON_FAILURE=1
      shift
      ;;
    --resume-bundle)
      RESUME_BUNDLE="${2:-}"
      shift 2
      ;;
    --engine-filter)
      ENGINE_FILTER="${2:-}"
      shift 2
      ;;
    --scenario-filter)
      SCENARIO_FILTER="${2:-}"
      shift 2
      ;;
    --template-filter)
      TEMPLATE_FILTER="${2:-}"
      shift 2
      ;;
    --max-runs)
      MAX_RUNS="${2:-}"
      shift 2
      ;;
    --gpu-benchmark-mode)
      GPU_BENCHMARK_MODE="${2:-}"
      shift 2
      ;;
    --no-prebuild)
      PREBUILD_IMAGES=0
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

case "${GPU_BENCHMARK_MODE}" in
  docker|local)
    ;;
  *)
    echo "error: --gpu-benchmark-mode must be docker or local" >&2
    exit 2
    ;;
esac

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
if [[ -n "${MAX_RUNS}" ]]; then
  if ! [[ "${MAX_RUNS}" =~ ^[0-9]+$ ]] || [[ "${MAX_RUNS}" -eq 0 ]]; then
    echo "error: --max-runs must be a positive integer" >&2
    exit 2
  fi
fi

CALIBRATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml"
D3_PARAMS_PATH="artifacts/calibration/d3_params.yaml"
if [[ "${SKIP_CALIBRATION}" -eq 0 && ! -f "${CALIBRATE_CONFIG_PATH}" ]]; then
  echo "error: missing ${CALIBRATE_CONFIG_PATH}" >&2
  exit 2
fi

if [[ -n "${RESUME_BUNDLE}" ]]; then
  if [[ -d "${RESUME_BUNDLE}" ]]; then
    RUN_BUNDLE_ROOT="$(cd "$(dirname "${RESUME_BUNDLE}")" && pwd)/$(basename "${RESUME_BUNDLE}")"
  elif [[ -d "${ROOT_DIR}/artifacts/workstation_runs/${RESUME_BUNDLE}" ]]; then
    RUN_BUNDLE_ROOT="${ROOT_DIR}/artifacts/workstation_runs/${RESUME_BUNDLE}"
  else
    echo "error: resume bundle not found: ${RESUME_BUNDLE}" >&2
    exit 2
  fi
  RUN_ID="$(basename "${RUN_BUNDLE_ROOT}")"
else
  RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_ID="workstation_${RUN_TS}"
  RUN_BUNDLE_ROOT="${ROOT_DIR}/artifacts/workstation_runs/${RUN_ID}"
fi
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
python -m maxionbench.cli report --input "\${INPUT_DIR}" --mode milestones --out "\${MILESTONES_OUT}"
python -m maxionbench.cli report --input "\${INPUT_DIR}" --mode final --out "\${FINAL_OUT}"
echo "figures generated:"
echo "- milestones: \${MILESTONES_OUT}"
echo "- final: \${FINAL_OUT}"
EOF
chmod +x "${RENDER_FIGURES_HELPER}"

export MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI="${RUN_RESULTS_LOCAL}/lancedb/service"

mb() {
  python -m maxionbench.cli "$@"
}

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
gpu_benchmark_mode=${GPU_BENCHMARK_MODE}
seed=${SEED}
scratch_dir=${SCRATCH_DIR}
skip_s6=${SKIP_S6}
skip_calibration=${SKIP_CALIBRATION}
skip_completed=${SKIP_COMPLETED}
continue_on_failure=${CONTINUE_ON_FAILURE}
prebuild_images=${PREBUILD_IMAGES}
resume_bundle=${RESUME_BUNDLE:-}
engine_filter=${ENGINE_FILTER:-}
scenario_filter=${SCENARIO_FILTER:-}
template_filter=${TEMPLATE_FILTER:-}
max_runs=${MAX_RUNS:-}
calibration_status=${CALIBRATION_STATUS}
completed_runs=${COMPLETED_RUNS}
skipped_runs=${SKIPPED_RUNS}
failed_runs=${FAILED_RUNS}
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
- gpu_benchmark_mode: \`${GPU_BENCHMARK_MODE}\`
- seed: \`${SEED}\`
- scratch_dir: \`${SCRATCH_DIR}\`
- skip_s6: \`${SKIP_S6}\`
- skip_calibration: \`${SKIP_CALIBRATION}\`
- skip_completed: \`${SKIP_COMPLETED}\`
- continue_on_failure: \`${CONTINUE_ON_FAILURE}\`
- prebuild_images: \`${PREBUILD_IMAGES}\`
- resume_bundle: \`${RESUME_BUNDLE:-}\`
- engine_filter: \`${ENGINE_FILTER:-}\`
- scenario_filter: \`${SCENARIO_FILTER:-}\`
- template_filter: \`${TEMPLATE_FILTER:-}\`
- max_runs: \`${MAX_RUNS:-}\`
- calibration_status: \`${CALIBRATION_STATUS}\`
- completed_runs: \`${COMPLETED_RUNS}\`
- skipped_runs: \`${SKIPPED_RUNS}\`
- failed_runs: \`${FAILED_RUNS}\`
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

output_dir_from_config() {
  sed -n 's/^output_dir:[[:space:]]*//p' "$1" | head -n1 | tr -d '"' | xargs
}

matches_csv_filter() {
  local filter_csv="$1"
  local value="$2"
  local item
  local normalized
  if [[ -z "${filter_csv}" ]]; then
    return 0
  fi
  IFS=',' read -ra items <<< "${filter_csv}"
  for item in "${items[@]}"; do
    normalized="$(printf '%s' "${item}" | xargs)"
    if [[ -n "${normalized}" && "${value}" == "${normalized}" ]]; then
      return 0
    fi
  done
  return 1
}

run_status_is_success() {
  local run_dir="$1"
  local status_path="${run_dir}/run_status.json"
  [[ -f "${status_path}" ]] && grep -q '"status":[[:space:]]*"success"' "${status_path}"
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

scenario_requires_d3_params() {
  local dataset_bundle="$1"
  local scenario="$2"
  if [[ "${dataset_bundle}" != "D3" ]]; then
    return 1
  fi
  case "${scenario}" in
    s2_filtered_ann|s3_churn_smooth|s3b_churn_bursty)
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

docker_has_nvidia_runtime() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
}

docker_gpu_runtime_ready() {
  docker compose run --rm --entrypoint python benchmark-gpu -c \
    'import sys; import torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
    >/dev/null 2>&1
}

local_gpu_runtime_ready() {
  python - <<'PY'
import importlib.util
import sys

errors = []

torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    errors.append("python package `torch` is missing")
else:
    import torch  # type: ignore[import-not-found]

    if not bool(getattr(torch.cuda, "is_available", lambda: False)()):
        errors.append("torch.cuda.is_available() is false")

transformers_spec = importlib.util.find_spec("transformers")
if transformers_spec is None:
    errors.append("python package `transformers` is missing")

faiss_spec = importlib.util.find_spec("faiss")
if faiss_spec is None:
    errors.append("python package `faiss` is missing")
else:
    import faiss  # type: ignore[import-not-found]

    if not hasattr(faiss, "StandardGpuResources"):
        errors.append("faiss import does not expose GPU bindings; uninstall `faiss-cpu` and reinstall GPU FAISS")

if errors:
    for item in errors:
        print(item, file=sys.stderr)
    sys.exit(1)
PY
}

run_generated_config() {
  local group="$1"
  local config_path="$2"
  local engine="$3"
  local dataset_bundle="$4"
  local config_stem
  config_stem="$(basename "${config_path}" .yaml)"
  local run_args=(--seed "${SEED}")
  if [[ -f "${D3_PARAMS_PATH}" ]]; then
    run_args+=(--d3-params "${D3_PARAMS_PATH}")
  fi

  if should_preflight "${dataset_bundle}"; then
    python -m maxionbench.orchestration.local_preflight \
      --config "${config_path}" \
      --scratch-dir "${SCRATCH_DIR}" \
      --json | tee "${RUN_PREFLIGHT_DIR}/${config_stem}.json"
  fi

  if [[ "${group}" == "gpu" && "${GPU_BENCHMARK_MODE}" == "local" ]]; then
    if service_engine "${engine}"; then
      bash run_docker_scenario.sh \
        --config "${config_path}" \
        --local-benchmark \
        -- "${run_args[@]}"
    else
      mb run --config "${config_path}" "${run_args[@]}"
    fi
    return 0
  fi

  if [[ "${group}" == "gpu" ]] || service_engine "${engine}"; then
    if ! command -v docker >/dev/null 2>&1; then
      echo "error: docker is required for ${group} lane / engine ${engine}" >&2
      exit 127
    fi
    local docker_build_args=()
    if [[ "${PREBUILD_IMAGES}" -eq 1 ]]; then
      docker_build_args+=(--no-build)
    fi
    bash run_docker_scenario.sh \
      --config "${config_path}" \
      "${docker_build_args[@]}" \
      --benchmark-service "$(docker_benchmark_service "${group}")" \
      -- "${run_args[@]}"
    return 0
  fi

  mb run --config "${config_path}" "${run_args[@]}"
}

echo "Run bundle root: ${RUN_BUNDLE_ROOT}"
echo "Run log: ${REPORT_LOG}"

echo "==> Step 1: workstation sanity"
mb verify-pins --config-dir configs/scenarios --json
if [[ "${SCENARIO_CONFIG_DIR}" != "configs/scenarios" ]]; then
  VERIFY_ARGS=(--config-dir "${SCENARIO_CONFIG_DIR}" --json)
  if [[ "${SCENARIO_CONFIG_DIR}" == *"scenarios_paper"* ]]; then
    VERIFY_ARGS+=(--strict-d3-scenario-scale)
  fi
  mb verify-pins "${VERIFY_ARGS[@]}"
fi
mb verify-dataset-manifests --manifest-dir maxionbench/datasets/manifests --json
mb verify-conformance-configs --config-dir configs/conformance --json

echo "==> Step 2: D3 calibration gate"
if [[ "${SKIP_CALIBRATION}" -eq 0 ]]; then
  CALIBRATION_PATH_CHECK="$(python - <<'PY' "${CALIBRATE_CONFIG_PATH}"
import pathlib
import sys
import yaml

from maxionbench.orchestration.config_schema import expand_env_placeholders

path = pathlib.Path(sys.argv[1])
payload = expand_env_placeholders(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
if not isinstance(payload, dict):
    raise SystemExit("invalid calibrate_d3 config")

required = bool(payload.get("calibration_require_real_data", False))
processed_root = str(payload.get("processed_dataset_path") or "").strip()
dataset_path = str(payload.get("dataset_path") or "").strip()

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
  if [[ "${SKIP_COMPLETED}" -eq 1 ]] && run_status_is_success "${RUN_RESULTS_LOCAL}/calibrate_d3"; then
    echo "Skipping completed calibration output: ${RUN_RESULTS_LOCAL}/calibrate_d3"
    CALIBRATION_STATUS="skipped_completed"
  else
    mb run \
      --config "${CALIBRATE_CONFIG_PATH}" \
      --seed "${SEED}" \
      --repeats 1 \
      --no-retry \
      --output-dir "${RUN_RESULTS_LOCAL}/calibrate_d3"
    CALIBRATION_STATUS="ran"
  fi
  mb verify-d3-calibration --d3-params "${D3_PARAMS_PATH}" --strict --json
else
  echo "Skipping D3 calibration run/check (--skip-calibration)"
  CALIBRATION_STATUS="skipped_flag"
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
        str(row.get("scenario", "")),
        str(row.get("template_name", "")),
    ]))
PY
)

if [[ "${#MATRIX_ROWS[@]}" -eq 0 ]]; then
  echo "error: generated run matrix is empty for lane ${LANE}" >&2
  exit 2
fi

FILTERED_ROWS=()
for row in "${MATRIX_ROWS[@]}"; do
  IFS=$'\t' read -r group config_path engine dataset_bundle scenario template_name <<< "${row}"
  if ! matches_csv_filter "${ENGINE_FILTER}" "${engine}"; then
    continue
  fi
  if ! matches_csv_filter "${SCENARIO_FILTER}" "${scenario}"; then
    continue
  fi
  if ! matches_csv_filter "${TEMPLATE_FILTER}" "${template_name}"; then
    continue
  fi
  FILTERED_ROWS+=("${row}")
done
MATRIX_ROWS=("${FILTERED_ROWS[@]}")

if [[ -n "${MAX_RUNS}" && "${#MATRIX_ROWS[@]}" -gt "${MAX_RUNS}" ]]; then
  MATRIX_ROWS=("${MATRIX_ROWS[@]:0:${MAX_RUNS}}")
fi

if [[ "${#MATRIX_ROWS[@]}" -eq 0 ]]; then
  echo "error: no matrix rows remain after applying filters" >&2
  exit 2
fi

NEEDS_BENCHMARK_BUILD=0
NEEDS_BENCHMARK_GPU_BUILD=0
REQUIRES_D3_PARAMS=0
for row in "${MATRIX_ROWS[@]}"; do
  IFS=$'\t' read -r group _config_path engine dataset_bundle scenario _template_name <<< "${row}"
  if [[ "${group}" == "gpu" ]]; then
    if [[ "${GPU_BENCHMARK_MODE}" == "docker" ]]; then
      NEEDS_BENCHMARK_GPU_BUILD=1
    fi
  elif service_engine "${engine}"; then
    NEEDS_BENCHMARK_BUILD=1
  fi
  if scenario_requires_d3_params "${dataset_bundle}" "${scenario}"; then
    REQUIRES_D3_PARAMS=1
  fi
done

if [[ "${REQUIRES_D3_PARAMS}" -eq 1 && ! -f "${D3_PARAMS_PATH}" ]]; then
  echo "error: selected strict D3 rows require ${D3_PARAMS_PATH}" >&2
  echo "error: rerun calibration first or provide the calibrated file before using --skip-calibration" >&2
  exit 2
fi

if [[ "${PREBUILD_IMAGES}" -eq 1 ]]; then
  if [[ "${NEEDS_BENCHMARK_BUILD}" -eq 1 || "${NEEDS_BENCHMARK_GPU_BUILD}" -eq 1 ]]; then
    if ! command -v docker >/dev/null 2>&1; then
      echo "error: docker is required for selected gpu/service-engine rows" >&2
      exit 127
    fi
  fi
  if [[ "${NEEDS_BENCHMARK_BUILD}" -eq 1 ]]; then
    docker compose build benchmark
  fi
  if [[ "${NEEDS_BENCHMARK_GPU_BUILD}" -eq 1 ]]; then
    if ! docker_has_nvidia_runtime; then
      echo "error: docker daemon does not expose an NVIDIA runtime" >&2
      echo "error: install/configure nvidia-container-toolkit and restart Docker before gpu/all lanes" >&2
      exit 2
    fi
    docker compose build benchmark-gpu
    if ! docker_gpu_runtime_ready; then
      echo "error: benchmark-gpu cannot access a GPU through Docker on this workstation" >&2
      echo "error: Docker has an NVIDIA runtime but the GPU is still unavailable to the container" >&2
      echo "error: fix the host NVIDIA driver/device state before rerunning gpu/all lanes" >&2
      exit 2
    fi
  fi
fi

if [[ "${LANE}" != "cpu" && "${GPU_BENCHMARK_MODE}" == "local" ]]; then
  if ! local_gpu_runtime_ready; then
    echo "error: local gpu benchmark mode is not ready in the current Python environment" >&2
    echo "error: install torch/transformers and ensure `import faiss` exposes GPU bindings before rerunning" >&2
    exit 2
  fi
fi

echo "==> Step 4: execute generated configs (${#MATRIX_ROWS[@]} selected runs)"
for row in "${MATRIX_ROWS[@]}"; do
  IFS=$'\t' read -r group config_path engine dataset_bundle scenario template_name <<< "${row}"
  output_dir="$(output_dir_from_config "${config_path}")"
  if [[ -z "${output_dir}" ]]; then
    echo "error: could not resolve output_dir from ${config_path}" >&2
    exit 2
  fi
  if [[ "${SKIP_COMPLETED}" -eq 1 ]] && run_status_is_success "${output_dir}"; then
    echo "--> skip completed: ${group}: ${engine} ${config_path}"
    SKIPPED_RUNS="$((SKIPPED_RUNS + 1))"
    continue
  fi
  echo "--> ${group}: ${engine} ${config_path}"
  if run_generated_config "${group}" "${config_path}" "${engine}" "${dataset_bundle}"; then
    COMPLETED_RUNS="$((COMPLETED_RUNS + 1))"
    continue
  fi
  FAILED_RUNS="$((FAILED_RUNS + 1))"
  FAILED_ROW_LABELS+=("${group}:${engine}:${scenario}:${template_name}")
  if [[ "${CONTINUE_ON_FAILURE}" -eq 1 ]]; then
    echo "warning: continuing after failed row ${group}:${engine}:${scenario}:${template_name}" >&2
    continue
  fi
  exit 1
done

if [[ "${FAILED_RUNS}" -gt 0 ]]; then
  echo "Failed rows (${FAILED_RUNS}):" >&2
  printf '  - %s\n' "${FAILED_ROW_LABELS[@]}" >&2
  exit 1
fi

echo "Figure helper script: ${RENDER_FIGURES_HELPER}"
