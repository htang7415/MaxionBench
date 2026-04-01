#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUN_BUNDLE="artifacts/workstation_runs/latest"
RESULTS_ROOT="results"
DATASET_ROOT="dataset"
CONFORMANCE_DIR="artifacts/conformance"
FIGURES_DIR="artifacts/figures"
COPY_DATASETS=0

usage() {
  cat <<'EOF'
Usage: bash save_results_bundle.sh [options]

Collect the latest MaxionBench run outputs under a repo-local results/ archive.

Options:
  --run-bundle <path>        Source run bundle or scenario output dir
                             (default: artifacts/workstation_runs/latest)
  --results-root <path>      Archive root (default: results)
  --dataset-root <path>      Dataset tree to include (default: dataset)
  --conformance-dir <path>   Conformance outputs to snapshot (default: artifacts/conformance)
  --figures-dir <path>       Figures to snapshot (default: artifacts/figures)
  --copy-datasets            Copy dataset files instead of linking dataset/
  -h, --help                 Show this help
EOF
}

resolve_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

resolve_existing_dir() {
  local path="$1"
  local abs_path
  abs_path="$(resolve_path "${path}")"
  if [[ ! -d "${abs_path}" ]]; then
    echo "error: directory not found: ${path}" >&2
    exit 2
  fi
  (
    cd "${abs_path}"
    pwd -P
  )
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-bundle)
      RUN_BUNDLE="${2:-}"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="${2:-}"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="${2:-}"
      shift 2
      ;;
    --conformance-dir)
      CONFORMANCE_DIR="${2:-}"
      shift 2
      ;;
    --figures-dir)
      FIGURES_DIR="${2:-}"
      shift 2
      ;;
    --copy-datasets)
      COPY_DATASETS=1
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

RUN_BUNDLE_ABS="$(resolve_existing_dir "${RUN_BUNDLE}")"
RESULTS_ROOT_ABS="$(resolve_path "${RESULTS_ROOT}")"
RUN_ID="$(basename "${RUN_BUNDLE_ABS}")"
DEST_DIR="${RESULTS_ROOT_ABS}/${RUN_ID}"
DATASET_ROOT_ABS=""
CONFORMANCE_DIR_ABS=""
FIGURES_DIR_ABS=""

if [[ -e "${DEST_DIR}" ]]; then
  echo "error: archive already exists: ${DEST_DIR}" >&2
  exit 2
fi

if [[ -d "$(resolve_path "${DATASET_ROOT}")" ]]; then
  DATASET_ROOT_ABS="$(resolve_existing_dir "${DATASET_ROOT}")"
fi
if [[ -d "$(resolve_path "${CONFORMANCE_DIR}")" ]]; then
  CONFORMANCE_DIR_ABS="$(resolve_existing_dir "${CONFORMANCE_DIR}")"
fi
if [[ -d "$(resolve_path "${FIGURES_DIR}")" ]]; then
  FIGURES_DIR_ABS="$(resolve_existing_dir "${FIGURES_DIR}")"
fi

mkdir -p "${DEST_DIR}/docs" "${DEST_DIR}/workflow"

cp -a "${RUN_BUNDLE_ABS}" "${DEST_DIR}/run_bundle"
cp -a configs "${DEST_DIR}/configs"
cp -a README.md "${DEST_DIR}/workflow/README.md"
cp -a run_workstation.sh "${DEST_DIR}/workflow/run_workstation.sh"
cp -a run_docker_scenario.sh "${DEST_DIR}/workflow/run_docker_scenario.sh"
cp -a preprocess_all_datasets.sh "${DEST_DIR}/workflow/preprocess_all_datasets.sh"

for doc in project.md prompt.md document.md command.md AGENTS.md; do
  if [[ -f "${doc}" ]]; then
    cp -a "${doc}" "${DEST_DIR}/docs/${doc}"
  fi
done

if [[ -n "${CONFORMANCE_DIR_ABS}" ]]; then
  cp -a "${CONFORMANCE_DIR_ABS}" "${DEST_DIR}/conformance"
fi

if [[ -n "${FIGURES_DIR_ABS}" ]]; then
  cp -a "${FIGURES_DIR_ABS}" "${DEST_DIR}/figures"
fi

dataset_mode="missing"
if [[ -n "${DATASET_ROOT_ABS}" ]]; then
  if [[ "${COPY_DATASETS}" -eq 1 ]]; then
    cp -a "${DATASET_ROOT_ABS}" "${DEST_DIR}/datasets"
    dataset_mode="copied"
  else
    ln -s "${DATASET_ROOT_ABS}" "${DEST_DIR}/datasets"
    dataset_mode="linked"
  fi
fi

cat > "${DEST_DIR}/archive_manifest.txt" <<EOF
run_id=${RUN_ID}
run_bundle=${RUN_BUNDLE_ABS}
archive_dir=${DEST_DIR}
dataset_root=${DATASET_ROOT_ABS:-missing}
dataset_mode=${dataset_mode}
conformance_dir=${CONFORMANCE_DIR_ABS:-missing}
figures_dir=${FIGURES_DIR_ABS:-missing}
EOF

ln -sfn "${RUN_ID}" "${RESULTS_ROOT_ABS}/latest"

echo "Saved run archive:"
echo "- archive_dir: ${DEST_DIR}"
echo "- run_bundle: ${RUN_BUNDLE_ABS}"
echo "- datasets: ${dataset_mode}"
if [[ -n "${CONFORMANCE_DIR_ABS}" ]]; then
  echo "- conformance: ${CONFORMANCE_DIR_ABS}"
fi
if [[ -n "${FIGURES_DIR_ABS}" ]]; then
  echo "- figures: ${FIGURES_DIR_ABS}"
fi
