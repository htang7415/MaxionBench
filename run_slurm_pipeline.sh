#!/usr/bin/env bash
set -euo pipefail

SUBMIT_ROOT="$(pwd -P)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CLUSTER=""
SLURM_PROFILE=""
CONTAINER_IMAGE="${MAXIONBENCH_CONTAINER_IMAGE:-}"
SHARED_ROOT="${MAXIONBENCH_SHARED_ROOT:-}"
DATASET_ROOT="${MAXIONBENCH_DATASET_ROOT:-}"
DATASET_CACHE_DIR="${MAXIONBENCH_DATASET_CACHE_DIR:-}"
OUTPUT_ROOT="${MAXIONBENCH_OUTPUT_ROOT:-}"
FIGURES_ROOT="${MAXIONBENCH_FIGURES_ROOT:-}"
HF_CACHE_DIR="${MAXIONBENCH_HF_CACHE_DIR:-}"
SCENARIO_CONFIG_DIR="configs/scenarios_paper"
ENGINE_CONFIG_DIR="configs/engines"
RUN_MANIFEST_DIR="artifacts/slurm_manifests/latest"
SLURM_DIR="maxionbench/orchestration/slurm"
SEED="42"
SKIP_GPU=0
SKIP_S6=0
DRY_RUN=1

usage() {
  cat <<'EOF'
Usage:
  bash run_slurm_pipeline.sh --cluster <euler|nrel> --container-image </path/to/maxionbench.sif> [options]

Options:
  --cluster <name>               Cluster profile selector: euler or nrel
  --slurm-profile <name>         Explicit Slurm profile key override
  --container-image <path>       Apptainer image path for Slurm jobs
  --shared-root <path>           Shared base directory; derives dataset/cache/results/figures/HF paths
  --dataset-root <path>          Shared dataset root (default: MAXIONBENCH_DATASET_ROOT)
  --dataset-cache-dir <path>     Shared dataset cache directory
  --output-root <path>           Shared run artifact root
  --figures-root <path>          Shared figure output root
  --hf-cache-dir <path>          Shared Hugging Face cache root
  --scenario-config-dir <path>   Scenario template directory for --full-matrix
  --engine-config-dir <path>     Engine config directory for --full-matrix
  --run-manifest-dir <path>      Output directory for generated run manifests
  --slurm-dir <path>             Slurm script directory
  --seed <int>                   Seed forwarded to submit-slurm-plan
  --skip-gpu                     Omit GPU jobs from the plan
  --skip-s6                      Defer S6 from the plan
  --launch                       Submit jobs; otherwise prints a dry-run plan
  --help                         Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster)
      CLUSTER="$2"
      shift 2
      ;;
    --slurm-profile)
      SLURM_PROFILE="$2"
      shift 2
      ;;
    --container-image)
      CONTAINER_IMAGE="$2"
      shift 2
      ;;
    --shared-root)
      SHARED_ROOT="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --dataset-cache-dir)
      DATASET_CACHE_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --figures-root)
      FIGURES_ROOT="$2"
      shift 2
      ;;
    --hf-cache-dir)
      HF_CACHE_DIR="$2"
      shift 2
      ;;
    --scenario-config-dir)
      SCENARIO_CONFIG_DIR="$2"
      shift 2
      ;;
    --engine-config-dir)
      ENGINE_CONFIG_DIR="$2"
      shift 2
      ;;
    --run-manifest-dir)
      RUN_MANIFEST_DIR="$2"
      shift 2
      ;;
    --slurm-dir)
      SLURM_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --skip-gpu)
      SKIP_GPU=1
      shift
      ;;
    --skip-s6)
      SKIP_S6=1
      shift
      ;;
    --launch)
      DRY_RUN=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CLUSTER}" && -z "${SLURM_PROFILE}" ]]; then
  echo "error: provide --cluster or --slurm-profile" >&2
  exit 2
fi
if [[ -z "${SLURM_PROFILE}" ]]; then
  case "${CLUSTER}" in
    euler)
      SLURM_PROFILE="euler_apptainer"
      ;;
    nrel)
      SLURM_PROFILE="nrel_apptainer"
      ;;
    *)
      echo "error: --cluster must be euler or nrel" >&2
      exit 2
      ;;
  esac
fi
if [[ -z "${CONTAINER_IMAGE}" ]]; then
  echo "error: --container-image is required" >&2
  exit 2
fi
if ! command -v maxionbench >/dev/null 2>&1; then
  echo "error: 'maxionbench' command not found in PATH" >&2
  exit 127
fi

default_shared_root() {
  if [[ -n "${SHARED_ROOT}" ]]; then
    printf '%s\n' "${SHARED_ROOT}"
    return 0
  fi
  printf '%s\n' "${SUBMIT_ROOT}"
}

DEFAULT_SHARED_ROOT="$(default_shared_root)"

export MAXIONBENCH_SHARED_ROOT="${DEFAULT_SHARED_ROOT}"
export MAXIONBENCH_DATASET_ROOT="${DATASET_ROOT:-${DEFAULT_SHARED_ROOT}/dataset}"
export MAXIONBENCH_DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-${DEFAULT_SHARED_ROOT}/.cache}"
export MAXIONBENCH_OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_SHARED_ROOT}/results}"
export MAXIONBENCH_FIGURES_ROOT="${FIGURES_ROOT:-${DEFAULT_SHARED_ROOT}/figures}"
export MAXIONBENCH_HF_CACHE_DIR="${HF_CACHE_DIR:-${DEFAULT_SHARED_ROOT}/.cache/huggingface}"
export MAXIONBENCH_D3_DATASET_PATH="${MAXIONBENCH_DATASET_ROOT}/processed/D3/yfcc-10M/base.npy"

CMD=(
  maxionbench submit-slurm-plan
  --slurm-dir "${SLURM_DIR}"
  --slurm-profile "${SLURM_PROFILE}"
  --scenario-config-dir "${SCENARIO_CONFIG_DIR}"
  --engine-config-dir "${ENGINE_CONFIG_DIR}"
  --run-manifest-dir "${RUN_MANIFEST_DIR}"
  --seed "${SEED}"
  --container-runtime apptainer
  --container-image "${CONTAINER_IMAGE}"
  --output-root "${MAXIONBENCH_OUTPUT_ROOT}"
  --download-datasets
  --preprocess-datasets
  --include-postprocess
  --full-matrix
  --json
)

if [[ -n "${MAXIONBENCH_HF_CACHE_DIR}" ]]; then
  CMD+=(--hf-cache-dir "${MAXIONBENCH_HF_CACHE_DIR}")
fi
if [[ "${SKIP_GPU}" -eq 1 ]]; then
  CMD+=(--skip-gpu)
fi
if [[ "${SKIP_S6}" -eq 1 ]]; then
  CMD+=(--skip-s6)
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  CMD+=(--dry-run)
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
