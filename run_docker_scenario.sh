#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mb() {
  python -m maxionbench.cli "$@"
}

CONFIG_PATH=""
CONFIG_PATH_CONTAINER=""
WAIT_TIMEOUT_S="120"
BUILD_IMAGE=1
STOP_STACK=0
BENCHMARK_SERVICE=""
LOCAL_BENCHMARK=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./run_docker_scenario.sh --config <config.yaml> [options] [-- <maxionbench run args>]

Builds the benchmark image when needed, starts the engine service implied by the
scenario config, waits for adapter health, then runs `maxionbench run` inside the
benchmark container. With `--local-benchmark`, only the backing services use Docker
and the benchmark itself runs from the current host environment.

Options:
  --config <path>      Scenario config path to run inside Docker
  --timeout-s <sec>    Adapter readiness timeout in seconds (default: 120)
  --benchmark-service  Override benchmark service (benchmark or benchmark-gpu)
  --local-benchmark    Run `wait-adapter` + `run` locally instead of in Docker
  --no-build           Skip `docker compose build benchmark`
  --down               Run `docker compose down --remove-orphans` after completion
  -h, --help           Show this help

Examples:
  ./run_docker_scenario.sh --config configs/scenarios/s1_ann_frontier_qdrant.yaml
  ./run_docker_scenario.sh --config configs/scenarios/s1_ann_frontier_pgvector.yaml -- --seed 42 --repeats 1 --no-retry
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --timeout-s)
      WAIT_TIMEOUT_S="${2:-}"
      shift 2
      ;;
    --benchmark-service)
      BENCHMARK_SERVICE="${2:-}"
      shift 2
      ;;
    --local-benchmark)
      LOCAL_BENCHMARK=1
      shift
      ;;
    --no-build)
      BUILD_IMAGE=0
      shift
      ;;
    --down)
      STOP_STACK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "error: --config is required" >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "error: config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

CONFIG_PATH_ABS="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"
case "${CONFIG_PATH_ABS}" in
  "${ROOT_DIR}"/*)
    CONFIG_PATH_CONTAINER="/workspace/${CONFIG_PATH_ABS#"${ROOT_DIR}/"}"
    ;;
  *)
    echo "error: config must live under repo root: ${CONFIG_PATH_ABS}" >&2
    exit 2
    ;;
esac

ENGINE="$(sed -n 's/^engine:[[:space:]]*//p' "${CONFIG_PATH}" | head -n1 | tr -d '"' | xargs)"
SCENARIO="$(sed -n 's/^scenario:[[:space:]]*//p' "${CONFIG_PATH}" | head -n1 | tr -d '"' | xargs)"
REQUIRE_HF="$(sed -n 's/^s5_require_hf_backend:[[:space:]]*//p' "${CONFIG_PATH}" | head -n1 | tr -d '"' | xargs)"
if [[ -z "${ENGINE}" ]]; then
  echo "error: could not resolve top-level engine from ${CONFIG_PATH}" >&2
  exit 2
fi

SERVICES=()
case "${ENGINE}" in
  qdrant)
    if grep -Eq '^[[:space:]]*location:' "${CONFIG_PATH}"; then
      SERVICES=()
    else
      SERVICES=(qdrant)
    fi
    ;;
  milvus)
    SERVICES=(milvus)
    ;;
  opensearch)
    SERVICES=(opensearch)
    ;;
  pgvector)
    SERVICES=(pgvector)
    ;;
  weaviate)
    SERVICES=(weaviate)
    ;;
  mock|faiss-cpu|faiss-gpu|lancedb-inproc|lancedb-service)
    SERVICES=()
    ;;
  *)
    echo "error: unsupported or unknown engine '${ENGINE}'" >&2
    exit 2
    ;;
esac

if [[ "${LOCAL_BENCHMARK}" -eq 0 ]]; then
  if [[ -z "${BENCHMARK_SERVICE}" ]]; then
    BENCHMARK_SERVICE="benchmark"
    case "${ENGINE}" in
      faiss-gpu)
        BENCHMARK_SERVICE="benchmark-gpu"
        ;;
    esac
    if [[ "${SCENARIO}" == "s5_rerank" || "${REQUIRE_HF}" == "true" ]]; then
      BENCHMARK_SERVICE="benchmark-gpu"
    fi
  fi

  case "${BENCHMARK_SERVICE}" in
    benchmark|benchmark-gpu)
      ;;
    *)
      echo "error: --benchmark-service must be benchmark or benchmark-gpu" >&2
      exit 2
      ;;
  esac
fi

if [[ "${LOCAL_BENCHMARK}" -eq 0 || "${#SERVICES[@]}" -gt 0 || "${STOP_STACK}" -eq 1 ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is required" >&2
    exit 127
  fi
fi

if [[ "${BUILD_IMAGE}" -eq 1 && "${LOCAL_BENCHMARK}" -eq 0 ]]; then
  docker compose build "${BENCHMARK_SERVICE}"
fi

if [[ "${#SERVICES[@]}" -gt 0 ]]; then
  if printf '%s\n' "${SERVICES[@]}" | grep -qx 'opensearch'; then
    OPENSEARCH_DATA_DIR="${MAXIONBENCH_OPENSEARCH_DATA_DIR:-${ROOT_DIR}/artifacts/containers/opensearch_data}"
    mkdir -p "${OPENSEARCH_DATA_DIR}"
    # OpenSearch runs as uid 1000 in the container; make the host bind mount writable.
    chmod 0777 "${OPENSEARCH_DATA_DIR}"
  fi
  docker compose up -d "${SERVICES[@]}"
fi

cleanup() {
  if [[ "${STOP_STACK}" -eq 1 ]]; then
    docker compose down --remove-orphans
  fi
}
trap cleanup EXIT

if [[ "${LOCAL_BENCHMARK}" -eq 1 ]]; then
  if ! command -v python >/dev/null 2>&1; then
    echo "error: python is required for --local-benchmark" >&2
    exit 127
  fi
  if [[ "${#SERVICES[@]}" -gt 0 ]]; then
    mb wait-adapter --config "${CONFIG_PATH_ABS}" --timeout-s "${WAIT_TIMEOUT_S}" --json
  fi
  mb run --config "${CONFIG_PATH_ABS}" "${EXTRA_ARGS[@]}"
else
  docker compose run --rm "${BENCHMARK_SERVICE}" wait-adapter --config "${CONFIG_PATH_CONTAINER}" --timeout-s "${WAIT_TIMEOUT_S}" --json
  docker compose run --rm "${BENCHMARK_SERVICE}" run --config "${CONFIG_PATH_CONTAINER}" "${EXTRA_ARGS[@]}"
fi
