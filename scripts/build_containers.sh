#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "${SCRIPT_PATH}" == */* ]]; then
  ROOT_DIR="$(cd "${SCRIPT_PATH%/*}/.." && pwd -P)"
else
  ROOT_DIR="$(pwd -P)"
fi
cd "${ROOT_DIR}"

OUTPUT_DIR=""
ONLY_MISSING=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/build_containers.sh --output-dir <path> [--only-missing]

Builds the main MaxionBench Apptainer image plus pinned service images.
EOF
}

resolve_root_path() {
  local raw_path="$1"
  if [[ "${raw_path}" = /* ]]; then
    printf '%s\n' "${raw_path}"
    return 0
  fi
  printf '%s\n' "${ROOT_DIR}/${raw_path}"
}

source_module_init() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi

  local candidate=""
  if [[ -n "${MAXIONBENCH_MODULE_INIT_SH:-}" ]]; then
    candidate="$(resolve_root_path "${MAXIONBENCH_MODULE_INIT_SH}")"
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      if command -v module >/dev/null 2>&1; then
        return 0
      fi
    fi
  fi

  for candidate in \
    /etc/profile.d/modules.sh \
    /usr/share/Modules/init/bash \
    /etc/profile.d/lmod.sh \
    /usr/share/lmod/lmod/init/bash
  do
    if [[ ! -f "${candidate}" ]]; then
      continue
    fi
    # shellcheck disable=SC1090
    source "${candidate}"
    if command -v module >/dev/null 2>&1; then
      return 0
    fi
  done

  return 1
}

ensure_apptainer() {
  if command -v apptainer >/dev/null 2>&1; then
    return 0
  fi

  local module_name="${MAXIONBENCH_APPTAINER_MODULE:-apptainer}"
  if source_module_init && command -v module >/dev/null 2>&1; then
    module load "${module_name}"
  fi
  if ! command -v apptainer >/dev/null 2>&1; then
    echo "error: apptainer is required to build container images" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --only-missing)
      ONLY_MISSING=1
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

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "error: --output-dir is required" >&2
  exit 2
fi

ensure_apptainer
mkdir -p "${OUTPUT_DIR}"

should_skip_target() {
  local target="$1"
  [[ "${ONLY_MISSING}" -eq 1 && -f "${target}" ]]
}

apptainer_build_supports_fakeroot() {
  apptainer build --help 2>/dev/null | grep -F -- "--fakeroot" >/dev/null 2>&1
}

build_main_image() {
  local target="$1"
  local definition_file="${ROOT_DIR}/maxionbench.def"
  if [[ ! -f "${definition_file}" ]]; then
    echo "error: missing ${definition_file}" >&2
    exit 2
  fi
  if should_skip_target "${target}"; then
    printf '%s\n' "+ skipping existing ${target}"
    return 0
  fi

  local -a cmd=(apptainer build)
  if [[ -n "${MAXIONBENCH_APPTAINER_BUILD_FLAGS:-}" ]]; then
    local -a extra_flags=()
    # shellcheck disable=SC2206
    extra_flags=(${MAXIONBENCH_APPTAINER_BUILD_FLAGS})
    cmd+=("${extra_flags[@]}")
  elif apptainer_build_supports_fakeroot; then
    cmd+=(--fakeroot)
  fi
  cmd+=("${target}" "${definition_file}")
  printf '%s\n' "+ ${cmd[*]}"
  "${cmd[@]}"
}

pull_service_image() {
  local target="$1"
  local source_uri="$2"
  if should_skip_target "${target}"; then
    printf '%s\n' "+ skipping existing ${target}"
    return 0
  fi
  printf '%s\n' "+ apptainer pull ${target} ${source_uri}"
  apptainer pull "${target}" "${source_uri}"
}

build_main_image "${OUTPUT_DIR}/maxionbench.sif"
pull_service_image "${OUTPUT_DIR}/qdrant.sif" "docker://qdrant/qdrant:v1.13.0"
pull_service_image "${OUTPUT_DIR}/pgvector.sif" "docker://pgvector/pgvector:0.8.0-pg16"
pull_service_image "${OUTPUT_DIR}/opensearch.sif" "docker://opensearchproject/opensearch:2.19.1"
pull_service_image "${OUTPUT_DIR}/weaviate.sif" "docker://semitechnologies/weaviate:1.28.4"
pull_service_image "${OUTPUT_DIR}/milvus.sif" "docker://milvusdb/milvus:v2.5.4"
pull_service_image "${OUTPUT_DIR}/milvus-etcd.sif" "docker://quay.io/coreos/etcd:v3.5.18"
pull_service_image "${OUTPUT_DIR}/milvus-minio.sif" "docker://minio/minio:RELEASE.2024-11-07T00-52-20Z"
