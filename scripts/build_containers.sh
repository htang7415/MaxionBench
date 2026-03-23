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
MAXIONBENCH_PGVECTOR_BIN_DIR="${MAXIONBENCH_PGVECTOR_BIN_DIR:-/usr/lib/postgresql/16/bin}"
SERVICE_CONTRACTS_SH="${ROOT_DIR}/maxionbench/orchestration/slurm/service_contracts.sh"

if [[ ! -f "${SERVICE_CONTRACTS_SH}" ]]; then
  echo "error: missing Apptainer service contract helper: ${SERVICE_CONTRACTS_SH}" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${SERVICE_CONTRACTS_SH}"

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

default_apptainer_cache_dir() {
  if [[ -n "${MAXIONBENCH_APPTAINER_CACHE_DIR:-}" ]]; then
    resolve_root_path "${MAXIONBENCH_APPTAINER_CACHE_DIR}"
    return 0
  fi
  if [[ -n "${APPTAINER_CACHEDIR:-}" ]]; then
    resolve_root_path "${APPTAINER_CACHEDIR}"
    return 0
  fi
  if [[ -n "${MAXIONBENCH_SHARED_ROOT:-}" ]]; then
    printf '%s\n' "${MAXIONBENCH_SHARED_ROOT}/.cache/apptainer"
    return 0
  fi
  local resolved_output_dir=""
  resolved_output_dir="$(resolve_root_path "${OUTPUT_DIR}")"
  printf '%s\n' "${resolved_output_dir%/*}/.cache/apptainer"
}

default_apptainer_tmpdir() {
  if [[ -n "${MAXIONBENCH_APPTAINER_TMPDIR:-}" ]]; then
    resolve_root_path "${MAXIONBENCH_APPTAINER_TMPDIR}"
    return 0
  fi
  if [[ -n "${APPTAINER_TMPDIR:-}" ]]; then
    resolve_root_path "${APPTAINER_TMPDIR}"
    return 0
  fi
  printf '%s\n' "${APPTAINER_CACHEDIR}/tmp"
}

configure_apptainer_storage() {
  export APPTAINER_CACHEDIR
  APPTAINER_CACHEDIR="$(default_apptainer_cache_dir)"
  export APPTAINER_TMPDIR
  APPTAINER_TMPDIR="$(default_apptainer_tmpdir)"
  mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
  printf '%s\n' "+ APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}"
  printf '%s\n' "+ APPTAINER_TMPDIR=${APPTAINER_TMPDIR}"
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
configure_apptainer_storage
mkdir -p "${OUTPUT_DIR}"

should_skip_target() {
  local target="$1"
  [[ "${ONLY_MISSING}" -eq 1 && -f "${target}" ]]
}

apptainer_build_supports_fakeroot() {
  apptainer build --help 2>/dev/null | grep -F -- "--fakeroot" >/dev/null 2>&1
}

default_pgvector_path() {
  printf '%s\n' "${MAXIONBENCH_PGVECTOR_BIN_DIR}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
}

service_probe_is_valid() {
  local target="$1"
  local service_name="$2"
  local -a probe_args=()
  mb_service_probe_args "${service_name}" probe_args || return 1
  apptainer inspect "${target}" >/dev/null 2>&1 && \
    apptainer exec --cleanenv "${target}" "${probe_args[@]}" >/dev/null 2>&1
}

service_layout_is_valid() {
  local target="$1"
  local service_name="$2"

  case "${service_name}" in
    qdrant)
      apptainer inspect "${target}" >/dev/null 2>&1 && \
        apptainer exec --cleanenv "${target}" /bin/sh -c \
          '[ -x /qdrant/entrypoint.sh ] && [ -x /qdrant/qdrant ]'
      ;;
    opensearch)
      apptainer inspect "${target}" >/dev/null 2>&1 && \
        apptainer exec --cleanenv "${target}" /bin/sh -c \
          '[ -x /usr/share/opensearch/bin/opensearch ] && [ -x /usr/share/opensearch/opensearch-docker-entrypoint.sh ] && [ -x /usr/share/opensearch/jdk/bin/java ]'
      ;;
    *)
      return 1
      ;;
  esac
}

main_image_is_valid() {
  local target="$1"
  if [[ ! -f "${target}" ]]; then
    return 1
  fi
  apptainer exec --cleanenv "${target}" python -s - <<'PY'
import sys

modules = (
    "docker",
    "faiss",
    "h5py",
    "jinja2",
    "maxionbench.cli",
    "numpy",
    "pandas",
    "psutil",
    "pytz",
    "scipy",
    "sklearn",
    "torch",
    "transformers",
)

for module_name in modules:
    try:
        __import__(module_name)
    except Exception as exc:
        print(f"main image validation failed while importing {module_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

import faiss

if not hasattr(faiss, "StandardGpuResources"):
    raise RuntimeError("main image validation failed: installed faiss module does not expose GPU support")
PY
}

prefetch_s5_reranker_cache() {
  local image_path="$1"
  local hf_cache_dir="${MAXIONBENCH_HF_CACHE_DIR:-}"
  local model_id="BAAI/bge-reranker-base"
  local revision="2026-03-04"
  local resolved_hf_cache=""

  if [[ -z "${hf_cache_dir}" ]]; then
    printf '%s\n' "+ skipping shared HF cache prefetch; MAXIONBENCH_HF_CACHE_DIR is empty"
    return 0
  fi

  resolved_hf_cache="$(resolve_root_path "${hf_cache_dir}")"
  mkdir -p "${resolved_hf_cache}/hub" "${resolved_hf_cache}/transformers"
  printf '%s\n' "+ prefetching ${model_id}@${revision} into ${resolved_hf_cache}"
  apptainer exec --cleanenv \
    --env "HF_HOME=${resolved_hf_cache}" \
    --env "HUGGINGFACE_HUB_CACHE=${resolved_hf_cache}/hub" \
    --env "TRANSFORMERS_CACHE=${resolved_hf_cache}/transformers" \
    "${image_path}" python -s - <<'PY'
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "BAAI/bge-reranker-base"
REVISION = "2026-03-04"

AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=REVISION)
PY
  printf '%s\n' "+ validated shared HF cache for ${model_id}@${revision}"
}

service_image_is_valid() {
  local target="$1"
  local service_name="$2"
  local contract_kind=""
  if [[ ! -f "${target}" ]]; then
    return 1
  fi
  contract_kind="$(mb_service_contract_kind "${service_name}")" || {
    echo "error: unsupported service image validation target: ${service_name}" >&2
    exit 2
  }
  case "${contract_kind}" in
    qdrant-layout|opensearch-layout)
      service_layout_is_valid "${target}" "${service_name}"
      ;;
    pgvector)
      local pgvector_path=""
      pgvector_path="$(default_pgvector_path)"
      apptainer inspect "${target}" >/dev/null 2>&1 && \
        apptainer exec --cleanenv --env "PATH=${pgvector_path}" "${target}" /bin/sh -c \
        'command -v docker-entrypoint.sh >/dev/null 2>&1 && command -v postgres >/dev/null 2>&1 && command -v initdb >/dev/null 2>&1'
      ;;
    probe)
      service_probe_is_valid "${target}" "${service_name}"
      ;;
    *)
      echo "error: unsupported service image validation contract ${contract_kind} for ${service_name}" >&2
      exit 2
      ;;
  esac
}

build_main_image() {
  local target="$1"
  local definition_file="${ROOT_DIR}/maxionbench.def"
  if [[ ! -f "${definition_file}" ]]; then
    echo "error: missing ${definition_file}" >&2
    exit 2
  fi
  if should_skip_target "${target}" && main_image_is_valid "${target}"; then
    printf '%s\n' "+ skipping existing ${target}"
    return 0
  fi
  if [[ "${ONLY_MISSING}" -eq 1 && -f "${target}" ]]; then
    printf '%s\n' "+ rebuilding stale ${target}; required runtime imports are missing"
    rm -f "${target}"
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
  if ! main_image_is_valid "${target}"; then
    echo "error: built ${target} is missing required runtime imports for MaxionBench and D3 dataset preparation" >&2
    exit 2
  fi
  printf '%s\n' "+ validated ${target} contains required runtime imports"
}

pull_service_image() {
  local target="$1"
  local source_uri="$2"
  local service_name="$3"
  if [[ "${ONLY_MISSING}" -eq 1 && -f "${target}" ]]; then
    if service_image_is_valid "${target}" "${service_name}"; then
      printf '%s\n' "+ skipping existing ${target}"
      return 0
    fi
    printf '%s\n' "+ rebuilding stale ${target}; required runtime contract is missing for ${service_name}"
    rm -f "${target}"
  fi
  local stderr_log=""
  local status=0
  stderr_log="$(mktemp "${TMPDIR:-/tmp}/maxionbench_apptainer_pull.XXXXXX")"
  printf '%s\n' "+ apptainer pull ${target} ${source_uri}"
  set +e
  apptainer pull "${target}" "${source_uri}" 2>"${stderr_log}"
  status=$?
  set -e
  while IFS= read -r line; do
    if [[ "${line}" == *'ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"'* ]]; then
      continue
    fi
    printf '%s\n' "${line}" >&2
  done < "${stderr_log}"
  rm -f "${stderr_log}"
  if [[ "${status}" -ne 0 ]]; then
    return "${status}"
  fi
  if ! service_image_is_valid "${target}" "${service_name}"; then
    echo "error: pulled ${target} failed ${service_name} runtime contract validation" >&2
    exit 2
  fi
  printf '%s\n' "+ validated ${target} contains required runtime contract for ${service_name}"
}

build_main_image "${OUTPUT_DIR}/maxionbench.sif"
prefetch_s5_reranker_cache "${OUTPUT_DIR}/maxionbench.sif"
pull_service_image "${OUTPUT_DIR}/qdrant.sif" "docker://qdrant/qdrant:v1.13.0" "qdrant"
pull_service_image "${OUTPUT_DIR}/pgvector.sif" "docker://pgvector/pgvector:0.8.0-pg16" "pgvector"
pull_service_image "${OUTPUT_DIR}/opensearch.sif" "docker://opensearchproject/opensearch:2.19.1" "opensearch"
pull_service_image "${OUTPUT_DIR}/weaviate.sif" "docker://semitechnologies/weaviate:1.28.4" "weaviate"
pull_service_image "${OUTPUT_DIR}/milvus.sif" "docker://milvusdb/milvus:v2.5.27" "milvus"
pull_service_image "${OUTPUT_DIR}/milvus-etcd.sif" "docker://quay.io/coreos/etcd:v3.5.18" "milvus-etcd"
pull_service_image "${OUTPUT_DIR}/milvus-minio.sif" "docker://minio/minio:RELEASE.2024-11-07T00-52-20Z" "milvus-minio"
