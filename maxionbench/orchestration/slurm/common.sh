#!/usr/bin/env bash
set -euo pipefail

# Shared Slurm helpers for MaxionBench jobs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVICE_CONTRACTS_SH="${ROOT_DIR}/maxionbench/orchestration/slurm/service_contracts.sh"
export PYTHONUNBUFFERED=1
export MAXIONBENCH_SCRATCH_SAFETY_FACTOR="${MAXIONBENCH_SCRATCH_SAFETY_FACTOR:-1.8}"
export MAXIONBENCH_SKIP_PRE_RUN_GATE="${MAXIONBENCH_SKIP_PRE_RUN_GATE:-0}"
export MAXIONBENCH_ALLOW_GPU_UNAVAILABLE="${MAXIONBENCH_ALLOW_GPU_UNAVAILABLE:-0}"
export MAXIONBENCH_CONFORMANCE_MATRIX="${MAXIONBENCH_CONFORMANCE_MATRIX:-${ROOT_DIR}/artifacts/conformance/conformance_matrix.csv}"
export MAXIONBENCH_OUTPUT_ROOT="${MAXIONBENCH_OUTPUT_ROOT:-artifacts/runs/slurm}"
export MAXIONBENCH_DATASET_ROOT="${MAXIONBENCH_DATASET_ROOT:-dataset}"
export MAXIONBENCH_DATASET_CACHE_DIR="${MAXIONBENCH_DATASET_CACHE_DIR:-.cache}"
export MAXIONBENCH_FIGURES_ROOT="${MAXIONBENCH_FIGURES_ROOT:-artifacts/figures}"
export MAXIONBENCH_CONTAINER_RUNTIME="${MAXIONBENCH_CONTAINER_RUNTIME:-}"
export MAXIONBENCH_CONTAINER_IMAGE="${MAXIONBENCH_CONTAINER_IMAGE:-}"
export MAXIONBENCH_CONTAINER_BIND="${MAXIONBENCH_CONTAINER_BIND:-}"
export MAXIONBENCH_HF_CACHE_DIR="${MAXIONBENCH_HF_CACHE_DIR:-}"
export MAXIONBENCH_APPTAINER_MODULE="${MAXIONBENCH_APPTAINER_MODULE:-apptainer}"
export MAXIONBENCH_MODULE_INIT_SH="${MAXIONBENCH_MODULE_INIT_SH:-}"
export MAXIONBENCH_APPTAINER_RUNTIME_LOGGED="${MAXIONBENCH_APPTAINER_RUNTIME_LOGGED:-0}"
export MAXIONBENCH_DATASET_ENV_SH="${MAXIONBENCH_DATASET_ENV_SH:-${ROOT_DIR}/artifacts/prefetch/dataset_env.sh}"
export MAXIONBENCH_D3_PARAMS_PATH="${MAXIONBENCH_D3_PARAMS_PATH:-}"
export MAXIONBENCH_SLURM_RUN_MANIFEST="${MAXIONBENCH_SLURM_RUN_MANIFEST:-}"
export MAXIONBENCH_ENGINE_WAIT_TIMEOUT_S="${MAXIONBENCH_ENGINE_WAIT_TIMEOUT_S:-300}"
export MAXIONBENCH_CONFORMANCE_TIMEOUT_S="${MAXIONBENCH_CONFORMANCE_TIMEOUT_S:-300}"
export MAXIONBENCH_SERVICE_START_GRACE_S="${MAXIONBENCH_SERVICE_START_GRACE_S:-5}"
export MAXIONBENCH_SERVICE_START_POLL_S="${MAXIONBENCH_SERVICE_START_POLL_S:-0.25}"
export MAXIONBENCH_SERVICE_LOG_TAIL_LINES="${MAXIONBENCH_SERVICE_LOG_TAIL_LINES:-80}"
export MAXIONBENCH_PGVECTOR_BIN_DIR="${MAXIONBENCH_PGVECTOR_BIN_DIR:-/usr/lib/postgresql/16/bin}"
export MAXIONBENCH_QDRANT_IMAGE="${MAXIONBENCH_QDRANT_IMAGE:-}"
export MAXIONBENCH_PGVECTOR_IMAGE="${MAXIONBENCH_PGVECTOR_IMAGE:-}"
export MAXIONBENCH_OPENSEARCH_IMAGE="${MAXIONBENCH_OPENSEARCH_IMAGE:-}"
export MAXIONBENCH_WEAVIATE_IMAGE="${MAXIONBENCH_WEAVIATE_IMAGE:-}"
export MAXIONBENCH_MILVUS_ETCD_IMAGE="${MAXIONBENCH_MILVUS_ETCD_IMAGE:-}"
export MAXIONBENCH_MILVUS_MINIO_IMAGE="${MAXIONBENCH_MILVUS_MINIO_IMAGE:-}"
export MAXIONBENCH_MILVUS_IMAGE="${MAXIONBENCH_MILVUS_IMAGE:-}"
export MAXIONBENCH_CLEANUP_LOCAL_SCRATCH="${MAXIONBENCH_CLEANUP_LOCAL_SCRATCH:-1}"

if [[ ! -f "${SERVICE_CONTRACTS_SH}" ]]; then
  echo "error: missing Apptainer service contract helper: ${SERVICE_CONTRACTS_SH}" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${SERVICE_CONTRACTS_SH}"

mb_log() {
  echo "[maxionbench][$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

mb_log_stderr() {
  echo "[maxionbench][$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2
}

mb_die() {
  mb_log "ERROR: $*"
  exit 1
}

mb_flag_enabled() {
  local raw="${1:-}"
  case "$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

mb_require_tmpdir() {
  if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    export SLURM_TMPDIR="/tmp"
    mb_log "SLURM_TMPDIR is not set; falling back to /tmp"
  fi
  mkdir -p "${SLURM_TMPDIR}"
}

mb_resolve_config() {
  local config_rel="$1"
  if [[ "${config_rel}" = /* ]]; then
    echo "${config_rel}"
  else
    echo "${ROOT_DIR}/${config_rel}"
  fi
}

mb_resolve_host_path() {
  local raw_path="$1"
  if [[ -z "${raw_path}" ]]; then
    echo ""
  elif [[ "${raw_path}" = /* ]]; then
    echo "${raw_path}"
  else
    echo "${ROOT_DIR}/${raw_path}"
  fi
}

mb_source_dataset_env() {
  local dataset_env="${MAXIONBENCH_DATASET_ENV_SH:-}"
  if [[ -z "${dataset_env}" ]]; then
    return
  fi
  local resolved_env
  resolved_env="$(mb_resolve_host_path "${dataset_env}")"
  if [[ -f "${resolved_env}" ]]; then
    # shellcheck disable=SC1090
    source "${resolved_env}"
    mb_log "loaded dataset env: ${resolved_env}"
  fi
}

mb_require_gpu_fail_fast() {
  if [[ "${MAXIONBENCH_ALLOW_GPU_UNAVAILABLE:-0}" == "1" ]]; then
    mb_die "MAXIONBENCH_ALLOW_GPU_UNAVAILABLE=1 is not allowed for this GPU-required Slurm rerun"
  fi
}

mb_require_visible_gpu() {
  local gpu_count
  gpu_count="$(mb_python - <<'PY'
from maxionbench.runtime.system_info import collect_system_info

try:
    payload = collect_system_info()
except Exception:
    payload = {}
print(int(payload.get("gpu_count", 0) or 0))
PY
)"
  if [[ "${gpu_count}" -lt 1 ]]; then
    mb_die "at least one visible GPU is required for this job"
  fi
  mb_log "visible GPUs=${gpu_count}"
}

mb_require_dataset_env_contract() {
  local config_path="$1"
  local resolved
  resolved="$(mb_resolve_config "${config_path}")"
  local dataset_env="${MAXIONBENCH_DATASET_ENV_SH:-}"
  local resolved_dataset_env=""
  if [[ -n "${dataset_env}" ]]; then
    resolved_dataset_env="$(mb_resolve_host_path "${dataset_env}")"
  fi
  local summary
  summary="$(mb_python - <<'PY' "${resolved}" "${resolved_dataset_env}"
from pathlib import Path
import json
import shlex
import sys

import yaml

cfg_path = Path(sys.argv[1]).resolve()
env_path_raw = sys.argv[2]
env_path = Path(env_path_raw).resolve() if env_path_raw else None
payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
if not isinstance(payload, dict):
    raise ValueError(f"Expected YAML mapping config: {cfg_path}")
dataset_bundle = str(payload.get("dataset_bundle", "")).upper()
need_d3 = bool(dataset_bundle == "D3" and (payload.get("dataset_path") or bool(payload.get("calibration_require_real_data", False))))
need_d4_beir = bool(dataset_bundle == "D4" and bool(payload.get("d4_use_real_data", False)) and payload.get("d4_beir_root"))
need_d4_crag = bool(
    dataset_bundle == "D4"
    and bool(payload.get("d4_use_real_data", False))
    and bool(payload.get("d4_include_crag", False))
    and payload.get("d4_crag_path")
)
required = need_d3 or need_d4_beir or need_d4_crag
if not required:
    print(json.dumps({"required": False}))
    raise SystemExit(0)
if env_path is None or not env_path.exists():
    raise FileNotFoundError(f"required dataset env export file is missing: {env_path}")

exports = {}
for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line.startswith("export "):
        continue
    assignment = line[len("export ") :]
    name, sep, value = assignment.partition("=")
    if sep != "=":
        continue
    tokens = shlex.split(value, posix=True)
    exports[name.strip()] = tokens[0] if tokens else ""

required_exports = []
if need_d3:
    required_exports.extend(["MAXIONBENCH_D3_DATASET_PATH", "MAXIONBENCH_D3_DATASET_SHA256"])
if need_d4_beir:
    required_exports.append("MAXIONBENCH_D4_BEIR_ROOT")
if need_d4_crag:
    required_exports.extend(["MAXIONBENCH_D4_CRAG_PATH", "MAXIONBENCH_D4_CRAG_SHA256"])

missing = [name for name in required_exports if not str(exports.get(name, "")).strip()]
if missing:
    raise ValueError(
        f"dataset env `{env_path}` is missing required export(s) for {cfg_path.name}: {', '.join(sorted(missing))}"
    )

print(
    json.dumps(
        {
            "required": True,
            "config_path": str(cfg_path),
            "env_sh_path": str(env_path),
            "required_exports": sorted(required_exports),
        },
        sort_keys=True,
    )
)
PY
)"
  if [[ "${summary}" != '{"required": false}' ]]; then
    mb_log "validated dataset env contract ${summary}"
  fi
}

mb_apptainer_use_nv() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

mb_log_apptainer_runtime_once() {
  if [[ "${MAXIONBENCH_APPTAINER_RUNTIME_LOGGED:-0}" == "1" ]]; then
    return 0
  fi
  local apptainer_path=""
  apptainer_path="$(command -v apptainer || true)"
  if [[ -z "${apptainer_path}" ]]; then
    return 1
  fi
  mb_log_stderr "using apptainer binary ${apptainer_path}"
  export MAXIONBENCH_APPTAINER_RUNTIME_LOGGED=1
}

mb_source_module_init() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi

  local candidate=""
  if [[ -n "${MAXIONBENCH_MODULE_INIT_SH:-}" ]]; then
    candidate="$(mb_resolve_host_path "${MAXIONBENCH_MODULE_INIT_SH}")"
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      mb_log_stderr "sourced module init ${candidate}"
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
    mb_log_stderr "sourced module init ${candidate}"
    if command -v module >/dev/null 2>&1; then
      return 0
    fi
  done
  mb_log_stderr "module command is unavailable; apptainer bootstrap could not source a module init script"
  return 1
}

mb_ensure_apptainer() {
  if command -v apptainer >/dev/null 2>&1; then
    mb_log_apptainer_runtime_once
    return 0
  fi

  local module_name="${MAXIONBENCH_APPTAINER_MODULE:-apptainer}"
  mb_log_stderr "apptainer not found in PATH; attempting module bootstrap with ${module_name}"
  if [[ -n "${module_name}" ]] && mb_source_module_init && command -v module >/dev/null 2>&1; then
    mb_log_stderr "loading apptainer module ${module_name}"
    local module_output=""
    local module_status=0
    local module_log_file=""
    module_log_file="$(mktemp "${SLURM_TMPDIR:-/tmp}/maxionbench_apptainer_module.XXXXXX")"
    set +e
    module load "${module_name}" >"${module_log_file}" 2>&1
    module_status=$?
    set -e
    if [[ -f "${module_log_file}" ]]; then
      module_output="$(cat "${module_log_file}")"
      rm -f "${module_log_file}"
    fi
    if [[ -n "${module_output}" ]]; then
      mb_log_stderr "module load ${module_name} output: ${module_output}"
    fi
    if [[ ${module_status} -eq 0 ]] && command -v apptainer >/dev/null 2>&1; then
      mb_log_apptainer_runtime_once
      return 0
    fi
    mb_log_stderr "module load ${module_name} did not make apptainer available (status=${module_status})"
  fi
  return 1
}

mb_container_bind_specs() {
  printf '%s\n' "${ROOT_DIR}:${ROOT_DIR}"
  if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    printf '%s\n' "${SLURM_TMPDIR}:${SLURM_TMPDIR}"
  fi

  local resolved_matrix
  resolved_matrix="$(mb_resolve_host_path "${MAXIONBENCH_CONFORMANCE_MATRIX:-}")"
  if [[ -n "${resolved_matrix}" ]]; then
    local matrix_dir
    matrix_dir="$(dirname "${resolved_matrix}")"
    if [[ -d "${matrix_dir}" ]]; then
      printf '%s\n' "${matrix_dir}:${matrix_dir}"
    fi
  fi

  local resolved_output_root
  resolved_output_root="$(mb_resolve_host_path "${MAXIONBENCH_OUTPUT_ROOT:-}")"
  if [[ -n "${resolved_output_root}" ]]; then
    mkdir -p "${resolved_output_root}"
    printf '%s\n' "${resolved_output_root}:${resolved_output_root}"
  fi

  local resolved_dataset_root
  resolved_dataset_root="$(mb_resolve_host_path "${MAXIONBENCH_DATASET_ROOT:-}")"
  if [[ -n "${resolved_dataset_root}" ]]; then
    mkdir -p "${resolved_dataset_root}"
    printf '%s\n' "${resolved_dataset_root}:${resolved_dataset_root}"
  fi

  local resolved_dataset_cache
  resolved_dataset_cache="$(mb_resolve_host_path "${MAXIONBENCH_DATASET_CACHE_DIR:-}")"
  if [[ -n "${resolved_dataset_cache}" ]]; then
    mkdir -p "${resolved_dataset_cache}"
    printf '%s\n' "${resolved_dataset_cache}:${resolved_dataset_cache}"
  fi

  local resolved_figures_root
  resolved_figures_root="$(mb_resolve_host_path "${MAXIONBENCH_FIGURES_ROOT:-}")"
  if [[ -n "${resolved_figures_root}" ]]; then
    mkdir -p "${resolved_figures_root}"
    printf '%s\n' "${resolved_figures_root}:${resolved_figures_root}"
  fi

  if [[ -n "${MAXIONBENCH_HF_CACHE_DIR:-}" ]]; then
    local resolved_hf_cache
    resolved_hf_cache="$(mb_resolve_host_path "${MAXIONBENCH_HF_CACHE_DIR}")"
    mkdir -p "${resolved_hf_cache}" "${resolved_hf_cache}/hub" "${resolved_hf_cache}/transformers"
    printf '%s\n' "${resolved_hf_cache}:${resolved_hf_cache}"
  fi

  if [[ -n "${MAXIONBENCH_CONTAINER_BIND:-}" ]]; then
    local -a extra_binds=()
    local bind_spec=""
    IFS='|' read -r -a extra_binds <<< "${MAXIONBENCH_CONTAINER_BIND}"
    for bind_spec in "${extra_binds[@]}"; do
      if [[ -n "${bind_spec}" ]]; then
        printf '%s\n' "${bind_spec}"
      fi
    done
  fi

  if [[ -n "${MAXIONBENCH_D3_PARAMS_PATH:-}" ]]; then
    local resolved_d3_params
    resolved_d3_params="$(mb_resolve_host_path "${MAXIONBENCH_D3_PARAMS_PATH}")"
    local d3_params_parent
    d3_params_parent="$(dirname "${resolved_d3_params}")"
    mkdir -p "${d3_params_parent}"
    printf '%s\n' "${d3_params_parent}:${d3_params_parent}"
  fi

  if [[ -n "${MAXIONBENCH_SLURM_RUN_MANIFEST:-}" ]]; then
    local resolved_run_manifest
    resolved_run_manifest="$(mb_resolve_host_path "${MAXIONBENCH_SLURM_RUN_MANIFEST}")"
    local run_manifest_parent
    run_manifest_parent="$(dirname "${resolved_run_manifest}")"
    mkdir -p "${run_manifest_parent}"
    printf '%s\n' "${run_manifest_parent}:${run_manifest_parent}"
  fi
}

mb_python() {
  if [[ -z "${MAXIONBENCH_CONTAINER_RUNTIME:-}" ]]; then
    python "$@"
    return
  fi

  case "${MAXIONBENCH_CONTAINER_RUNTIME}" in
    apptainer)
      if ! mb_ensure_apptainer; then
        mb_die "MAXIONBENCH_CONTAINER_RUNTIME=apptainer requires apptainer in PATH or a loadable ${MAXIONBENCH_APPTAINER_MODULE} module"
      fi
      if [[ -z "${MAXIONBENCH_CONTAINER_IMAGE:-}" ]]; then
        mb_die "MAXIONBENCH_CONTAINER_IMAGE must be set when MAXIONBENCH_CONTAINER_RUNTIME=apptainer"
      fi
      local resolved_image
      resolved_image="$(mb_resolve_host_path "${MAXIONBENCH_CONTAINER_IMAGE}")"
      if [[ ! -f "${resolved_image}" ]]; then
        mb_die "apptainer image not found: ${resolved_image}"
      fi

      local -a container_cmd=(apptainer exec --cleanenv)
      if mb_apptainer_use_nv; then
        container_cmd+=(--nv)
      fi
      local bind_spec=""
      while IFS= read -r bind_spec; do
        if [[ -n "${bind_spec}" ]]; then
          container_cmd+=(--bind "${bind_spec}")
        fi
      done < <(mb_container_bind_specs)
      container_cmd+=("${resolved_image}" env "PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}" "PYTHONNOUSERSITE=1")
      local env_name=""
      while IFS='=' read -r env_name _; do
        case "${env_name}" in
          MAXIONBENCH_*)
            container_cmd+=("${env_name}=${!env_name}")
            ;;
        esac
      done < <(env)

      if [[ -n "${MAXIONBENCH_HF_CACHE_DIR:-}" ]]; then
        local resolved_hf_cache
        resolved_hf_cache="$(mb_resolve_host_path "${MAXIONBENCH_HF_CACHE_DIR}")"
        container_cmd+=(
          "HF_HOME=${resolved_hf_cache}"
          "HUGGINGFACE_HUB_CACHE=${resolved_hf_cache}/hub"
          "TRANSFORMERS_CACHE=${resolved_hf_cache}/transformers"
        )
      fi
      container_cmd+=(python -s "$@")
      "${container_cmd[@]}"
      ;;
    *)
      mb_die "unsupported MAXIONBENCH_CONTAINER_RUNTIME=${MAXIONBENCH_CONTAINER_RUNTIME}; supported: apptainer"
      ;;
  esac
}

mb_run_scratch_preflight() {
  local resolved="$1"
  local summary
  local status

  set +e
  summary="$(mb_python -m maxionbench.orchestration.slurm.preflight \
    --config "${resolved}" \
    --tmpdir "${SLURM_TMPDIR}" \
    --safety-factor "${MAXIONBENCH_SCRATCH_SAFETY_FACTOR}")"
  status=$?
  set -e

  export MB_PREFLIGHT_SUMMARY="${summary}"
  mb_log "preflight=${summary}"

  return "${status}"
}

mb_scratch_preflight() {
  local config_path="$1"
  local resolved
  resolved="$(mb_resolve_config "${config_path}")"

  if mb_run_scratch_preflight "${resolved}"; then
    export MB_PREFLIGHT_CONFIG="${resolved}"
    return 0
  fi

  local fallback
  fallback="$(python - <<'PY'
import json, os
summary = json.loads(os.environ.get("MB_PREFLIGHT_SUMMARY", "{}") or "{}")
value = summary.get("fallback_config")
print(value or "")
PY
)"
  if [[ -n "${fallback}" ]]; then
    local resolved_fallback
    resolved_fallback="$(mb_resolve_config "${fallback}")"
    if [[ -f "${resolved_fallback}" ]]; then
      mb_log "scratch preflight failed, validating fallback config ${resolved_fallback}"
      if mb_run_scratch_preflight "${resolved_fallback}"; then
        export MB_PREFLIGHT_CONFIG="${resolved_fallback}"
        return 0
      fi
      mb_log "fallback config ${resolved_fallback} also failed scratch preflight"
    fi
  fi

  mb_die "scratch preflight failed and no valid fallback config was available"
}

mb_allocate_ports() {
  local payload
  payload="$(mb_python - <<'PY'
from maxionbench.runtime.ports import allocate_named_ports
ports = allocate_named_ports(
    [
        "qdrant",
        "qdrant_grpc",
        "weaviate",
        "opensearch",
        "opensearch_transport",
        "lancedb",
        "milvus",
        "milvus_metrics",
        "milvus_etcd",
        "milvus_minio",
        "milvus_minio_console",
        "pgvector",
    ],
    base=20000,
    span=20000,
)
for name, port in sorted(ports.items()):
    print(f"{name}={port}")
PY
)"
  while IFS='=' read -r name value; do
    if [[ -n "${name}" && -n "${value}" ]]; then
      upper="$(echo "${name}" | tr '[:lower:]' '[:upper:]')"
      export "MAXIONBENCH_PORT_${upper}=${value}"
    fi
  done <<<"${payload}"

  export MAXIONBENCH_QDRANT_HOST="127.0.0.1"
  export MAXIONBENCH_QDRANT_PORT="${MAXIONBENCH_PORT_QDRANT}"
  export MAXIONBENCH_MILVUS_HOST="127.0.0.1"
  export MAXIONBENCH_MILVUS_PORT="${MAXIONBENCH_PORT_MILVUS}"
  export MAXIONBENCH_OPENSEARCH_HOST="127.0.0.1"
  export MAXIONBENCH_OPENSEARCH_PORT="${MAXIONBENCH_PORT_OPENSEARCH}"
  export MAXIONBENCH_OPENSEARCH_SCHEME="http"
  export MAXIONBENCH_WEAVIATE_HOST="127.0.0.1"
  export MAXIONBENCH_WEAVIATE_PORT="${MAXIONBENCH_PORT_WEAVIATE}"
  export MAXIONBENCH_WEAVIATE_SCHEME="http"
  export MAXIONBENCH_PGVECTOR_PORT="${MAXIONBENCH_PORT_PGVECTOR}"
  export MAXIONBENCH_PGVECTOR_DSN="postgresql://postgres:postgres@127.0.0.1:${MAXIONBENCH_PORT_PGVECTOR}/postgres"
  export MAXIONBENCH_LANCEDB_SERVICE_URL="http://127.0.0.1:${MAXIONBENCH_PORT_LANCEDB}"
  if [[ -z "${MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI:-}" ]]; then
    export MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI="${SLURM_TMPDIR}/lancedb/service"
  fi
  mkdir -p "${MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI}"
}

mb_engine_runtime_root() {
  local job_id="${SLURM_JOB_ID:-local}"
  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  echo "${SLURM_TMPDIR}/maxionbench_engine_runtime/${job_id}_${task_id}"
}

mb_resolve_apptainer_image() {
  local raw_path="$1"
  if [[ -z "${raw_path}" ]]; then
    echo ""
    return 0
  fi
  mb_resolve_host_path "${raw_path}"
}

mb_require_apptainer_service_image() {
  local env_name="$1"
  local label="$2"
  local raw_path="${!env_name:-}"
  if [[ -z "${raw_path}" ]]; then
    mb_die "${env_name} must be set to a prebuilt Apptainer image for ${label}"
  fi
  local resolved
  resolved="$(mb_resolve_apptainer_image "${raw_path}")"
  if [[ ! -f "${resolved}" ]]; then
    mb_die "${label} Apptainer image not found: ${resolved}"
  fi
  echo "${resolved}"
}

mb_quote_command_args() {
  local rendered=""
  local arg=""
  for arg in "$@"; do
    if [[ -n "${rendered}" ]]; then
      rendered="${rendered} "
    fi
    rendered="${rendered}$(printf '%q' "${arg}")"
  done
  printf '%s\n' "${rendered}"
}

mb_validate_apptainer_service_inspect() {
  local image_path="$1"
  local label="$2"
  if ! mb_ensure_apptainer; then
    mb_die "Apptainer is required to validate ${label} image ${image_path}"
  fi
  if ! apptainer inspect "${image_path}" >/dev/null 2>&1; then
    mb_die "${label} Apptainer image failed inspect: ${image_path}"
  fi
}

mb_validate_apptainer_service_probe() {
  local image_path="$1"
  local label="$2"
  local probe_array_name="$3"
  local -a probe_args_ref=()
  eval "probe_args_ref=(\"\${${probe_array_name}[@]-}\")"

  if [[ "${#probe_args_ref[@]}" -eq 0 ]]; then
    mb_die "missing Apptainer probe command for ${label}"
  fi

  mb_validate_apptainer_service_inspect "${image_path}" "${label}"
  if ! apptainer exec --cleanenv "${image_path}" "${probe_args_ref[@]}" >/dev/null 2>&1; then
    local rendered_probe=""
    rendered_probe="$(mb_quote_command_args "${probe_args_ref[@]}")"
    mb_die "${label} Apptainer image probe command unavailable or image contract mismatch (probe: ${rendered_probe}): ${image_path}"
  fi
}

mb_validate_qdrant_service_image() {
  local image_path="$1"
  mb_validate_apptainer_service_inspect "${image_path}" "qdrant"
  if ! apptainer exec --cleanenv "${image_path}" /bin/sh -c \
    '[ -x /qdrant/entrypoint.sh ] && [ -x /qdrant/qdrant ]' >/dev/null 2>&1; then
    mb_die "qdrant Apptainer image does not expose expected /qdrant/entrypoint.sh and /qdrant/qdrant: ${image_path}"
  fi
}

mb_pgvector_path_env() {
  printf '%s\n' "${MAXIONBENCH_PGVECTOR_BIN_DIR}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
}

mb_validate_pgvector_service_image() {
  local image_path="$1"
  local pgvector_path
  pgvector_path="$(mb_pgvector_path_env)"
  mb_validate_apptainer_service_inspect "${image_path}" "pgvector"
  if ! apptainer exec --cleanenv "${image_path}" /bin/sh -c \
    'command -v docker-entrypoint.sh >/dev/null 2>&1' >/dev/null 2>&1; then
    mb_die "pgvector Apptainer image does not expose required entrypoint `docker-entrypoint.sh`: ${image_path}"
  fi
  if ! apptainer exec --cleanenv "${image_path}" env "PATH=${pgvector_path}" /bin/sh -c \
    'command -v postgres >/dev/null 2>&1 && command -v initdb >/dev/null 2>&1' >/dev/null 2>&1; then
    mb_die "pgvector Apptainer image does not expose required PostgreSQL binaries `postgres` and `initdb` under preserved PATH ${pgvector_path}: ${image_path}"
  fi
}

mb_validate_opensearch_service_image() {
  local image_path="$1"
  mb_validate_apptainer_service_inspect "${image_path}" "opensearch"
  if ! apptainer exec --cleanenv "${image_path}" /bin/sh -c \
    '[ -x /usr/share/opensearch/bin/opensearch ] && [ -x /usr/share/opensearch/opensearch-docker-entrypoint.sh ] && [ -x /usr/share/opensearch/jdk/bin/java ]' >/dev/null 2>&1; then
    mb_die "opensearch Apptainer image does not expose required /usr/share/opensearch/bin/opensearch, /usr/share/opensearch/opensearch-docker-entrypoint.sh, and /usr/share/opensearch/jdk/bin/java: ${image_path}"
  fi
}

mb_validate_named_service_image() {
  local image_path="$1"
  local service_name="$2"
  local contract_kind=""
  contract_kind="$(mb_service_contract_kind "${service_name}")" || mb_die "unsupported Apptainer service contract target: ${service_name}"

  case "${contract_kind}" in
    qdrant-layout)
      mb_validate_qdrant_service_image "${image_path}"
      ;;
    pgvector)
      mb_validate_pgvector_service_image "${image_path}"
      ;;
    opensearch-layout)
      mb_validate_opensearch_service_image "${image_path}"
      ;;
    probe)
      local -a probe_args=()
      mb_service_probe_args "${service_name}" probe_args || mb_die "missing Apptainer probe command for ${service_name}"
      mb_validate_apptainer_service_probe "${image_path}" "${service_name}" probe_args
      ;;
    *)
      mb_die "unsupported Apptainer service contract kind ${contract_kind} for ${service_name}"
      ;;
  esac
}

mb_log_file_tail() {
  local label="$1"
  local file_path="$2"
  local tail_lines="${MAXIONBENCH_SERVICE_LOG_TAIL_LINES:-80}"
  if [[ ! -f "${file_path}" ]]; then
    mb_log "no log file available for ${label}: ${file_path}"
    return 0
  fi
  mb_log "tail ${tail_lines} from ${label} log ${file_path}:"
  tail -n "${tail_lines}" "${file_path}" || true
}

mb_detect_engine_runtime_mode() {
  local config_path="$1"
  local resolved
  resolved="$(mb_resolve_config "${config_path}")"
  mb_python - <<'PY' "${resolved}"
from pathlib import Path
import sys

from maxionbench.orchestration.config_schema import load_run_config

cfg = load_run_config(Path(sys.argv[1]).resolve())
engine = str(cfg.engine).strip()
mode = "embedded"
if engine == "qdrant" and cfg.adapter_options.get("location"):
    mode = "embedded"
elif engine == "lancedb-service":
    mode = "embedded" if cfg.adapter_options.get("inproc_uri") else "service"
elif engine in {"qdrant", "milvus", "opensearch", "pgvector", "weaviate"}:
    mode = "service"
print(f"{engine}|{mode}")
PY
}

mb_engine_requires_service() {
  local config_path="$1"
  local runtime_mode
  runtime_mode="$(mb_detect_engine_runtime_mode "${config_path}")"
  local engine=""
  local mode=""
  IFS='|' read -r engine mode <<<"${runtime_mode}"
  [[ "${mode}" == "service" ]]
}

mb_register_engine_pid() {
  local name="$1"
  local pid="$2"
  local runtime_root
  runtime_root="$(mb_engine_runtime_root)"
  mkdir -p "${runtime_root}"
  local pid_file="${runtime_root}/pids.tsv"
  printf '%s\t%s\n' "${name}" "${pid}" >> "${pid_file}"
}

mb_build_apptainer_service_prefix() {
  local output_array_name="$1"
  local image_path="$2"
  local env_array_name="$3"
  local bind_array_name="$4"
  local -a env_specs_ref=()
  local -a bind_specs_ref=()
  local -a container_cmd_ref=()
  eval "env_specs_ref=(\"\${${env_array_name}[@]-}\")"
  eval "bind_specs_ref=(\"\${${bind_array_name}[@]-}\")"

  if ! mb_ensure_apptainer; then
    mb_die "Apptainer is required to start managed engine services"
  fi

  container_cmd_ref=(apptainer exec --cleanenv)
  if mb_apptainer_use_nv; then
    container_cmd_ref+=(--nv)
  fi

  local bind_spec=""
  while IFS= read -r bind_spec; do
    if [[ -n "${bind_spec}" ]]; then
      container_cmd_ref+=(--bind "${bind_spec}")
    fi
  done < <(mb_container_bind_specs)

  for bind_spec in "${bind_specs_ref[@]}"; do
    if [[ -n "${bind_spec}" ]]; then
      container_cmd_ref+=(--bind "${bind_spec}")
    fi
  done

  container_cmd_ref+=("${image_path}" env)
  local env_spec=""
  for env_spec in "${env_specs_ref[@]}"; do
    if [[ -n "${env_spec}" ]]; then
      container_cmd_ref+=("${env_spec}")
    fi
  done

  mb_assign_shell_array "${output_array_name}" "${container_cmd_ref[@]}"
}

mb_launch_apptainer_service_process() {
  local name="$1"
  local cmd_array_name="$2"
  local -a container_cmd_ref=()
  eval "container_cmd_ref=(\"\${${cmd_array_name}[@]-}\")"

  local runtime_root
  runtime_root="$(mb_engine_runtime_root)"
  local log_dir="${runtime_root}/logs"
  mkdir -p "${log_dir}"
  local log_file="${log_dir}/${name}.log"

  "${container_cmd_ref[@]}" >"${log_file}" 2>&1 &
  local pid=$!
  local grace_s="${MAXIONBENCH_SERVICE_START_GRACE_S:-5}"
  local poll_s="${MAXIONBENCH_SERVICE_START_POLL_S:-0.25}"
  local max_checks=1
  max_checks="$(awk -v g="${grace_s}" -v p="${poll_s}" 'BEGIN { v=int((g / p)+0.999999); if (v < 1) v = 1; print v }')"
  local check_idx=0
  while [[ "${check_idx}" -lt "${max_checks}" ]]; do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      mb_log_file_tail "managed engine service ${name}" "${log_file}"
      mb_die "managed engine service ${name} exited early; see ${log_file}"
    fi
    check_idx=$((check_idx + 1))
    if [[ "${check_idx}" -lt "${max_checks}" ]]; then
      sleep "${poll_s}"
    fi
  done
  mb_register_engine_pid "${name}" "${pid}"
  mb_log "started engine service ${name} pid=${pid} log=${log_file}"
}

mb_start_apptainer_service_process() {
  local name="$1"
  local image_path="$2"
  local service_cmd="$3"
  local env_array_name="$4"
  local bind_array_name="$5"
  local workdir="${6:-}"
  local -a container_cmd=()

  mb_build_apptainer_service_prefix container_cmd "${image_path}" "${env_array_name}" "${bind_array_name}"
  # Login shells can rewrite PATH and other injected env vars inside OCI-derived
  # images, so managed services must run under a non-login shell.
  local shell_cmd="exec ${service_cmd}"
  if [[ -n "${workdir}" ]]; then
    shell_cmd="cd ${workdir} && exec ${service_cmd}"
  fi
  container_cmd+=(/bin/sh -c "${shell_cmd}")
  mb_launch_apptainer_service_process "${name}" container_cmd
}

mb_start_apptainer_service_argv_process() {
  local name="$1"
  local image_path="$2"
  local cmd_array_name="$3"
  local env_array_name="$4"
  local bind_array_name="$5"
  local -a cmd_args_ref=()
  local -a container_cmd=()
  eval "cmd_args_ref=(\"\${${cmd_array_name}[@]-}\")"

  if [[ "${#cmd_args_ref[@]}" -eq 0 ]]; then
    mb_die "missing argv command for managed engine service ${name}"
  fi

  mb_build_apptainer_service_prefix container_cmd "${image_path}" "${env_array_name}" "${bind_array_name}"
  container_cmd+=("${cmd_args_ref[@]}")
  mb_launch_apptainer_service_process "${name}" container_cmd
}

mb_start_qdrant_service() {
  local image_path
  image_path="$(mb_require_apptainer_service_image "MAXIONBENCH_QDRANT_IMAGE" "qdrant")"
  mb_validate_named_service_image "${image_path}" "qdrant"
  local runtime_root
  runtime_root="$(mb_engine_runtime_root)"
  local storage_dir="${runtime_root}/qdrant/storage"
  mkdir -p "${storage_dir}"
  local -a env_specs=(
    "QDRANT__SERVICE__HOST=0.0.0.0"
    "QDRANT__SERVICE__HTTP_PORT=${MAXIONBENCH_PORT_QDRANT}"
    "QDRANT__SERVICE__GRPC_PORT=${MAXIONBENCH_PORT_QDRANT_GRPC}"
    "QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage"
  )
  local -a bind_specs=("${storage_dir}:/qdrant/storage")
  local cmd="${MAXIONBENCH_QDRANT_START_CMD:-./entrypoint.sh}"
  mb_start_apptainer_service_process "qdrant" "${image_path}" "${cmd}" env_specs bind_specs "/qdrant"
}

mb_start_pgvector_service() {
  local image_path
  image_path="$(mb_require_apptainer_service_image "MAXIONBENCH_PGVECTOR_IMAGE" "pgvector")"
  mb_validate_named_service_image "${image_path}" "pgvector"
  local runtime_root
  runtime_root="$(mb_engine_runtime_root)"
  local data_dir="${runtime_root}/pgvector/data"
  local run_dir="${runtime_root}/pgvector/run"
  local pgvector_path
  pgvector_path="$(mb_pgvector_path_env)"
  mkdir -p "${data_dir}" "${run_dir}"
  local -a env_specs=(
    "PATH=${pgvector_path}"
    "POSTGRES_USER=postgres"
    "POSTGRES_PASSWORD=postgres"
    "POSTGRES_DB=postgres"
    "PGDATA=/var/lib/postgresql/data/pgdata"
  )
  local -a bind_specs=(
    "${data_dir}:/var/lib/postgresql/data"
    "${run_dir}:/var/run/postgresql"
  )
  if [[ -n "${MAXIONBENCH_PGVECTOR_START_CMD:-}" ]]; then
    mb_start_apptainer_service_process "pgvector" "${image_path}" "${MAXIONBENCH_PGVECTOR_START_CMD}" env_specs bind_specs
  else
    local -a cmd_args=()
    mb_service_default_start_args "pgvector" cmd_args
    mb_start_apptainer_service_argv_process "pgvector" "${image_path}" cmd_args env_specs bind_specs
  fi
}

mb_start_opensearch_service() {
  local image_path
  image_path="$(mb_require_apptainer_service_image "MAXIONBENCH_OPENSEARCH_IMAGE" "opensearch")"
  mb_validate_named_service_image "${image_path}" "opensearch"
  local runtime_root
  runtime_root="$(mb_engine_runtime_root)"
  local data_dir="${runtime_root}/opensearch/data"
  local logs_dir="${runtime_root}/opensearch/logs"
  local config_dir="${runtime_root}/opensearch/config"
  local config_path="${config_dir}/opensearch.yml"
  mkdir -p "${data_dir}" "${logs_dir}" "${config_dir}"
  cat > "${config_path}" <<EOF
cluster.name: maxionbench-slurm
network.host: 0.0.0.0
http.port: ${MAXIONBENCH_PORT_OPENSEARCH}
transport.port: ${MAXIONBENCH_PORT_OPENSEARCH_TRANSPORT}
discovery.type: single-node
plugins.security.disabled: true
path.data: /usr/share/opensearch/data
path.logs: /usr/share/opensearch/logs
EOF
  local -a env_specs=(
    "DISABLE_SECURITY_PLUGIN=true"
    "OPENSEARCH_JAVA_OPTS=${MAXIONBENCH_OPENSEARCH_JAVA_OPTS:--Xms512m -Xmx512m}"
  )
  local -a bind_specs=(
    "${data_dir}:/usr/share/opensearch/data"
    "${logs_dir}:/usr/share/opensearch/logs"
    "${config_path}:/usr/share/opensearch/config/opensearch.yml"
  )
  if [[ -n "${MAXIONBENCH_OPENSEARCH_START_CMD:-}" ]]; then
    mb_start_apptainer_service_process "opensearch" "${image_path}" "${MAXIONBENCH_OPENSEARCH_START_CMD}" env_specs bind_specs
  else
    mb_start_apptainer_service_process "opensearch" "${image_path}" "./opensearch-docker-entrypoint.sh opensearch" env_specs bind_specs "/usr/share/opensearch"
  fi
}

mb_start_weaviate_service() {
  local image_path
  image_path="$(mb_require_apptainer_service_image "MAXIONBENCH_WEAVIATE_IMAGE" "weaviate")"
  mb_validate_named_service_image "${image_path}" "weaviate"
  local runtime_root
  runtime_root="$(mb_engine_runtime_root)"
  local data_dir="${runtime_root}/weaviate/data"
  mkdir -p "${data_dir}"
  local -a env_specs=(
    "QUERY_DEFAULTS_LIMIT=20"
    "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true"
    "PERSISTENCE_DATA_PATH=/var/lib/weaviate"
    "DEFAULT_VECTORIZER_MODULE=none"
    "ENABLE_MODULES="
    "CLUSTER_HOSTNAME=node1"
  )
  local -a bind_specs=("${data_dir}:/var/lib/weaviate")
  if [[ -n "${MAXIONBENCH_WEAVIATE_START_CMD:-}" ]]; then
    mb_start_apptainer_service_process "weaviate" "${image_path}" "${MAXIONBENCH_WEAVIATE_START_CMD}" env_specs bind_specs
  else
    local -a cmd_args=()
    mb_service_default_start_args "weaviate" cmd_args
    mb_start_apptainer_service_argv_process "weaviate" "${image_path}" cmd_args env_specs bind_specs
  fi
}

mb_start_milvus_services() {
  local etcd_image
  local minio_image
  local milvus_image
  etcd_image="$(mb_require_apptainer_service_image "MAXIONBENCH_MILVUS_ETCD_IMAGE" "milvus-etcd")"
  minio_image="$(mb_require_apptainer_service_image "MAXIONBENCH_MILVUS_MINIO_IMAGE" "milvus-minio")"
  milvus_image="$(mb_require_apptainer_service_image "MAXIONBENCH_MILVUS_IMAGE" "milvus")"
  mb_validate_named_service_image "${etcd_image}" "milvus-etcd"
  mb_validate_named_service_image "${minio_image}" "milvus-minio"
  mb_validate_named_service_image "${milvus_image}" "milvus"

  local runtime_root
  runtime_root="$(mb_engine_runtime_root)"

  local etcd_dir="${runtime_root}/milvus/etcd"
  mkdir -p "${etcd_dir}"
  local -a etcd_env=()
  local -a etcd_binds=("${etcd_dir}:/etcd")
  if [[ -n "${MAXIONBENCH_MILVUS_ETCD_START_CMD:-}" ]]; then
    mb_start_apptainer_service_process "milvus-etcd" "${etcd_image}" "${MAXIONBENCH_MILVUS_ETCD_START_CMD}" etcd_env etcd_binds
  else
    local -a etcd_cmd_args=()
    mb_service_default_start_args "milvus-etcd" etcd_cmd_args
    mb_start_apptainer_service_argv_process "milvus-etcd" "${etcd_image}" etcd_cmd_args etcd_env etcd_binds
  fi

  local minio_dir="${runtime_root}/milvus/minio"
  mkdir -p "${minio_dir}"
  local -a minio_env=(
    "MINIO_ROOT_USER=minioadmin"
    "MINIO_ROOT_PASSWORD=minioadmin"
  )
  local -a minio_binds=("${minio_dir}:/minio_data")
  if [[ -n "${MAXIONBENCH_MILVUS_MINIO_START_CMD:-}" ]]; then
    mb_start_apptainer_service_process "milvus-minio" "${minio_image}" "${MAXIONBENCH_MILVUS_MINIO_START_CMD}" minio_env minio_binds
  else
    local -a minio_cmd_args=()
    mb_service_default_start_args "milvus-minio" minio_cmd_args
    mb_start_apptainer_service_argv_process "milvus-minio" "${minio_image}" minio_cmd_args minio_env minio_binds
  fi

  local milvus_dir="${runtime_root}/milvus/data"
  mkdir -p "${milvus_dir}"
  local -a milvus_env=(
    "ETCD_ENDPOINTS=127.0.0.1:${MAXIONBENCH_PORT_MILVUS_ETCD}"
    "MINIO_ADDRESS=127.0.0.1:${MAXIONBENCH_PORT_MILVUS_MINIO}"
    "MILVUS_PROXY_PORT=${MAXIONBENCH_PORT_MILVUS}"
    "MILVUS_METRICS_PORT=${MAXIONBENCH_PORT_MILVUS_METRICS}"
  )
  local -a milvus_binds=("${milvus_dir}:/var/lib/milvus")
  if [[ -n "${MAXIONBENCH_MILVUS_START_CMD:-}" ]]; then
    mb_start_apptainer_service_process "milvus" "${milvus_image}" "${MAXIONBENCH_MILVUS_START_CMD}" milvus_env milvus_binds
  else
    local -a milvus_cmd_args=()
    mb_service_default_start_args "milvus" milvus_cmd_args
    mb_start_apptainer_service_argv_process "milvus" "${milvus_image}" milvus_cmd_args milvus_env milvus_binds
  fi
}

mb_wait_named_adapter_health() {
  local adapter_name="$1"
  local adapter_options_json="$2"
  mb_log "waiting for adapter health adapter=${adapter_name} timeout=${MAXIONBENCH_ENGINE_WAIT_TIMEOUT_S}s"
  mb_python -m maxionbench.cli wait-adapter \
    --adapter "${adapter_name}" \
    --adapter-options-json "${adapter_options_json}" \
    --timeout-s "${MAXIONBENCH_ENGINE_WAIT_TIMEOUT_S}" \
    --json
}

mb_start_engine_services() {
  local config_path="$1"
  local runtime_mode
  runtime_mode="$(mb_detect_engine_runtime_mode "${config_path}")"
  local engine=""
  local mode=""
  IFS='|' read -r engine mode <<<"${runtime_mode}"

  if [[ "${mode}" != "service" ]]; then
    mb_log "engine ${engine} uses embedded/inproc mode; no managed service start needed"
    return 0
  fi

  case "${engine}" in
    qdrant)
      mb_start_qdrant_service
      ;;
    pgvector)
      mb_start_pgvector_service
      ;;
    opensearch)
      mb_start_opensearch_service
      ;;
    weaviate)
      mb_start_weaviate_service
      ;;
    milvus)
      mb_start_milvus_services
      ;;
    lancedb-service)
      mb_die "lancedb-service HTTP mode is not implemented for Slurm Apptainer jobs; set MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI or use lancedb-inproc"
      ;;
    *)
      mb_log "engine ${engine} does not require a managed service"
      ;;
  esac
}

mb_stop_engine_services() {
  local runtime_root
  runtime_root="$(mb_engine_runtime_root)"
  local pid_file="${runtime_root}/pids.tsv"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi

  awk '{ lines[NR] = $0 } END { for (idx = NR; idx >= 1; idx--) print lines[idx] }' "${pid_file}" | while IFS=$'\t' read -r name pid; do
    if [[ -z "${pid:-}" ]]; then
      continue
    fi
    if kill -0 "${pid}" >/dev/null 2>&1; then
      mb_log "stopping engine service ${name} pid=${pid}"
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -f "${pid_file}"
}

mb_wait_engine_health() {
  local config_path="$1"
  local resolved
  resolved="$(mb_resolve_config "${config_path}")"
  mb_log "waiting for adapter health config=${resolved} timeout=${MAXIONBENCH_ENGINE_WAIT_TIMEOUT_S}s"
  mb_python -m maxionbench.cli wait-adapter \
    --config "${resolved}" \
    --timeout-s "${MAXIONBENCH_ENGINE_WAIT_TIMEOUT_S}" \
    --json
}

mb_stage_config_to_tmp() {
  local config_path="$1"
  local resolved
  resolved="$(mb_resolve_config "${config_path}")"
  local stage_root="${SLURM_TMPDIR}/maxionbench_stage/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
  local staged="${stage_root}/config.yaml"
  export MB_STAGE_ROOT="${stage_root}"
  export MB_STAGED_CONFIG_PATH="${staged}"
  mb_python - <<'PY' "${resolved}" "${staged}" "${stage_root}" "${MB_OUTPUT_TMP:-}" "${ROOT_DIR}"
import pathlib
import shutil
import sys
import yaml

from maxionbench.orchestration.config_schema import expand_env_placeholders

src = pathlib.Path(sys.argv[1]).resolve()
dst = pathlib.Path(sys.argv[2]).resolve()
stage_root = pathlib.Path(sys.argv[3]).resolve()
out_tmp = pathlib.Path(sys.argv[4]).resolve() if sys.argv[4] else None
repo_root = pathlib.Path(sys.argv[5]).resolve()

with src.open("r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}
if not isinstance(payload, dict):
    raise ValueError(f"Expected mapping config: {src}")
payload = expand_env_placeholders(payload)


def resolve_source(raw_value: str) -> pathlib.Path:
    source = pathlib.Path(str(raw_value))
    if source.is_absolute():
        return source.resolve()
    config_relative = (src.parent / source).resolve()
    if config_relative.exists():
        return config_relative
    repo_relative = (repo_root / source).resolve()
    return repo_relative

def stage_any_path(key: str, bucket: str) -> None:
    value = payload.get(key)
    if not value:
        return
    source = resolve_source(str(value))
    if not source.exists():
        return
    bucket_dir = stage_root / "datasets" / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    target = bucket_dir / source.name
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    payload[key] = str(target)

stage_any_path("dataset_path", "dataset")
stage_any_path("processed_dataset_path", "processed")
stage_any_path("d2_base_fvecs_path", "d2")
stage_any_path("d2_query_fvecs_path", "d2")
stage_any_path("d2_gt_ivecs_path", "d2")
stage_any_path("d4_beir_root", "d4_beir")
stage_any_path("d4_crag_path", "d4_crag")

if out_tmp is not None and "output_d3_params_path" in payload:
    payload["output_d3_params_path"] = str(out_tmp / "d3_params.yaml")

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(payload, handle, sort_keys=True)
PY
  echo "${staged}"
}

mb_read_config_field() {
  local config_path="$1"
  local field_name="$2"
  local resolved
  resolved="$(mb_resolve_config "${config_path}")"
  mb_python - <<'PY' "${resolved}" "${field_name}"
import sys
import yaml

cfg_path = sys.argv[1]
field_name = sys.argv[2]
with open(cfg_path, "r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}
if not isinstance(payload, dict):
    raise ValueError(f"Expected mapping config: {cfg_path}")
value = payload.get(field_name)
print("" if value is None else str(value))
PY
}

mb_prepare_output_paths() {
  local scenario="$1"
  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  local job_id="${SLURM_JOB_ID:-local}"
  local run_id="${job_id}_${task_id}_${scenario}_$(date -u +%Y%m%dT%H%M%SZ)"
  local output_root="${MAXIONBENCH_OUTPUT_ROOT:-artifacts/runs/slurm}"
  local resolved_output_root
  if [[ "${output_root}" = /* ]]; then
    resolved_output_root="${output_root}"
  else
    resolved_output_root="${ROOT_DIR}/${output_root}"
  fi
  export MB_RUN_ID="${run_id}"
  export MB_OUTPUT_TMP="${SLURM_TMPDIR}/maxionbench/${run_id}"
  export MB_OUTPUT_FINAL="${resolved_output_root}/${run_id}"
  mkdir -p "${MB_OUTPUT_TMP}" "$(dirname "${MB_OUTPUT_FINAL}")"
}

mb_copy_back_output() {
  if [[ -z "${MB_OUTPUT_TMP:-}" || -z "${MB_OUTPUT_FINAL:-}" ]]; then
    mb_die "output paths are not initialized"
  fi
  mkdir -p "${MB_OUTPUT_FINAL}"
  cp -R "${MB_OUTPUT_TMP}/." "${MB_OUTPUT_FINAL}/"
  mb_log "copied artifacts to ${MB_OUTPUT_FINAL}"
}

mb_capture_local_diagnostics() {
  if [[ -z "${MB_OUTPUT_TMP:-}" ]]; then
    return 0
  fi

  local capture_root="${MB_OUTPUT_TMP}/logs/local_runtime"
  mkdir -p "${capture_root}"

  local runtime_root
  runtime_root="$(mb_engine_runtime_root)"
  if [[ -e "${runtime_root}" ]]; then
    cp -R "${runtime_root}" "${capture_root}/engine_runtime"
    mb_log "captured engine runtime diagnostics to ${capture_root}/engine_runtime"
  fi

  if [[ -n "${MB_STAGE_ROOT:-}" && -e "${MB_STAGE_ROOT}" ]]; then
    cp -R "${MB_STAGE_ROOT}" "${capture_root}/stage_root"
    mb_log "captured staged config diagnostics to ${capture_root}/stage_root"
  fi
}

mb_cleanup_local_path() {
  local target="$1"
  if [[ -z "${target}" ]]; then
    return 0
  fi
  if [[ ! -e "${target}" ]]; then
    return 0
  fi
  if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    mb_log "skipping local scratch cleanup for ${target}; SLURM_TMPDIR is unset"
    return 0
  fi
  case "${target}" in
    "${SLURM_TMPDIR}"/*)
      rm -rf "${target}"
      mb_log "removed local scratch ${target}"
      ;;
    *)
      mb_log "refusing to remove non-scratch path ${target}"
      ;;
  esac
}

mb_cleanup_local_runtime() {
  if ! mb_flag_enabled "${MAXIONBENCH_CLEANUP_LOCAL_SCRATCH:-1}"; then
    mb_log "keeping local scratch (MAXIONBENCH_CLEANUP_LOCAL_SCRATCH=${MAXIONBENCH_CLEANUP_LOCAL_SCRATCH:-0})"
    return 0
  fi
  mb_cleanup_local_path "${MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI:-}"
  mb_cleanup_local_path "$(mb_engine_runtime_root)"
  mb_cleanup_local_path "${MB_STAGE_ROOT:-}"
  mb_cleanup_local_path "${MB_OUTPUT_TMP:-}"
}

mb_finalize_job() {
  local status="${1:-0}"
  local service_started="${2:-0}"

  set +e
  if [[ "${service_started}" == "1" ]]; then
    mb_stop_engine_services
  fi
  mb_capture_local_diagnostics
  if [[ -n "${MB_OUTPUT_TMP:-}" && -d "${MB_OUTPUT_TMP:-}" && -n "${MB_OUTPUT_FINAL:-}" ]]; then
    mb_copy_back_output
  fi
  mb_cleanup_local_runtime
  set -e
  return 0
}

mb_run_benchmark() {
  local config_path="$1"
  shift || true
  local resolved_config
  resolved_config="$(mb_resolve_config "${config_path}")"
  if [[ "${MAXIONBENCH_SKIP_PRE_RUN_GATE}" != "1" ]]; then
    local gate_args=(
      pre-run-gate
      --config "${resolved_config}"
      --conformance-matrix "${MAXIONBENCH_CONFORMANCE_MATRIX}"
      --behavior-dir "${ROOT_DIR}/docs/behavior"
      --json
    )
    if [[ "${MAXIONBENCH_ALLOW_GPU_UNAVAILABLE}" == "1" ]]; then
      gate_args+=(--allow-gpu-unavailable)
    fi
    mb_log "running pre-run readiness gate for config=${resolved_config}"
    mb_python -m maxionbench.cli "${gate_args[@]}"
  else
    mb_log "skipping pre-run readiness gate (MAXIONBENCH_SKIP_PRE_RUN_GATE=1)"
  fi
  mb_log "running scenario config=${resolved_config}"
  mb_python -m maxionbench.orchestration.runner \
    --config "${resolved_config}" \
    --no-retry \
    --repeats "${MAXIONBENCH_REPEATS:-3}" \
    --output-dir "${MB_OUTPUT_TMP}" \
    "$@"
}
