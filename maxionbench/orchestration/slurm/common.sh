#!/usr/bin/env bash
set -euo pipefail

# Shared Slurm helpers for MaxionBench jobs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
export MAXIONBENCH_DATASET_ENV_SH="${MAXIONBENCH_DATASET_ENV_SH:-${ROOT_DIR}/artifacts/prefetch/dataset_env.sh}"
export MAXIONBENCH_D3_PARAMS_PATH="${MAXIONBENCH_D3_PARAMS_PATH:-}"
export MAXIONBENCH_SLURM_RUN_MANIFEST="${MAXIONBENCH_SLURM_RUN_MANIFEST:-}"

mb_log() {
  echo "[maxionbench][$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

mb_die() {
  mb_log "ERROR: $*"
  exit 1
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

mb_apptainer_use_nv() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    return 0
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
      if ! command -v apptainer >/dev/null 2>&1; then
        mb_die "MAXIONBENCH_CONTAINER_RUNTIME=apptainer requires `apptainer` in PATH"
      fi
      if [[ -z "${MAXIONBENCH_CONTAINER_IMAGE:-}" ]]; then
        mb_die "MAXIONBENCH_CONTAINER_IMAGE must be set when MAXIONBENCH_CONTAINER_RUNTIME=apptainer"
      fi
      local resolved_image
      resolved_image="$(mb_resolve_host_path "${MAXIONBENCH_CONTAINER_IMAGE}")"
      if [[ ! -f "${resolved_image}" ]]; then
        mb_die "apptainer image not found: ${resolved_image}"
      fi

      local -a container_cmd=(apptainer exec)
      if mb_apptainer_use_nv; then
        container_cmd+=(--nv)
      fi
      local bind_spec=""
      while IFS= read -r bind_spec; do
        if [[ -n "${bind_spec}" ]]; then
          container_cmd+=(--bind "${bind_spec}")
        fi
      done < <(mb_container_bind_specs)
      container_cmd+=("${resolved_image}" env "PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}")

      if [[ -n "${MAXIONBENCH_HF_CACHE_DIR:-}" ]]; then
        local resolved_hf_cache
        resolved_hf_cache="$(mb_resolve_host_path "${MAXIONBENCH_HF_CACHE_DIR}")"
        container_cmd+=(
          "HF_HOME=${resolved_hf_cache}"
          "HUGGINGFACE_HUB_CACHE=${resolved_hf_cache}/hub"
          "TRANSFORMERS_CACHE=${resolved_hf_cache}/transformers"
        )
      fi
      container_cmd+=(python "$@")
      "${container_cmd[@]}"
      ;;
    *)
      mb_die "unsupported MAXIONBENCH_CONTAINER_RUNTIME=${MAXIONBENCH_CONTAINER_RUNTIME}; supported: apptainer"
      ;;
  esac
}

mb_scratch_preflight() {
  local config_path="$1"
  local resolved
  resolved="$(mb_resolve_config "${config_path}")"
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

  if [[ "${status}" -eq 0 ]]; then
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
      mb_log "scratch preflight failed, applying fallback config ${resolved_fallback}"
      export MB_PREFLIGHT_CONFIG="${resolved_fallback}"
      return 0
    fi
  fi

  mb_die "scratch preflight failed and no valid fallback config was available"
}

mb_allocate_ports() {
  local payload
  payload="$(mb_python - <<'PY'
from maxionbench.runtime.ports import allocate_named_ports
ports = allocate_named_ports(["qdrant", "weaviate", "opensearch", "lancedb", "milvus"], base=20000, span=20000)
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
}

mb_stage_config_to_tmp() {
  local config_path="$1"
  local resolved
  resolved="$(mb_resolve_config "${config_path}")"
  local staged="${SLURM_TMPDIR}/maxionbench_config_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.yaml"
  mb_python - <<'PY' "${resolved}" "${staged}" "${SLURM_TMPDIR}" "${MB_OUTPUT_TMP:-}" "${ROOT_DIR}"
import pathlib
import shutil
import sys
import yaml

from maxionbench.orchestration.config_schema import expand_env_placeholders

src = pathlib.Path(sys.argv[1]).resolve()
dst = pathlib.Path(sys.argv[2]).resolve()
tmpdir = pathlib.Path(sys.argv[3]).resolve()
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
    bucket_dir = tmpdir / "datasets" / bucket
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
