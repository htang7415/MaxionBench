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

mb_scratch_preflight() {
  local config_path="$1"
  local resolved
  resolved="$(mb_resolve_config "${config_path}")"
  local summary
  local status

  set +e
  summary="$(python -m maxionbench.orchestration.slurm.preflight \
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
  payload="$(python - <<'PY'
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
  python - <<'PY' "${resolved}" "${staged}" "${SLURM_TMPDIR}" "${MB_OUTPUT_TMP:-}"
import os
import pathlib
import shutil
import sys
import yaml

src = pathlib.Path(sys.argv[1]).resolve()
dst = pathlib.Path(sys.argv[2]).resolve()
tmpdir = pathlib.Path(sys.argv[3]).resolve()
out_tmp = pathlib.Path(sys.argv[4]).resolve() if sys.argv[4] else None

with src.open("r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}
if not isinstance(payload, dict):
    raise ValueError(f"Expected mapping config: {src}")

def stage_any_path(key: str, bucket: str) -> None:
    value = payload.get(key)
    if not value:
        return
    source = pathlib.Path(str(value))
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
    python -m maxionbench.cli "${gate_args[@]}"
  else
    mb_log "skipping pre-run readiness gate (MAXIONBENCH_SKIP_PRE_RUN_GATE=1)"
  fi
  mb_log "running scenario config=${resolved_config}"
  python -m maxionbench.orchestration.runner \
    --config "${resolved_config}" \
    --no-retry \
    --repeats "${MAXIONBENCH_REPEATS:-3}" \
    --output-dir "${MB_OUTPUT_TMP}" \
    "$@"
}
