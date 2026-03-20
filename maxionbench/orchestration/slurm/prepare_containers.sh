#!/usr/bin/env bash
set -euo pipefail
SLURM_DIR="${MAXIONBENCH_SLURM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=/dev/null
source "${SLURM_DIR}/common.sh"

mb_require_tmpdir

build_script="${ROOT_DIR}/scripts/build_containers.sh"
definition_file="${ROOT_DIR}/maxionbench.def"
shared_root="${MAXIONBENCH_SHARED_ROOT:-${ROOT_DIR}}"
containers_dir="$(mb_resolve_host_path "${shared_root}/containers")"

if [[ ! -f "${build_script}" ]]; then
  mb_die "missing container build helper: ${build_script}"
fi
if [[ ! -f "${definition_file}" ]]; then
  mb_die "missing Apptainer definition file: ${definition_file}"
fi
if ! mb_ensure_apptainer; then
  mb_die "apptainer is required inside the prepare_containers Slurm job"
fi

mkdir -p "${containers_dir}"
mb_log "building shared Apptainer images into ${containers_dir}"
bash "${build_script}" --output-dir "${containers_dir}" --only-missing
mb_log "shared Apptainer image preparation finished"
