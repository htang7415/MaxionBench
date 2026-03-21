#!/usr/bin/env bash

# Shared Apptainer service contracts used by both Slurm runtime startup and
# build-time image validation.

mb_assign_shell_array() {
  local array_name="$1"
  shift

  local rendered=""
  local value=""
  for value in "$@"; do
    rendered="${rendered} $(printf '%q' "${value}")"
  done

  if [[ $# -eq 0 ]]; then
    eval "${array_name}=()"
    return 0
  fi

  eval "${array_name}=(${rendered# })"
}

mb_service_contract_kind() {
  case "$1" in
    qdrant)
      printf '%s\n' "qdrant-layout"
      ;;
    pgvector)
      printf '%s\n' "pgvector"
      ;;
    opensearch)
      printf '%s\n' "opensearch-layout"
      ;;
    weaviate|milvus|milvus-etcd|milvus-minio)
      printf '%s\n' "probe"
      ;;
    *)
      return 1
      ;;
  esac
}

mb_service_default_start_mode() {
  case "$1" in
    qdrant|opensearch)
      printf '%s\n' "shell"
      ;;
    pgvector|weaviate|milvus|milvus-etcd|milvus-minio)
      printf '%s\n' "argv"
      ;;
    *)
      return 1
      ;;
  esac
}

mb_service_probe_args() {
  local service_name="$1"
  local array_name="$2"

  case "${service_name}" in
    weaviate)
      mb_assign_shell_array "${array_name}" "weaviate" "--help"
      ;;
    milvus)
      mb_assign_shell_array "${array_name}" "milvus" "--help"
      ;;
    milvus-etcd)
      mb_assign_shell_array "${array_name}" "etcd" "--version"
      ;;
    milvus-minio)
      mb_assign_shell_array "${array_name}" "minio" "--version"
      ;;
    *)
      return 1
      ;;
  esac
}

mb_service_default_start_args() {
  local service_name="$1"
  local array_name="$2"

  case "${service_name}" in
    pgvector)
      mb_assign_shell_array "${array_name}" \
        "docker-entrypoint.sh" \
        "postgres" \
        "-p" \
        "${MAXIONBENCH_PORT_PGVECTOR}"
      ;;
    weaviate)
      mb_assign_shell_array "${array_name}" \
        "weaviate" \
        "--host" \
        "0.0.0.0" \
        "--port" \
        "${MAXIONBENCH_PORT_WEAVIATE}"
      ;;
    milvus-etcd)
      mb_assign_shell_array "${array_name}" \
        "etcd" \
        "-advertise-client-urls=http://127.0.0.1:${MAXIONBENCH_PORT_MILVUS_ETCD}" \
        "-listen-client-urls=http://0.0.0.0:${MAXIONBENCH_PORT_MILVUS_ETCD}" \
        "--data-dir=/etcd"
      ;;
    milvus-minio)
      mb_assign_shell_array "${array_name}" \
        "minio" \
        "server" \
        "/minio_data" \
        "--address" \
        ":${MAXIONBENCH_PORT_MILVUS_MINIO}" \
        "--console-address" \
        ":${MAXIONBENCH_PORT_MILVUS_MINIO_CONSOLE}"
      ;;
    milvus)
      mb_assign_shell_array "${array_name}" "milvus" "run" "standalone"
      ;;
    *)
      return 1
      ;;
  esac
}
