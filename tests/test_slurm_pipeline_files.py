from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tomllib


def _service_image_payload(image_name: str) -> str:
    return f"{Path(image_name).stem}-ok\n"


def test_slurm_pipeline_files_exist_and_reference_full_matrix_flow() -> None:
    script = Path("run_slurm_pipeline.sh")
    smoke_script = Path("test_slrum_pipeline.sh")
    build_script = Path("scripts/build_containers.sh")
    compose_file = Path("docker-compose.yml")
    service_contracts = Path("maxionbench/orchestration/slurm/service_contracts.sh")
    definition = Path("maxionbench.def")
    prepare_containers = Path("maxionbench/orchestration/slurm/prepare_containers.sh")
    download = Path("maxionbench/orchestration/slurm/download_datasets.sh")
    preprocess = Path("maxionbench/orchestration/slurm/preprocess_datasets.sh")
    conformance = Path("maxionbench/orchestration/slurm/conformance_matrix.sh")
    postprocess = Path("maxionbench/orchestration/slurm/postprocess.sh")
    profiles = Path("maxionbench/orchestration/slurm/profiles_clusters.example.yaml")
    env_example = Path(".env.slurm.example")

    for path in (script, smoke_script, build_script, compose_file, service_contracts, definition, prepare_containers, download, preprocess, conformance, postprocess, profiles, env_example):
        assert path.exists(), path

    text = script.read_text(encoding="utf-8")
    assert "submit-slurm-plan" in text
    assert "CLI_CMD=()" in text
    assert "--download-datasets" in text
    assert "--preprocess-datasets" in text
    assert "--prefetch-datasets" in text
    assert "--include-postprocess" in text
    assert "--full-matrix" in text
    assert "--container-runtime apptainer" in text
    assert "python -m maxionbench.cli" in text
    assert "python3 -m maxionbench.cli" in text
    assert "euler_apptainer" in text
    assert "nrel_apptainer" in text
    assert "--shared-root <path>" in text
    assert ".env.slurm.<cluster> is auto-loaded before CLI overrides" in text
    assert 'SUBMIT_ROOT="$(pwd -P)"' in text
    assert 'printf \'%s\\n\' "${ROOT_DIR}"' in text
    assert "prepare_shared_layout()" in text
    assert "validate_container_build_prerequisites()" in text
    assert "scripts/build_containers.sh" in text
    assert "prepare_containers.sh" in text
    assert "--prepare-containers" in text
    assert "full-matrix reruns require all GPU jobs" in text
    assert "--allow-reduced-matrix" in text

    smoke_text = smoke_script.read_text(encoding="utf-8")
    assert "run_slurm_pipeline.sh" in smoke_text
    assert "--allow-reduced-matrix" in smoke_text
    assert "s2_filtered_ann" in smoke_text
    assert "s4_hybrid" in smoke_text
    assert "s5_rerank" in smoke_text
    assert "faiss_gpu" in smoke_text
    assert "qdrant" in smoke_text
    assert "opensearch" in smoke_text

    build_text = build_script.read_text(encoding="utf-8")
    assert "apptainer build" in build_text
    assert "apptainer pull" in build_text
    assert "apptainer exec --cleanenv" in build_text
    assert "--env" in build_text
    assert "/bin/sh -c" in build_text
    assert "apptainer inspect" in build_text
    assert "python -s" in build_text
    assert "--output-dir" in build_text
    assert "--only-missing" in build_text
    assert "service_contracts.sh" in build_text
    assert "[ -x /qdrant/entrypoint.sh ] && [ -x /qdrant/qdrant ]" in build_text
    assert "docker://milvusdb/milvus:v2.5.27" in build_text

    compose_text = compose_file.read_text(encoding="utf-8")
    assert "quay.io/coreos/etcd:v3.5.18" in compose_text
    assert "minio/minio:RELEASE.2024-11-07T00-52-20Z" in compose_text
    assert "milvusdb/milvus:v2.5.27" in compose_text
    assert "milvusdb/milvus:latest" not in compose_text

    prepare_text = prepare_containers.read_text(encoding="utf-8")
    assert "mb_ensure_apptainer" in prepare_text
    assert "scripts/build_containers.sh" in prepare_text
    assert "--only-missing" in prepare_text

    definition_text = definition.read_text(encoding="utf-8")
    assert "Bootstrap: docker" in definition_text
    assert "From: python:3.11-slim" in definition_text
    assert "%post" in definition_text
    assert "set -eu" in definition_text
    assert "%runscript" in definition_text
    assert "python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch" in definition_text
    assert 'python -m pip install --no-build-isolation ".[dev,engines,reporting,datasets,rerank]"' in definition_text

    download_text = download.read_text(encoding="utf-8")
    assert "maxionbench.cli download-datasets" in download_text
    assert "--root" in download_text
    assert "--cache-dir" in download_text

    preprocess_text = preprocess.read_text(encoding="utf-8")
    assert "preprocess-datasets ann-hdf5" in preprocess_text
    assert "preprocess-datasets d3-yfcc" in preprocess_text
    assert "preprocess-datasets beir" in preprocess_text
    assert "preprocess-datasets crag" in preprocess_text

    conformance_text = conformance.read_text(encoding="utf-8")
    assert "mb_wait_named_adapter_health" in conformance_text
    assert "maxionbench.cli conformance-matrix" in conformance_text

    postprocess_text = postprocess.read_text(encoding="utf-8")
    assert "maxionbench.cli validate" in postprocess_text
    assert "maxionbench.cli report" in postprocess_text

    profiles_text = profiles.read_text(encoding="utf-8")
    assert "cluster_a_apptainer:" in profiles_text
    assert "cluster_b_apptainer:" in profiles_text
    assert "prepare_containers:" in profiles_text
    assert "download_datasets:" in profiles_text
    assert "preprocess_datasets:" in profiles_text
    assert "conformance:" in profiles_text
    assert "postprocess:" in profiles_text

    env_text = env_example.read_text(encoding="utf-8")
    assert "MAXIONBENCH_SLURM_ACCOUNT=" in env_text
    assert "MAXIONBENCH_APPTAINER_MODULE=" in env_text
    assert "MAXIONBENCH_APPTAINER_CACHE_DIR=" in env_text
    assert "MAXIONBENCH_APPTAINER_TMPDIR=" in env_text
    assert "MAXIONBENCH_CLEANUP_LOCAL_SCRATCH=" in env_text
    assert "MAXIONBENCH_CONTAINER_IMAGE=" in env_text
    assert "MAXIONBENCH_QDRANT_IMAGE=" in env_text
    assert "MAXIONBENCH_PGVECTOR_IMAGE=" in env_text
    assert "MAXIONBENCH_OPENSEARCH_IMAGE=" in env_text
    assert "MAXIONBENCH_WEAVIATE_IMAGE=" in env_text
    assert "MAXIONBENCH_MILVUS_ETCD_IMAGE=" in env_text
    assert "MAXIONBENCH_MILVUS_MINIO_IMAGE=" in env_text
    assert "MAXIONBENCH_MILVUS_IMAGE=" in env_text
    assert "MAXIONBENCH_DATASET_ROOT=" in env_text
    assert "MAXIONBENCH_OUTPUT_ROOT=" in env_text


def test_datasets_extra_pins_known_good_pytz() -> None:
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = payload["project"]["optional-dependencies"]["datasets"]
    assert "pytz==2023.3.post1" in extras


def test_slurm_pipeline_shell_scripts_are_bash_parseable() -> None:
    for path in (
        "run_slurm_pipeline.sh",
        "test_slrum_pipeline.sh",
        "scripts/build_containers.sh",
        "maxionbench/orchestration/slurm/service_contracts.sh",
        "maxionbench/orchestration/slurm/prepare_containers.sh",
        "maxionbench/orchestration/slurm/download_datasets.sh",
        "maxionbench/orchestration/slurm/preprocess_datasets.sh",
        "maxionbench/orchestration/slurm/conformance_matrix.sh",
        "maxionbench/orchestration/slurm/postprocess.sh",
    ):
        completed = subprocess.run(
            ["bash", "-n", path],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, f"{path}: {completed.stdout}{completed.stderr}"


def test_main_container_definition_enables_fail_fast_post_install() -> None:
    definition_text = Path("maxionbench.def").read_text(encoding="utf-8")

    assert "%post" in definition_text
    assert "\n    set -eu\n" in definition_text
    assert "apt-get update && apt-get install -y --no-install-recommends" in definition_text
    assert 'python -m pip install --no-build-isolation ".[dev,engines,reporting,datasets,rerank]"' in definition_text


def test_build_containers_only_missing_accepts_opensearch_layout_and_shellless_milvus_images(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = repo_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    service_contracts_dir = repo_dir / "maxionbench" / "orchestration" / "slurm"
    service_contracts_dir.mkdir(parents=True, exist_ok=True)
    build_script = scripts_dir / "build_containers.sh"
    build_script.write_text(Path("scripts/build_containers.sh").read_text(encoding="utf-8"), encoding="utf-8")
    build_script.chmod(0o755)
    (service_contracts_dir / "service_contracts.sh").write_text(
        Path("maxionbench/orchestration/slurm/service_contracts.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo_dir / "maxionbench.def").write_text("Bootstrap: docker\nFrom: python:3.11-slim\n", encoding="utf-8")

    output_dir = repo_dir / "containers"
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_name in (
        "maxionbench.sif",
        "qdrant.sif",
        "pgvector.sif",
        "opensearch.sif",
        "weaviate.sif",
        "milvus.sif",
        "milvus-etcd.sif",
        "milvus-minio.sif",
    ):
        (output_dir / image_name).write_text(_service_image_payload(image_name), encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    apptainer_log = tmp_path / "apptainer.log"
    apptainer_stub = bin_dir / "apptainer"
    apptainer_stub.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MAXIONBENCH_TEST_APPTAINER_LOG}"
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
if [[ "${1:-}" == "build" && "${2:-}" == "--help" ]]; then
  printf '%s\\n' "--fakeroot"
  exit 0
fi
if [[ "${1:-}" == "exec" && "${2:-}" == "--cleanenv" ]]; then
  raw_args="$*"
  shift 2
  while [[ "${1:-}" == "--env" ]]; do
    shift 2
  done
  image="${1:-}"
  shift
  post_args="$*"
  case "${image##*/}" in
    maxionbench.sif)
      if [[ "${post_args}" == python* ]]; then
        exit 0
      fi
      ;;
    qdrant.sif)
      if [[ "${post_args}" == *"/bin/sh -c [ -x /qdrant/entrypoint.sh ] && [ -x /qdrant/qdrant ]"* ]]; then
        exit 0
      fi
      ;;
    pgvector.sif)
      if [[ "${raw_args}" == *"--env PATH=/usr/lib/postgresql/16/bin:"* && "${post_args}" == *"/bin/sh -c"* && "${post_args}" == *"command -v docker-entrypoint.sh"* && "${post_args}" == *"command -v postgres"* && "${post_args}" == *"command -v initdb"* ]]; then
        exit 0
      fi
      ;;
    opensearch.sif)
      if [[ "${post_args}" == *"/bin/sh -c [ -x /usr/share/opensearch/bin/opensearch ] && [ -x /usr/share/opensearch/opensearch-docker-entrypoint.sh ] && [ -x /usr/share/opensearch/jdk/bin/java ]"* ]]; then
        exit 0
      fi
      ;;
    weaviate.sif)
      if [[ "${post_args}" == "weaviate --help" ]]; then
        exit 0
      fi
      ;;
    milvus.sif)
      if [[ "${post_args}" == "milvus --help" ]]; then
        exit 0
      fi
      ;;
    milvus-etcd.sif)
      if [[ "${post_args}" == "etcd --version" ]]; then
        exit 0
      fi
      ;;
    milvus-minio.sif)
      if [[ "${post_args}" == "minio --version" ]]; then
        exit 0
      fi
      ;;
  esac
fi
exit 1
""",
        encoding="utf-8",
    )
    apptainer_stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
    env["MAXIONBENCH_TEST_APPTAINER_LOG"] = str(apptainer_log)

    completed = subprocess.run(
        [
            "bash",
            str(build_script),
            "--output-dir",
            str(output_dir),
            "--only-missing",
        ],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"+ skipping existing {output_dir / 'pgvector.sif'}" in completed.stdout
    log_text = apptainer_log.read_text(encoding="utf-8")
    assert "/bin/sh -c command -v docker-entrypoint.sh" in log_text
    assert "/bin/sh -lc" not in log_text
    assert "--env PATH=/usr/lib/postgresql/16/bin:" in log_text
    assert "/bin/sh -c [ -x /usr/share/opensearch/bin/opensearch ] && [ -x /usr/share/opensearch/opensearch-docker-entrypoint.sh ] && [ -x /usr/share/opensearch/jdk/bin/java ]" in log_text
    assert "opensearch.sif opensearch --version" not in log_text
    assert "milvus-etcd.sif etcd --version" in log_text
    assert "milvus-minio.sif minio --version" in log_text
    assert "milvus.sif milvus --help" in log_text
    assert "/bin/sh -c command -v etcd" not in log_text
    assert "/bin/sh -c command -v minio" not in log_text


def test_run_slurm_pipeline_derives_cluster_storage_defaults(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    source_script = Path("run_slurm_pipeline.sh")
    script_path = repo_dir / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)
    submit_root = tmp_path / "submit_root"
    submit_root.mkdir(parents=True, exist_ok=True)
    container_image = tmp_path / "images" / "maxionbench.sif"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_log = tmp_path / "maxionbench_env.log"
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'DATASET=%s\\n' "${MAXIONBENCH_DATASET_ROOT:-}"
  printf 'CACHE=%s\\n' "${MAXIONBENCH_DATASET_CACHE_DIR:-}"
  printf 'OUTPUT=%s\\n' "${MAXIONBENCH_OUTPUT_ROOT:-}"
  printf 'FIGURES=%s\\n' "${MAXIONBENCH_FIGURES_ROOT:-}"
  printf 'HF=%s\\n' "${MAXIONBENCH_HF_CACHE_DIR:-}"
  printf 'ARGS=%s\\n' "$*"
} > "${MAXIONBENCH_STUB_LOG}"
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MAXIONBENCH_STUB_LOG"] = str(stub_log)
    env["USER"] = "tester"
    env.pop("MAXIONBENCH_SHARED_ROOT", None)
    env.pop("MAXIONBENCH_DATASET_ROOT", None)
    env.pop("MAXIONBENCH_DATASET_CACHE_DIR", None)
    env.pop("MAXIONBENCH_OUTPUT_ROOT", None)
    env.pop("MAXIONBENCH_FIGURES_ROOT", None)
    env.pop("MAXIONBENCH_HF_CACHE_DIR", None)

    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "--cluster",
            "euler",
            "--container-image",
            str(container_image),
        ],
        cwd=submit_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    env_file = repo_dir / ".env.slurm.euler"
    assert env_file.exists()
    env_text = env_file.read_text(encoding="utf-8")
    assert f"MAXIONBENCH_SHARED_ROOT={repo_dir}" in env_text
    assert f"MAXIONBENCH_DATASET_ROOT={repo_dir}/dataset" in env_text
    assert f"MAXIONBENCH_APPTAINER_CACHE_DIR={repo_dir}/.cache/apptainer" in env_text
    assert f"MAXIONBENCH_APPTAINER_TMPDIR={repo_dir}/.cache/apptainer/tmp" in env_text
    assert "MAXIONBENCH_APPTAINER_MODULE=apptainer" in env_text
    assert f"+ wrote cluster env {env_file}" in completed.stdout
    log_text = stub_log.read_text(encoding="utf-8")
    assert f"DATASET={repo_dir}/dataset" in log_text
    assert f"CACHE={repo_dir}/.cache" in log_text
    assert f"OUTPUT={repo_dir}/results" in log_text
    assert f"FIGURES={repo_dir}/figures" in log_text
    assert f"HF={repo_dir}/.cache/huggingface" in log_text
    assert "ARGS=submit-slurm-plan" in log_text
    assert "--prepare-containers" in log_text


def test_run_slurm_pipeline_shared_root_override_derives_all_paths(tmp_path: Path) -> None:
    source_script = Path("run_slurm_pipeline.sh")
    script_path = tmp_path / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)
    container_image = tmp_path / "images" / "maxionbench.sif"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_log = tmp_path / "maxionbench_env.log"
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'DATASET=%s\\n' "${MAXIONBENCH_DATASET_ROOT:-}"
  printf 'CACHE=%s\\n' "${MAXIONBENCH_DATASET_CACHE_DIR:-}"
  printf 'OUTPUT=%s\\n' "${MAXIONBENCH_OUTPUT_ROOT:-}"
  printf 'FIGURES=%s\\n' "${MAXIONBENCH_FIGURES_ROOT:-}"
  printf 'HF=%s\\n' "${MAXIONBENCH_HF_CACHE_DIR:-}"
} > "${MAXIONBENCH_STUB_LOG}"
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MAXIONBENCH_STUB_LOG"] = str(stub_log)
    env["USER"] = "tester"

    completed = subprocess.run(
        [
            "bash",
            "run_slurm_pipeline.sh",
            "--cluster",
            "nrel",
            "--container-image",
            str(container_image),
            "--shared-root",
            "/projects/demo/maxionbench",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    env_file = tmp_path / ".env.slurm.nrel"
    assert env_file.exists()
    env_text = env_file.read_text(encoding="utf-8")
    assert "MAXIONBENCH_SHARED_ROOT=/projects/demo/maxionbench" in env_text
    assert "MAXIONBENCH_DATASET_ROOT=/projects/demo/maxionbench/dataset" in env_text
    assert "MAXIONBENCH_APPTAINER_CACHE_DIR=/projects/demo/maxionbench/.cache/apptainer" in env_text
    assert "MAXIONBENCH_APPTAINER_TMPDIR=/projects/demo/maxionbench/.cache/apptainer/tmp" in env_text
    log_text = stub_log.read_text(encoding="utf-8")
    assert "DATASET=/projects/demo/maxionbench/dataset" in log_text
    assert "CACHE=/projects/demo/maxionbench/.cache" in log_text
    assert "OUTPUT=/projects/demo/maxionbench/results" in log_text
    assert "FIGURES=/projects/demo/maxionbench/figures" in log_text
    assert "HF=/projects/demo/maxionbench/.cache/huggingface" in log_text


def test_run_slurm_pipeline_auto_loads_cluster_env_file(tmp_path: Path) -> None:
    source_script = Path("run_slurm_pipeline.sh")
    script_path = tmp_path / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)

    submit_root = tmp_path / "submit_root"
    submit_root.mkdir(parents=True, exist_ok=True)
    shared_root = tmp_path / "env_shared_root"
    containers_dir = shared_root / "containers"
    containers_dir.mkdir(parents=True, exist_ok=True)
    for image_name in (
        "maxionbench.sif",
        "qdrant.sif",
        "pgvector.sif",
        "opensearch.sif",
        "weaviate.sif",
        "milvus-etcd.sif",
        "milvus-minio.sif",
        "milvus.sif",
    ):
        (containers_dir / image_name).write_text("image\n", encoding="utf-8")

    (tmp_path / ".env.slurm.nrel").write_text(
        "\n".join(
            (
                f"MAXIONBENCH_SHARED_ROOT={shared_root}",
                "MAXIONBENCH_HF_CACHE_DIR=${MAXIONBENCH_SHARED_ROOT}/.cache/huggingface",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_log = tmp_path / "maxionbench_env.log"
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'DATASET=%s\\n' "${MAXIONBENCH_DATASET_ROOT:-}"
  printf 'CACHE=%s\\n' "${MAXIONBENCH_DATASET_CACHE_DIR:-}"
  printf 'OUTPUT=%s\\n' "${MAXIONBENCH_OUTPUT_ROOT:-}"
  printf 'FIGURES=%s\\n' "${MAXIONBENCH_FIGURES_ROOT:-}"
  printf 'HF=%s\\n' "${MAXIONBENCH_HF_CACHE_DIR:-}"
  printf 'CONTAINER=%s\\n' "${MAXIONBENCH_CONTAINER_IMAGE:-}"
} > "${MAXIONBENCH_STUB_LOG}"
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MAXIONBENCH_STUB_LOG"] = str(stub_log)
    env["USER"] = "tester"
    env.pop("MAXIONBENCH_SHARED_ROOT", None)
    env.pop("MAXIONBENCH_DATASET_ROOT", None)
    env.pop("MAXIONBENCH_DATASET_CACHE_DIR", None)
    env.pop("MAXIONBENCH_OUTPUT_ROOT", None)
    env.pop("MAXIONBENCH_FIGURES_ROOT", None)
    env.pop("MAXIONBENCH_HF_CACHE_DIR", None)
    env.pop("MAXIONBENCH_CONTAINER_IMAGE", None)

    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "--cluster",
            "nrel",
        ],
        cwd=submit_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"+ loaded cluster env {tmp_path / '.env.slurm.nrel'}" in completed.stdout
    assert f"+ wrote cluster env {tmp_path / '.env.slurm.nrel'}" in completed.stdout
    log_text = stub_log.read_text(encoding="utf-8")
    assert f"DATASET={shared_root}/dataset" in log_text
    assert f"CACHE={shared_root}/.cache" in log_text
    assert f"OUTPUT={shared_root}/results" in log_text
    assert f"FIGURES={shared_root}/figures" in log_text
    assert f"HF={shared_root}/.cache/huggingface" in log_text
    assert f"CONTAINER={shared_root}/containers/maxionbench.sif" in log_text


def test_run_slurm_pipeline_cli_overrides_auto_loaded_cluster_env_file(tmp_path: Path) -> None:
    source_script = Path("run_slurm_pipeline.sh")
    script_path = tmp_path / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)

    submit_root = tmp_path / "submit_root"
    submit_root.mkdir(parents=True, exist_ok=True)
    env_shared_root = tmp_path / "env_shared_root"
    cli_shared_root = tmp_path / "cli_shared_root"
    for root in (env_shared_root, cli_shared_root):
        containers_dir = root / "containers"
        containers_dir.mkdir(parents=True, exist_ok=True)
        for image_name in (
            "maxionbench.sif",
            "qdrant.sif",
            "pgvector.sif",
            "opensearch.sif",
            "weaviate.sif",
            "milvus-etcd.sif",
            "milvus-minio.sif",
            "milvus.sif",
        ):
            (containers_dir / image_name).write_text("image\n", encoding="utf-8")

    (tmp_path / ".env.slurm.nrel").write_text(
        f"MAXIONBENCH_SHARED_ROOT={env_shared_root}\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_log = tmp_path / "maxionbench_env.log"
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'DATASET=%s\\n' "${MAXIONBENCH_DATASET_ROOT:-}"
  printf 'CONTAINER=%s\\n' "${MAXIONBENCH_CONTAINER_IMAGE:-}"
} > "${MAXIONBENCH_STUB_LOG}"
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MAXIONBENCH_STUB_LOG"] = str(stub_log)
    env["USER"] = "tester"
    env.pop("MAXIONBENCH_SHARED_ROOT", None)
    env.pop("MAXIONBENCH_CONTAINER_IMAGE", None)

    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "--cluster",
            "nrel",
            "--shared-root",
            str(cli_shared_root),
        ],
        cwd=submit_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    env_text = (tmp_path / ".env.slurm.nrel").read_text(encoding="utf-8")
    assert f"MAXIONBENCH_SHARED_ROOT={cli_shared_root}" in env_text
    assert f"MAXIONBENCH_CONTAINER_IMAGE={cli_shared_root}/containers/maxionbench.sif" in env_text
    assert f"MAXIONBENCH_APPTAINER_CACHE_DIR={cli_shared_root}/.cache/apptainer" in env_text
    assert f"MAXIONBENCH_APPTAINER_TMPDIR={cli_shared_root}/.cache/apptainer/tmp" in env_text
    log_text = stub_log.read_text(encoding="utf-8")
    assert f"DATASET={cli_shared_root}/dataset" in log_text
    assert f"CONTAINER={cli_shared_root}/containers/maxionbench.sif" in log_text


def test_run_slurm_pipeline_falls_back_to_python_module_cli(tmp_path: Path) -> None:
    source_script = Path("run_slurm_pipeline.sh")
    script_path = tmp_path / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)
    container_image = tmp_path / "images" / "maxionbench.sif"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_log = tmp_path / "python_calls.log"
    stub_python = bin_dir / "python"
    stub_python.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\\n' "$*" > "${MAXIONBENCH_STUB_LOG}"
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
    env["MAXIONBENCH_STUB_LOG"] = str(stub_log)

    completed = subprocess.run(
        [
            "/bin/bash",
            "run_slurm_pipeline.sh",
            "--cluster",
            "euler",
            "--container-image",
            str(container_image),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "+ python -m maxionbench.cli submit-slurm-plan" in completed.stdout
    log_text = stub_log.read_text(encoding="utf-8").strip()
    assert log_text.startswith("-m maxionbench.cli submit-slurm-plan ")
    assert "--prepare-containers" in log_text


def test_run_slurm_pipeline_derives_container_paths_from_shared_root(tmp_path: Path) -> None:
    source_script = Path("run_slurm_pipeline.sh")
    script_path = tmp_path / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)

    shared_root = tmp_path / "shared_root"
    containers_dir = shared_root / "containers"
    containers_dir.mkdir(parents=True, exist_ok=True)
    for image_name in (
        "maxionbench.sif",
        "qdrant.sif",
        "pgvector.sif",
        "opensearch.sif",
        "weaviate.sif",
        "milvus-etcd.sif",
        "milvus-minio.sif",
        "milvus.sif",
    ):
        (containers_dir / image_name).write_text("image\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_log = tmp_path / "maxionbench_env.log"
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'CONTAINER=%s\\n' "${MAXIONBENCH_CONTAINER_IMAGE:-}"
  printf 'QDRANT=%s\\n' "${MAXIONBENCH_QDRANT_IMAGE:-}"
  printf 'PGVECTOR=%s\\n' "${MAXIONBENCH_PGVECTOR_IMAGE:-}"
  printf 'OPENSEARCH=%s\\n' "${MAXIONBENCH_OPENSEARCH_IMAGE:-}"
  printf 'WEAVIATE=%s\\n' "${MAXIONBENCH_WEAVIATE_IMAGE:-}"
  printf 'MILVUS_ETCD=%s\\n' "${MAXIONBENCH_MILVUS_ETCD_IMAGE:-}"
  printf 'MILVUS_MINIO=%s\\n' "${MAXIONBENCH_MILVUS_MINIO_IMAGE:-}"
  printf 'MILVUS=%s\\n' "${MAXIONBENCH_MILVUS_IMAGE:-}"
  printf 'ARGS=%s\\n' "$*"
} > "${MAXIONBENCH_STUB_LOG}"
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MAXIONBENCH_STUB_LOG"] = str(stub_log)
    env["USER"] = "tester"
    env.pop("MAXIONBENCH_CONTAINER_IMAGE", None)
    env.pop("MAXIONBENCH_QDRANT_IMAGE", None)
    env.pop("MAXIONBENCH_PGVECTOR_IMAGE", None)
    env.pop("MAXIONBENCH_OPENSEARCH_IMAGE", None)
    env.pop("MAXIONBENCH_WEAVIATE_IMAGE", None)
    env.pop("MAXIONBENCH_MILVUS_ETCD_IMAGE", None)
    env.pop("MAXIONBENCH_MILVUS_MINIO_IMAGE", None)
    env.pop("MAXIONBENCH_MILVUS_IMAGE", None)

    completed = subprocess.run(
        [
            "bash",
            "run_slurm_pipeline.sh",
            "--cluster",
            "nrel",
            "--shared-root",
            str(shared_root),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    log_text = stub_log.read_text(encoding="utf-8")
    assert f"CONTAINER={shared_root}/containers/maxionbench.sif" in log_text
    assert f"QDRANT={shared_root}/containers/qdrant.sif" in log_text
    assert f"PGVECTOR={shared_root}/containers/pgvector.sif" in log_text
    assert f"OPENSEARCH={shared_root}/containers/opensearch.sif" in log_text
    assert f"WEAVIATE={shared_root}/containers/weaviate.sif" in log_text
    assert f"MILVUS_ETCD={shared_root}/containers/milvus-etcd.sif" in log_text
    assert f"MILVUS_MINIO={shared_root}/containers/milvus-minio.sif" in log_text
    assert f"MILVUS={shared_root}/containers/milvus.sif" in log_text
    assert f"--container-image {shared_root}/containers/maxionbench.sif" in log_text


def test_run_slurm_pipeline_launch_submits_cluster_side_prepare_containers_without_local_apptainer(tmp_path: Path) -> None:
    source_script = Path("run_slurm_pipeline.sh")
    script_path = tmp_path / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    build_script_path = scripts_dir / "build_containers.sh"
    build_script_path.write_text(
        Path("scripts/build_containers.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    build_script_path.chmod(0o755)

    slurm_dir = tmp_path / "maxionbench" / "orchestration" / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    prepare_script_path = slurm_dir / "prepare_containers.sh"
    prepare_script_path.write_text(
        Path("maxionbench/orchestration/slurm/prepare_containers.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    prepare_script_path.chmod(0o755)

    definition_path = tmp_path / "maxionbench.def"
    definition_path.write_text(Path("maxionbench.def").read_text(encoding="utf-8"), encoding="utf-8")

    (tmp_path / ".env.slurm.nrel").write_text(
        "MAXIONBENCH_SLURM_ACCOUNT=nawimem\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_log = tmp_path / "maxionbench_env.log"
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'DATASET=%s\\n' "${MAXIONBENCH_DATASET_ROOT:-}"
  printf 'CONTAINER=%s\\n' "${MAXIONBENCH_CONTAINER_IMAGE:-}"
  printf 'ARGS=%s\\n' "$*"
} > "${MAXIONBENCH_STUB_LOG}"
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
    env["MAXIONBENCH_STUB_LOG"] = str(stub_log)
    env["USER"] = "tester"

    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "--cluster",
            "nrel",
            "--launch",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "error: apptainer is required on the login/build node" not in completed.stderr
    assert "+ missing images will be prepared by the cluster-side prepare_containers job" in completed.stderr
    log_text = stub_log.read_text(encoding="utf-8")
    assert f"DATASET={tmp_path}/dataset" in log_text
    assert f"CONTAINER={tmp_path}/containers/maxionbench.sif" in log_text
    assert "ARGS=submit-slurm-plan" in log_text
    assert "--prepare-containers" in log_text


def test_run_slurm_pipeline_launch_rejects_missing_prepare_containers_script(tmp_path: Path) -> None:
    source_script = Path("run_slurm_pipeline.sh")
    script_path = tmp_path / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    build_script_path = scripts_dir / "build_containers.sh"
    build_script_path.write_text(
        Path("scripts/build_containers.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    build_script_path.chmod(0o755)

    definition_path = tmp_path / "maxionbench.def"
    definition_path.write_text(Path("maxionbench.def").read_text(encoding="utf-8"), encoding="utf-8")

    (tmp_path / ".env.slurm.nrel").write_text(
        "MAXIONBENCH_SLURM_ACCOUNT=nawimem\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["USER"] = "tester"

    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "--cluster",
            "nrel",
            "--launch",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "missing Slurm container prep script" in completed.stderr


def test_run_slurm_pipeline_rejects_placeholder_cluster_defaults(tmp_path: Path) -> None:
    source_script = Path("run_slurm_pipeline.sh")
    script_path = tmp_path / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)

    (tmp_path / ".env.slurm.nrel").write_text(
        "MAXIONBENCH_SLURM_ACCOUNT=your-account\n",
        encoding="utf-8",
    )

    shared_root = tmp_path / "shared_root"
    containers_dir = shared_root / "containers"
    containers_dir.mkdir(parents=True, exist_ok=True)
    for image_name in (
        "maxionbench.sif",
        "qdrant.sif",
        "pgvector.sif",
        "opensearch.sif",
        "weaviate.sif",
        "milvus-etcd.sif",
        "milvus-minio.sif",
        "milvus.sif",
    ):
        (containers_dir / image_name).write_text("image\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "--cluster",
            "nrel",
            "--shared-root",
            str(shared_root),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "placeholder cluster-local value detected: your-account" in completed.stderr
