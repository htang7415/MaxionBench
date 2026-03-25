from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from maxionbench.orchestration.slurm import preflight as preflight_mod
from maxionbench.orchestration.slurm.preflight import evaluate_preflight
from maxionbench.runtime.ports import allocate_named_ports, allocate_port, allocate_port_range


def test_allocate_port_range_and_named_ports() -> None:
    ports = allocate_port_range(count=3, base=25000, span=1000, offset=10)
    assert len(ports) == 3
    assert ports[1] == ports[0] + 1

    named = allocate_named_ports(["a", "b", "a"], base=26000, span=1000)
    assert set(named.keys()) == {"a", "b"}
    assert named["b"] == named["a"] + 1


def test_allocate_port_uses_array_task_id(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    first = allocate_port(base=27000, span=1000)

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    second = allocate_port(base=27000, span=1000)

    assert first != second
    assert second == first + 1


def test_preflight_uses_manifest_when_available(tmp_path: Path) -> None:
    cfg = {
        "dataset_bundle": "D3",
        "num_vectors": 1000,
        "vector_dim": 16,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["dataset_bundle"] == "D3"
    assert summary["manifest_ok"] is True
    assert summary["manifest_error"] is None
    assert summary["dataset_bytes"] > 0
    assert summary["fallback_config"] == "configs/scenarios/s2_filtered_ann.yaml"


def test_preflight_estimate_without_manifest(tmp_path: Path) -> None:
    cfg = {
        "dataset_bundle": "UNKNOWN",
        "num_vectors": 100,
        "vector_dim": 8,
    }
    cfg_path = tmp_path / "cfg_unknown.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["dataset_bundle"] == "UNKNOWN"
    assert summary["manifest_ok"] is True
    assert summary["manifest_error"] is None
    assert summary["dataset_bytes"] > 0
    assert summary["engine_bytes"] > 0
    assert summary["temp_bytes"] > 0


def test_preflight_verifies_dataset_cache_checksum_when_provided(tmp_path: Path) -> None:
    dataset_file = tmp_path / "sample.bin"
    dataset_file.write_bytes(b"cache-check")
    checksum = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    cfg = {
        "dataset_bundle": "UNKNOWN",
        "dataset_path": str(dataset_file),
        "dataset_path_sha256": checksum,
        "num_vectors": 100,
        "vector_dim": 8,
    }
    cfg_path = tmp_path / "cfg_checksum_ok.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["integrity_ok"] is True
    assert int(summary["integrity_error_count"]) == 0
    assert int(summary["integrity_checked_files"]) == 1


def test_preflight_fails_when_dataset_cache_checksum_mismatches(tmp_path: Path) -> None:
    dataset_file = tmp_path / "sample_bad.bin"
    dataset_file.write_bytes(b"cache-check-bad")
    cfg = {
        "dataset_bundle": "UNKNOWN",
        "dataset_path": str(dataset_file),
        "dataset_path_sha256": ("0" * 64),
        "num_vectors": 100,
        "vector_dim": 8,
    }
    cfg_path = tmp_path / "cfg_checksum_bad.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["integrity_ok"] is False
    assert int(summary["integrity_error_count"]) >= 1
    assert summary["ok"] is False
    assert any("sha256 mismatch" in msg for msg in summary["integrity_errors"])


def test_preflight_fails_when_known_bundle_manifest_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    cfg = {
        "dataset_bundle": "D3",
        "num_vectors": 1000,
        "vector_dim": 16,
    }
    cfg_path = tmp_path / "cfg_known_bundle_missing_manifest.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    monkeypatch.setattr(preflight_mod, "_load_manifest", lambda _: None)
    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["manifest_ok"] is False
    assert "missing manifest" in str(summary["manifest_error"])
    assert summary["ok"] is False


def test_preflight_fails_when_known_bundle_manifest_has_invalid_size_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    cfg = {
        "dataset_bundle": "D3",
        "num_vectors": 1000,
        "vector_dim": 16,
    }
    cfg_path = tmp_path / "cfg_known_bundle_bad_manifest.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    monkeypatch.setattr(
        preflight_mod,
        "_load_manifest",
        lambda _: {
            "dataset_bundle": "D3",
            "approx_bytes_dataset": 0,
            "approx_bytes_engine": 100,
            "approx_bytes_temp": 10,
        },
    )
    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["manifest_ok"] is False
    assert "must be > 0" in str(summary["manifest_error"])
    assert summary["ok"] is False


def test_slurm_common_runs_pre_run_gate_before_runner() -> None:
    text = Path("maxionbench/orchestration/slurm/common.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SKIP_PRE_RUN_GATE" in text
    assert "MAXIONBENCH_ALLOW_GPU_UNAVAILABLE" in text
    assert "MAXIONBENCH_CONFORMANCE_MATRIX" in text
    assert "MAXIONBENCH_OUTPUT_ROOT" in text
    assert "MAXIONBENCH_DATASET_ROOT" in text
    assert "MAXIONBENCH_DATASET_CACHE_DIR" in text
    assert "MAXIONBENCH_FIGURES_ROOT" in text
    assert "MAXIONBENCH_D3_PARAMS_PATH" in text
    assert "MAXIONBENCH_SLURM_RUN_MANIFEST" in text
    assert "MAXIONBENCH_CONFORMANCE_TIMEOUT_S" in text
    assert "MAXIONBENCH_SERVICE_START_GRACE_S" in text
    assert "MAXIONBENCH_SERVICE_START_VERIFY_TIMEOUT_S" in text
    assert "MAXIONBENCH_SERVICE_START_POLL_S" in text
    assert "MAXIONBENCH_SERVICE_LOG_TAIL_LINES" in text
    assert "MAXIONBENCH_PGVECTOR_BIN_DIR" in text
    assert "MAXIONBENCH_CONTAINER_RUNTIME" in text
    assert "MAXIONBENCH_CONTAINER_IMAGE" in text
    assert "MAXIONBENCH_CONTAINER_BIND" in text
    assert "MAXIONBENCH_HF_CACHE_DIR" in text
    assert "MAXIONBENCH_APPTAINER_MODULE" in text
    assert "MAXIONBENCH_MODULE_INIT_SH" in text
    assert "MAXIONBENCH_CLEANUP_LOCAL_SCRATCH" in text
    assert "MAXIONBENCH_DATASET_ENV_SH" in text
    assert "MAXIONBENCH_QDRANT_IMAGE" in text
    assert "MAXIONBENCH_PGVECTOR_IMAGE" in text
    assert "MAXIONBENCH_OPENSEARCH_IMAGE" in text
    assert "MAXIONBENCH_WEAVIATE_IMAGE" in text
    assert "MAXIONBENCH_MILVUS_ETCD_IMAGE" in text
    assert "MAXIONBENCH_MILVUS_MINIO_IMAGE" in text
    assert "MAXIONBENCH_MILVUS_IMAGE" in text
    assert "mb_source_dataset_env()" in text
    assert "mb_require_dataset_env_contract()" in text
    assert "mb_require_gpu_fail_fast()" in text
    assert "mb_require_visible_gpu()" in text
    assert "mb_ensure_apptainer()" in text
    assert "module load" in text
    assert "apptainer exec --cleanenv" in text
    assert "--env" in text
    assert "PYTHONNOUSERSITE=1" in text
    assert "python -s" in text
    assert "mb_python()" in text
    assert "mb_cleanup_local_runtime()" in text
    assert "expand_env_placeholders" in text
    assert "mb_read_config_field()" in text
    gate_marker = "pre-run-gate"
    runner_marker = "python -m maxionbench.orchestration.runner"
    assert gate_marker in text
    assert runner_marker in text
    assert text.index(gate_marker) < text.index(runner_marker)


def test_slurm_common_uses_longer_default_startup_timeout_for_milvus() -> None:
    text = Path("maxionbench/orchestration/slurm/common.sh").read_text(encoding="utf-8")

    assert "MAXIONBENCH_SERVICE_START_VERIFY_TIMEOUT_MILVUS_S" in text
    assert 'if [[ "${name}" == "milvus" ]]; then' in text
    assert 'verify_timeout_s="${MAXIONBENCH_SERVICE_START_VERIFY_TIMEOUT_MILVUS_S:-120}"' in text


def test_slurm_common_revalidates_fallback_config_after_preflight_failure() -> None:
    text = Path("maxionbench/orchestration/slurm/common.sh").read_text(encoding="utf-8")
    assert 'mb_log "scratch preflight failed, validating fallback config ${resolved_fallback}"' in text
    assert 'if mb_run_scratch_preflight "${resolved_fallback}"; then' in text
    assert 'mb_log "fallback config ${resolved_fallback} also failed scratch preflight"' in text


def test_slurm_common_has_managed_engine_service_lifecycle_helpers() -> None:
    text = Path("maxionbench/orchestration/slurm/common.sh").read_text(encoding="utf-8")
    assert "mb_detect_engine_runtime_mode()" in text
    assert "mb_engine_requires_service()" in text
    assert "mb_start_engine_services()" in text
    assert "mb_stop_engine_services()" in text
    assert "mb_wait_engine_health()" in text
    assert "mb_start_qdrant_service()" in text
    assert "mb_start_pgvector_service()" in text
    assert "mb_start_opensearch_service()" in text
    assert "mb_start_weaviate_service()" in text
    assert "mb_start_milvus_services()" in text
    assert "mb_start_apptainer_service_process()" in text
    assert "mb_start_apptainer_service_argv_process()" in text
    assert "mb_assert_service_internal_port_contract()" in text
    assert "mb_assert_service_runtime_bind_contract()" in text
    assert "mb_service_startup_adapter_options_json()" in text
    assert "mb_verify_managed_service_startup()" in text
    assert "mb_validate_apptainer_service_probe()" in text
    assert "mb_validate_opensearch_service_image()" in text
    assert "mb_validate_named_service_image()" in text
    assert "mb_wait_named_adapter_health_timeout()" in text
    assert "mb_wait_named_adapter_health()" in text
    assert "mb_wait_http_health_timeout()" in text
    assert "mb_service_startup_http_url" in text
    assert "mb_seed_milvus_config_dir()" in text
    assert "mb_write_milvus_runtime_config()" in text
    assert "mb_capture_local_diagnostics()" in text
    assert "mb_finalize_job()" in text
    assert 'MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI="${SLURM_TMPDIR}/lancedb/service"' in text
    assert "MAXIONBENCH_PORT_WEAVIATE_GOSSIP" in text
    assert "MAXIONBENCH_PORT_WEAVIATE_DATA" in text
    assert "CLUSTER_GOSSIP_BIND_PORT=" in text
    assert "CLUSTER_DATA_BIND_PORT=" in text
    assert "RAFT_BOOTSTRAP_EXPECT=1" in text
    assert "RAFT_BOOTSTRAP_TIMEOUT=90" in text
    assert "MAXIONBENCH_PGVECTOR_DSN=" in text
    assert "apptainer inspect" in text
    assert "command -v" in text
    assert "mb_log_file_tail" in text


def test_slurm_service_contracts_define_startup_metadata_and_runtime_paths() -> None:
    service_contracts_path = Path("maxionbench/orchestration/slurm/service_contracts.sh").resolve()
    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{service_contracts_path}"; '
                'export MAXIONBENCH_PORT_MILVUS_ETCD="12379"; '
                'export MAXIONBENCH_PORT_MILVUS_MINIO="19000"; '
                "for service in qdrant pgvector opensearch weaviate milvus milvus-etcd milvus-minio; do "
                'kind="$(mb_service_contract_kind "${service}")"; '
                'start_mode="$(mb_service_default_start_mode "${service}")"; '
                'verify_mode="$(mb_service_startup_verify_mode "${service}")"; '
                'adapter_name="$(mb_service_startup_adapter_name "${service}")"; '
                'mb_service_runtime_writable_paths "${service}" writable_paths; '
                'mb_service_runtime_seed_paths "${service}" seeded_paths; '
                'mb_service_internal_port_envs "${service}" internal_port_envs; '
                'printf "SERVICE|%s|%s|%s|%s|%s|%s|%s|%s\\n" '
                '"${service}" "${kind}" "${start_mode}" "${verify_mode}" "${adapter_name}" "${#writable_paths[@]}" "${#seeded_paths[@]}" "${#internal_port_envs[@]}"; '
                'if [[ "${kind}" == "probe" ]]; then '
                '  mb_service_probe_args "${service}" probe_args; '
                '  printf "PROBE|%s|%s\\n" "${service}" "${#probe_args[@]}"; '
                "fi; "
                'if [[ "${verify_mode}" == "http" ]]; then '
                '  health_url="$(mb_service_startup_http_url "${service}")"; '
                '  printf "HTTP|%s|%s\\n" "${service}" "${health_url}"; '
                "fi; "
                "done"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    service_rows: dict[str, dict[str, str]] = {}
    probe_rows: dict[str, int] = {}
    http_rows: dict[str, str] = {}
    for raw_line in completed.stdout.splitlines():
        parts = raw_line.split("|")
        if parts[0] == "SERVICE":
            service_rows[parts[1]] = {
                "kind": parts[2],
                "start_mode": parts[3],
                "verify_mode": parts[4],
                "adapter_name": parts[5],
                "writable_count": parts[6],
                "seed_count": parts[7],
                "internal_port_count": parts[8],
            }
        elif parts[0] == "PROBE":
            probe_rows[parts[1]] = int(parts[2])
        elif parts[0] == "HTTP":
            http_rows[parts[1]] = parts[2]

    expected_verify_modes = {
        "qdrant": "health",
        "pgvector": "health",
        "opensearch": "health",
        "weaviate": "health",
        "milvus": "health",
        "milvus-etcd": "http",
        "milvus-minio": "http",
    }
    for service_name, verify_mode in expected_verify_modes.items():
        row = service_rows[service_name]
        assert row["start_mode"] in {"shell", "argv"}
        assert row["verify_mode"] == verify_mode
        assert int(row["writable_count"]) >= 1
        if verify_mode == "health":
            assert row["adapter_name"] == service_name
        else:
            assert row["adapter_name"] == ""

    assert service_rows["qdrant"]["kind"] == "qdrant-layout"
    assert service_rows["pgvector"]["kind"] == "pgvector"
    assert service_rows["opensearch"]["kind"] == "opensearch-layout"
    assert int(service_rows["opensearch"]["seed_count"]) == 1
    assert int(service_rows["milvus"]["seed_count"]) == 1
    assert int(service_rows["qdrant"]["seed_count"]) == 0
    assert int(service_rows["weaviate"]["seed_count"]) == 0
    assert int(service_rows["weaviate"]["internal_port_count"]) == 2
    assert int(service_rows["milvus"]["internal_port_count"]) == 3
    assert int(service_rows["qdrant"]["internal_port_count"]) == 0
    assert int(service_rows["opensearch"]["internal_port_count"]) == 0
    assert probe_rows == {
        "weaviate": 2,
        "milvus": 2,
        "milvus-etcd": 2,
        "milvus-minio": 2,
    }
    assert http_rows == {
        "milvus-etcd": "http://127.0.0.1:12379/readyz",
        "milvus-minio": "http://127.0.0.1:19000/minio/health/live",
    }


def test_slurm_service_contracts_weaviate_start_args_include_http_scheme() -> None:
    service_contracts_path = Path("maxionbench/orchestration/slurm/service_contracts.sh").resolve()
    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{service_contracts_path}"; '
                'export MAXIONBENCH_PORT_WEAVIATE="28080"; '
                'mb_service_default_start_args "weaviate" start_args; '
                'printf "%s\\n" "${start_args[@]}"'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.splitlines() == [
        "weaviate",
        "--scheme",
        "http",
        "--host",
        "0.0.0.0",
        "--port",
        "28080",
    ]


def test_slurm_common_health_startup_verifier_uses_adapter_metadata(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    slurm_tmpdir = tmp_path / "slurm_tmp"
    adapter_log = tmp_path / "startup_adapter.jsonl"
    service_log = tmp_path / "service.log"
    service_log.write_text("service still booting\n", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_wait_named_adapter_health_timeout() { '
                f'  printf "%s\\n%s\\n%s\\n" "$1" "$2" "$3" > "{adapter_log}"; '
                '  return 0; '
                '}; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'sleep 30 & '
                'pid=$!; '
                f'mb_verify_managed_service_startup "qdrant" "${{pid}}" "{service_log}"; '
                'mb_stop_engine_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    adapter_name, adapter_options_json, timeout_s = adapter_log.read_text(encoding="utf-8").splitlines()
    assert adapter_name == "qdrant"
    assert timeout_s == "30"
    adapter_options = json.loads(adapter_options_json)
    assert adapter_options["host"] == "127.0.0.1"
    assert isinstance(adapter_options["port"], int)
    assert set(adapter_options.keys()) == {"host", "port"}


def test_slurm_common_require_tmpdir_fails_when_unset() -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                'unset SLURM_TMPDIR; '
                'mb_require_tmpdir'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode != 0
    assert "SLURM_TMPDIR must be set to node-local scratch" in completed.stdout + completed.stderr


def test_slurm_common_allocate_ports_exports_weaviate_internal_ports(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    slurm_tmpdir = tmp_path / "slurm_tmp"

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'printf "HTTP=%s\\n" "${MAXIONBENCH_PORT_WEAVIATE}"; '
                'printf "GOSSIP=%s\\n" "${MAXIONBENCH_PORT_WEAVIATE_GOSSIP}"; '
                'printf "DATA=%s\\n" "${MAXIONBENCH_PORT_WEAVIATE_DATA}"; '
                'printf "GOSSIP_ALIAS=%s\\n" "${MAXIONBENCH_WEAVIATE_GOSSIP_PORT}"; '
                'printf "DATA_ALIAS=%s\\n" "${MAXIONBENCH_WEAVIATE_DATA_PORT}"'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = dict(
        line.split("=", maxsplit=1)
        for line in completed.stdout.splitlines()
        if "=" in line
    )
    assert payload["GOSSIP"] == payload["GOSSIP_ALIAS"]
    assert payload["DATA"] == payload["DATA_ALIAS"]
    allocated_ports = {int(payload["HTTP"]), int(payload["GOSSIP"]), int(payload["DATA"])}
    assert len(allocated_ports) == 3
    assert int(payload["HTTP"]) != 8300
    assert int(payload["GOSSIP"]) != 8300
    assert int(payload["DATA"]) != 8301


def test_slurm_common_allocate_ports_exports_milvus_internal_ports(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    slurm_tmpdir = tmp_path / "slurm_tmp"

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'printf "PROXY=%s\\n" "${MAXIONBENCH_PORT_MILVUS}"; '
                'printf "ROOTCOORD=%s\\n" "${MAXIONBENCH_PORT_MILVUS_ROOTCOORD}"; '
                'printf "DATACOORD=%s\\n" "${MAXIONBENCH_PORT_MILVUS_DATACOORD}"; '
                'printf "QUERYCOORD=%s\\n" "${MAXIONBENCH_PORT_MILVUS_QUERYCOORD}"'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = dict(
        line.split("=", maxsplit=1)
        for line in completed.stdout.splitlines()
        if "=" in line
    )
    allocated_ports = {
        int(payload["PROXY"]),
        int(payload["ROOTCOORD"]),
        int(payload["DATACOORD"]),
        int(payload["QUERYCOORD"]),
    }
    assert len(allocated_ports) == 4
    assert int(payload["PROXY"]) != 19530
    assert int(payload["ROOTCOORD"]) != 53100
    assert int(payload["DATACOORD"]) != 13333
    assert int(payload["QUERYCOORD"]) != 19531


def test_slurm_common_startup_http_verifier_uses_service_contract_url(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    slurm_tmpdir = tmp_path / "slurm_tmp"
    http_log = tmp_path / "startup_http.txt"
    service_log = tmp_path / "service.log"
    service_log.write_text("service still booting\n", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_wait_http_health_timeout() { '
                f'  printf "%s\\n%s\\n" "$1" "$2" > "{http_log}"; '
                '  return 0; '
                '}; '
                'mb_require_tmpdir; '
                'export MAXIONBENCH_PORT_MILVUS_ETCD="12379"; '
                'sleep 30 & '
                'pid=$!; '
                f'mb_verify_managed_service_startup "milvus-etcd" "${{pid}}" "{service_log}"; '
                'mb_stop_engine_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    health_url, timeout_s = http_log.read_text(encoding="utf-8").splitlines()
    assert health_url == "http://127.0.0.1:12379/readyz"
    assert timeout_s == "30"


def test_slurm_common_loads_apptainer_module_when_binary_missing(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_log = tmp_path / "apptainer.log"
    fake_image = tmp_path / "maxionbench.sif"
    fake_image.write_text("image\n", encoding="utf-8")

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" > "${MAXIONBENCH_TEST_APPTAINER_LOG}"
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    module_init = tmp_path / "modules.sh"
    module_init.write_text(
        f"""module() {{
  if [[ "${{1:-}}" == "load" && "${{2:-}}" == "apptainer" ]]; then
    export PATH="{fake_bin_dir}:$PATH"
    return 0
  fi
  return 1
}}
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                'export PATH="/usr/bin:/bin"; '
                f'export MAXIONBENCH_CONTAINER_IMAGE="{fake_image}"; '
                'export MAXIONBENCH_CONTAINER_RUNTIME="apptainer"; '
                'export MAXIONBENCH_APPTAINER_MODULE="apptainer"; '
                f'export MAXIONBENCH_MODULE_INIT_SH="{module_init}"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{fake_log}"; '
                'mb_python -V'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "apptainer not found in PATH; attempting module bootstrap with apptainer" in completed.stderr
    assert f"sourced module init {module_init}" in completed.stderr
    assert "loading apptainer module apptainer" in completed.stderr
    assert f"using apptainer binary {fake_apptainer}" in completed.stderr
    logged_args = fake_log.read_text(encoding="utf-8")
    assert "exec" in logged_args
    assert "--cleanenv" in logged_args
    assert str(fake_image) in logged_args
    assert "PYTHONNOUSERSITE=1" in logged_args
    assert "python -s -V" in logged_args


def test_stage_config_command_substitution_stays_clean_with_apptainer(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_log = tmp_path / "apptainer_stage.log"
    fake_image = tmp_path / "maxionbench.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    slurm_tmpdir = tmp_path / "slurm_tmp"
    dataset_source = tmp_path / "real_d3.npy"
    dataset_source.write_bytes(b"vectors\n")
    config_path = tmp_path / "stage_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "scenario": "cleanup_probe",
                "dataset_bundle": "D3",
                "dataset_path": "${MAXIONBENCH_D3_DATASET_PATH}",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" > "${MAXIONBENCH_TEST_APPTAINER_LOG}"
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -ge ${#args[@]} ]]; then
  exit 0
fi
index=$((index + 1))
exec "${args[@]:${index}}"
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_CONTAINER_IMAGE="{fake_image}"; '
                'export MAXIONBENCH_CONTAINER_RUNTIME="apptainer"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{fake_log}"; '
                f'export MAXIONBENCH_D3_DATASET_PATH="{dataset_source}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                f'STAGED_CONFIG="$(mb_stage_config_to_tmp "{config_path}")"; '
                'printf "STAGED=%s\\n" "${STAGED_CONFIG}"; '
                'test -f "${STAGED_CONFIG}"'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    stdout_lines = dict(
        line.split("=", maxsplit=1)
        for line in completed.stdout.splitlines()
        if "=" in line
    )
    staged_path = Path(stdout_lines["STAGED"])
    assert staged_path == slurm_tmpdir / "maxionbench_stage" / "4242_7" / "config.yaml"
    assert staged_path.exists()
    assert "[maxionbench]" not in stdout_lines["STAGED"]
    staged_payload = yaml.safe_load(staged_path.read_text(encoding="utf-8"))
    assert staged_payload["dataset_path"] == str(
        slurm_tmpdir / "maxionbench_stage" / "4242_7" / "datasets" / "dataset" / dataset_source.name
    )
    assert Path(staged_payload["dataset_path"]).exists()
    assert f"using apptainer binary {fake_apptainer}" in completed.stderr


def test_slurm_common_passes_service_env_inside_apptainer_exec(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_image = tmp_path / "qdrant.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    full_log = tmp_path / "apptainer_service.log"
    post_image_log = tmp_path / "apptainer_service_post_image.log"
    slurm_tmpdir = tmp_path / "slurm_tmp"

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MAXIONBENCH_TEST_APPTAINER_LOG}"
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  index=$((index + 1))
fi
printf '%s\\n' "${args[@]:${index}}" > "${MAXIONBENCH_TEST_POST_IMAGE_LOG}"
if [[ "${args[${index}]:-}" == "/bin/sh" && "${args[$((index + 1))]:-}" == "-c" && "${args[$((index + 2))]:-}" == "[ -x /qdrant/entrypoint.sh ] && [ -x /qdrant/qdrant ]" ]]; then
  exit 0
fi
sleep 30
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_QDRANT_IMAGE="{fake_image}"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{full_log}"; '
                f'export MAXIONBENCH_TEST_POST_IMAGE_LOG="{post_image_log}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_wait_named_adapter_health_timeout() { sleep 0.1; return 0; }; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_qdrant_service; '
                'mb_stop_engine_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    post_image_args = post_image_log.read_text(encoding="utf-8")
    assert post_image_args.startswith("/bin/sh\n")
    assert "/bin/sh" in post_image_args
    assert "\n-c\n" in post_image_args
    assert "-lc" not in post_image_args
    assert "cd /qdrant && exec ./entrypoint.sh" in post_image_args
    apptainer_calls = full_log.read_text(encoding="utf-8")
    assert f"inspect {fake_image}" in apptainer_calls
    assert "--env QDRANT__SERVICE__HOST=0.0.0.0" in apptainer_calls
    assert "QDRANT__SERVICE__HTTP_PORT=" in apptainer_calls
    assert "QDRANT__SERVICE__GRPC_PORT=" in apptainer_calls
    assert "QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage" in apptainer_calls
    assert "QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/storage/snapshots" in apptainer_calls
    assert "/bin/sh -c [ -x /qdrant/entrypoint.sh ] && [ -x /qdrant/qdrant ]" in apptainer_calls
    runtime_root = slurm_tmpdir / "maxionbench_engine_runtime" / "4242_7" / "qdrant"
    assert (runtime_root / "storage" / "snapshots").is_dir()


def test_slurm_common_pgvector_service_uses_writable_runtime_dirs_and_pg_bin_path(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_image = tmp_path / "pgvector.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    apptainer_log = tmp_path / "apptainer_pgvector.log"
    post_image_log = tmp_path / "apptainer_pgvector_post_image.log"
    slurm_tmpdir = tmp_path / "slurm_tmp"

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MAXIONBENCH_TEST_APPTAINER_LOG}"
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  index=$((index + 1))
fi
printf '%s\\n' "${args[@]:${index}}" > "${MAXIONBENCH_TEST_POST_IMAGE_LOG}"
post_args="${args[*]:${index}}"
if [[ "${post_args}" == *"/bin/sh -c"* && "${post_args}" == *"command -v docker-entrypoint.sh"* ]]; then
  exit 0
fi
if [[ "${post_args}" == *"/bin/sh -c"* && "${post_args}" == *"command -v postgres"* && "${post_args}" == *"command -v initdb"* ]]; then
  exit 0
fi
sleep 30
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_PGVECTOR_IMAGE="{fake_image}"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{apptainer_log}"; '
                f'export MAXIONBENCH_TEST_POST_IMAGE_LOG="{post_image_log}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_wait_named_adapter_health_timeout() { sleep 0.1; return 0; }; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_pgvector_service; '
                'mb_stop_engine_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    apptainer_calls = apptainer_log.read_text(encoding="utf-8")
    post_image_args = post_image_log.read_text(encoding="utf-8")
    runtime_root = slurm_tmpdir / "maxionbench_engine_runtime" / "4242_7"
    assert f"--bind {runtime_root / 'pgvector' / 'data'}:/var/lib/postgresql/data" in apptainer_calls
    assert f"--bind {runtime_root / 'pgvector' / 'run'}:/var/run/postgresql" in apptainer_calls
    assert "--env PATH=/usr/lib/postgresql/16/bin:" in apptainer_calls
    assert "/bin/sh -c command -v docker-entrypoint.sh" in apptainer_calls
    assert "/bin/sh -c command -v postgres" in apptainer_calls
    assert "/bin/sh -lc" not in apptainer_calls
    assert post_image_args.startswith("docker-entrypoint.sh\n")
    assert "\n-p\n" in post_image_args
    assert "/bin/sh" not in post_image_args


def test_slurm_common_pgvector_start_cmd_override_uses_shell_launcher(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_image = tmp_path / "pgvector.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    apptainer_log = tmp_path / "apptainer_pgvector_override.log"
    post_image_log = tmp_path / "apptainer_pgvector_override_post_image.log"
    slurm_tmpdir = tmp_path / "slurm_tmp"

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MAXIONBENCH_TEST_APPTAINER_LOG}"
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  index=$((index + 1))
fi
printf '%s\\n' "${args[@]:${index}}" > "${MAXIONBENCH_TEST_POST_IMAGE_LOG}"
post_args="${args[*]:${index}}"
if [[ "${post_args}" == *"/bin/sh -c"* && "${post_args}" == *"command -v docker-entrypoint.sh"* ]]; then
  exit 0
fi
if [[ "${post_args}" == *"/bin/sh -c"* && "${post_args}" == *"command -v postgres"* && "${post_args}" == *"command -v initdb"* ]]; then
  exit 0
fi
if [[ "${post_args}" == *"/bin/sh -c"* && "${post_args}" == *"exec custom-pgvector-start --flag"* ]]; then
  sleep 30
  exit 0
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_PGVECTOR_IMAGE="{fake_image}"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{apptainer_log}"; '
                f'export MAXIONBENCH_TEST_POST_IMAGE_LOG="{post_image_log}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export MAXIONBENCH_PGVECTOR_START_CMD="custom-pgvector-start --flag"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_wait_named_adapter_health_timeout() { sleep 0.1; return 0; }; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_pgvector_service; '
                'mb_stop_engine_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    post_image_args = post_image_log.read_text(encoding="utf-8")
    assert post_image_args.startswith("/bin/sh\n")
    assert "/bin/sh" in post_image_args
    assert "\n-c\n" in post_image_args
    assert "exec custom-pgvector-start --flag" in post_image_args
    assert "--env PATH=/usr/lib/postgresql/16/bin:" in apptainer_log.read_text(encoding="utf-8")


def test_slurm_common_opensearch_uses_layout_validation_and_writable_config(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_image = tmp_path / "opensearch.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    apptainer_log = tmp_path / "apptainer_opensearch.log"
    post_image_log = tmp_path / "apptainer_opensearch_post_image.log"
    slurm_tmpdir = tmp_path / "slurm_tmp"

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MAXIONBENCH_TEST_APPTAINER_LOG}"
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
bind_specs=()
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      bind_specs+=("${args[$((index + 1))]}")
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  index=$((index + 1))
fi
printf '%s\\n' "${args[@]:${index}}" > "${MAXIONBENCH_TEST_POST_IMAGE_LOG}"
post_args="${args[*]:${index}}"
if [[ "${post_args}" == *"/bin/sh -c"* && "${post_args}" == *"[ -x /usr/share/opensearch/bin/opensearch ]"* && "${post_args}" == *"[ -x /usr/share/opensearch/opensearch-docker-entrypoint.sh ]"* && "${post_args}" == *"[ -x /usr/share/opensearch/jdk/bin/java ]"* ]]; then
  exit 0
fi
if [[ "${post_args}" == *"opensearch --version"* ]]; then
  exit 1
fi
if [[ "${post_args}" == *"cp -R /usr/share/opensearch/config/. "* ]]; then
  last_index=$((${#args[@]} - 1))
  target="${args[${last_index}]}"
  mkdir -p "${target}"
  printf '%s\\n' "# seeded jvm options" > "${target}/jvm.options"
  printf '%s\\n' "# seeded log4j config" > "${target}/log4j2.properties"
  exit 0
fi
if [[ "${post_args}" == *"/bin/sh -c"* && "${post_args}" == *"cd /usr/share/opensearch && exec ./opensearch-docker-entrypoint.sh opensearch"* ]]; then
  config_bind=""
  for bind_spec in "${bind_specs[@]}"; do
    if [[ "${bind_spec}" == *":/usr/share/opensearch/config" ]]; then
      config_bind="${bind_spec%%:/usr/share/opensearch/config}"
      break
    fi
  done
  if [[ -z "${config_bind}" || ! -d "${config_bind}" ]]; then
    echo "Likely root cause: java.nio.file.FileSystemException: /usr/share/opensearch/config/opensearch.keystore.tmp: Read-only file system" >&2
    exit 1
  fi
  touch "${config_bind}/opensearch.keystore.tmp"
  sleep 30
  exit 0
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_OPENSEARCH_IMAGE="{fake_image}"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{apptainer_log}"; '
                f'export MAXIONBENCH_TEST_POST_IMAGE_LOG="{post_image_log}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_wait_named_adapter_health_timeout() { sleep 0.1; return 0; }; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_opensearch_service; '
                'mb_stop_engine_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    apptainer_calls = apptainer_log.read_text(encoding="utf-8")
    post_image_args = post_image_log.read_text(encoding="utf-8")
    runtime_root = slurm_tmpdir / "maxionbench_engine_runtime" / "4242_7"
    config_dir = runtime_root / "opensearch" / "config"
    config_path = runtime_root / "opensearch" / "config" / "opensearch.yml"
    config_text = config_path.read_text(encoding="utf-8")
    assert f"inspect {fake_image}" in apptainer_calls
    assert "/bin/sh -c [ -x /usr/share/opensearch/bin/opensearch ] && [ -x /usr/share/opensearch/opensearch-docker-entrypoint.sh ] && [ -x /usr/share/opensearch/jdk/bin/java ]" in apptainer_calls
    assert "opensearch --version" not in apptainer_calls
    assert f"--bind {runtime_root / 'opensearch' / 'data'}:/usr/share/opensearch/data" in apptainer_calls
    assert f"--bind {runtime_root / 'opensearch' / 'logs'}:/usr/share/opensearch/logs" in apptainer_calls
    assert f"--bind {config_dir}:/usr/share/opensearch/config" in apptainer_calls
    assert "--env DISABLE_SECURITY_PLUGIN=true" in apptainer_calls
    assert "OPENSEARCH_JAVA_OPTS=" in apptainer_calls
    assert post_image_args.startswith("/bin/sh\n")
    assert "/bin/sh" in post_image_args
    assert "\n-c\n" in post_image_args
    assert "-lc" not in post_image_args
    assert "cd /usr/share/opensearch && exec ./opensearch-docker-entrypoint.sh opensearch" in post_image_args
    assert "path.logs: /usr/share/opensearch/logs" in config_text
    assert (config_dir / "jvm.options").exists()
    assert (config_dir / "log4j2.properties").exists()
    assert (config_dir / "opensearch.keystore.tmp").exists()


def test_slurm_common_opensearch_binds_writable_config_dir_for_keystore_updates(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_image = tmp_path / "opensearch.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    slurm_tmpdir = tmp_path / "slurm_tmp"

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
bind_specs=()
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      bind_specs+=("${args[$((index + 1))]}")
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  index=$((index + 1))
fi
post_args="${args[*]:${index}}"
if [[ "${post_args}" == *"/bin/sh -c"* && "${post_args}" == *"[ -x /usr/share/opensearch/bin/opensearch ]"* && "${post_args}" == *"[ -x /usr/share/opensearch/opensearch-docker-entrypoint.sh ]"* && "${post_args}" == *"[ -x /usr/share/opensearch/jdk/bin/java ]"* ]]; then
  exit 0
fi
if [[ "${post_args}" == *"cp -R /usr/share/opensearch/config/. "* ]]; then
  last_index=$((${#args[@]} - 1))
  target="${args[${last_index}]}"
  mkdir -p "${target}"
  exit 0
fi
if [[ "${post_args}" == *"cd /usr/share/opensearch && exec ./opensearch-docker-entrypoint.sh opensearch"* ]]; then
  config_bind=""
  for bind_spec in "${bind_specs[@]}"; do
    if [[ "${bind_spec}" == *":/usr/share/opensearch/config" ]]; then
      config_bind="${bind_spec%%:/usr/share/opensearch/config}"
      break
    fi
  done
  if [[ -z "${config_bind}" || ! -d "${config_bind}" ]]; then
    echo "Likely root cause: java.nio.file.FileSystemException: /usr/share/opensearch/config/opensearch.keystore.tmp: Read-only file system" >&2
    exit 1
  fi
  touch "${config_bind}/opensearch.keystore.tmp"
  sleep 30
  exit 0
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_OPENSEARCH_IMAGE="{fake_image}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="8"; '
                'mb_wait_named_adapter_health_timeout() { sleep 0.1; return 0; }; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_opensearch_service; '
                'mb_stop_engine_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (
        slurm_tmpdir
        / "maxionbench_engine_runtime"
        / "4242_8"
        / "opensearch"
        / "config"
        / "opensearch.keystore.tmp"
    ).exists()


def test_slurm_common_weaviate_startup_accepts_detached_parent_when_health_recovers(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_image = tmp_path / "weaviate.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    slurm_tmpdir = tmp_path / "slurm_tmp"
    health_log = tmp_path / "weaviate_health.txt"
    post_image_log = tmp_path / "weaviate_post_image.log"

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  index=$((index + 1))
fi
printf '%s\\n' "${args[@]:${index}}" > "${MAXIONBENCH_TEST_POST_IMAGE_LOG}"
post_args="${args[*]:${index}}"
all_args="$*"
if [[ "${post_args}" == "weaviate --help" ]]; then
  exit 0
fi
if [[ "${all_args}" == *"--env QUERY_DEFAULTS_LIMIT=20"* && "${all_args}" == *"CLUSTER_GOSSIP_BIND_PORT="* && "${all_args}" == *"CLUSTER_DATA_BIND_PORT="* && "${all_args}" == *"RAFT_BOOTSTRAP_EXPECT=1"* && "${all_args}" == *"RAFT_BOOTSTRAP_TIMEOUT=90"* && "${post_args}" == *"weaviate --scheme http --host 0.0.0.0 --port"* ]]; then
  echo "grpc server listening on detached child"
  exit 0
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_WEAVIATE_IMAGE="{fake_image}"; '
                f'export MAXIONBENCH_TEST_POST_IMAGE_LOG="{post_image_log}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="9"; '
                'mb_wait_named_adapter_health_timeout() { '
                f'  printf "%s|%s|%s\\n" "$1" "$2" "$3" > "{health_log}"; '
                '  return 0; '
                '}; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_weaviate_service'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "started engine service weaviate detached_after_startup=1" in completed.stdout
    health_adapter, health_options_json, health_timeout = health_log.read_text(encoding="utf-8").strip().split("|", maxsplit=2)
    assert health_adapter == "weaviate"
    assert health_timeout == "30"
    health_options = json.loads(health_options_json)
    assert health_options["host"] == "127.0.0.1"
    assert health_options["scheme"] == "http"
    assert isinstance(health_options["port"], int)
    post_image_args = post_image_log.read_text(encoding="utf-8")
    assert post_image_args.startswith("weaviate\n")
    assert "weaviate" in post_image_args
    assert "\n--scheme\nhttp\n" in post_image_args
    assert "\n--host\n0.0.0.0\n" in post_image_args
    assert "8300" not in post_image_args
    assert "8301" not in post_image_args


def test_slurm_common_weaviate_startup_fails_when_parent_exits_before_health(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_image = tmp_path / "weaviate.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    slurm_tmpdir = tmp_path / "slurm_tmp"

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  index=$((index + 1))
fi
post_args="${args[*]:${index}}"
all_args="$*"
if [[ "${post_args}" == "weaviate --help" ]]; then
  exit 0
fi
if [[ "${all_args}" == *"--env QUERY_DEFAULTS_LIMIT=20"* && "${all_args}" == *"CLUSTER_GOSSIP_BIND_PORT="* && "${all_args}" == *"CLUSTER_DATA_BIND_PORT="* && "${post_args}" == *"weaviate --scheme http --host 0.0.0.0 --port"* ]]; then
  echo "weaviate bootstrap failed after parent exit"
  exit 0
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_WEAVIATE_IMAGE="{fake_image}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="10"; '
                'mb_wait_named_adapter_health_timeout() { return 1; }; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_weaviate_service'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert "weaviate bootstrap failed after parent exit" in output
    assert "managed engine service weaviate exited before adapter health became reachable" in output


def test_slurm_common_weaviate_startup_uses_allocated_internal_cluster_ports(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_image = tmp_path / "weaviate.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    slurm_tmpdir = tmp_path / "slurm_tmp"
    post_image_log = tmp_path / "weaviate_collision_post_image.log"

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  index=$((index + 1))
fi
printf '%s\\n' "${args[@]:${index}}" > "${MAXIONBENCH_TEST_POST_IMAGE_LOG}"
post_args="${args[*]:${index}}"
all_args="$*"
if [[ "${post_args}" == "weaviate --help" ]]; then
  exit 0
fi
if [[ "${all_args}" != *"CLUSTER_GOSSIP_BIND_PORT="* || "${all_args}" != *"CLUSTER_DATA_BIND_PORT="* ]]; then
  echo "shared-node port collision on implicit weaviate cluster ports" >&2
  exit 1
fi
if [[ "${all_args}" == *"CLUSTER_GOSSIP_BIND_PORT=8300"* || "${all_args}" == *"CLUSTER_DATA_BIND_PORT=8301"* ]]; then
  echo "shared-node port collision on default weaviate cluster ports" >&2
  exit 1
fi
if [[ "${post_args}" == *"weaviate --scheme http --host 0.0.0.0 --port"* ]]; then
  sleep 30
  exit 0
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_WEAVIATE_IMAGE="{fake_image}"; '
                f'export MAXIONBENCH_TEST_POST_IMAGE_LOG="{post_image_log}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="11"; '
                'mb_wait_named_adapter_health_timeout() { sleep 0.1; return 0; }; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_weaviate_service; '
                'mb_stop_engine_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    post_image_args = post_image_log.read_text(encoding="utf-8")
    assert post_image_args.startswith("weaviate\n")
    assert "\n--scheme\nhttp\n" in post_image_args
    assert "\n--host\n0.0.0.0\n" in post_image_args
    assert "8300" not in post_image_args
    assert "8301" not in post_image_args


def test_slurm_common_milvus_services_use_direct_exec_without_shell(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    apptainer_log = tmp_path / "apptainer_milvus.log"
    http_log = tmp_path / "milvus_startup_http.log"
    slurm_tmpdir = tmp_path / "slurm_tmp"

    etcd_image = tmp_path / "milvus-etcd.sif"
    minio_image = tmp_path / "milvus-minio.sif"
    milvus_image = tmp_path / "milvus.sif"
    for image_path in (etcd_image, minio_image, milvus_image):
        image_path.write_text("image\n", encoding="utf-8")

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MAXIONBENCH_TEST_APPTAINER_LOG}"
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  image="${args[${index}]}"
  index=$((index + 1))
else
  image=""
fi
post_args="${args[*]:${index}}"
all_args="$*"
if [[ "${image##*/}" == "milvus.sif" && "${post_args}" == *"/milvus/configs/."* ]]; then
  target="${args[$(( ${#args[@]} - 1 ))]}"
  mkdir -p "${target}"
  cat > "${target}/milvus.yaml" <<'EOF'
etcd: {}
minio: {}
proxy:
  http: {}
EOF
  exit 0
fi
if [[ "${post_args}" == *"/bin/sh"* ]]; then
  exit 1
fi
case "${image##*/}" in
  milvus-etcd.sif)
    if [[ "${post_args}" == "etcd --version" ]]; then
      exit 0
    fi
    if [[ "${post_args}" == *"etcd -advertise-client-urls=http://127.0.0.1:"* && "${post_args}" == *"--data-dir=/etcd"* ]]; then
      sleep 30
      exit 0
    fi
    ;;
  milvus-minio.sif)
    if [[ "${post_args}" == "minio --version" ]]; then
      exit 0
    fi
    if [[ "${all_args}" == *"--env MINIO_ROOT_USER=minioadmin,MINIO_ROOT_PASSWORD=minioadmin"* && "${post_args}" == *"minio server /minio_data --address :"* && "${post_args}" == *"--console-address :"* ]]; then
      sleep 30
      exit 0
    fi
    ;;
  milvus.sif)
    if [[ "${post_args}" == "milvus --help" ]]; then
      exit 0
    fi
    if [[ "${all_args}" == *"--env ETCD_ENDPOINTS=127.0.0.1:"* && "${all_args}" == *"MINIO_ADDRESS=127.0.0.1:"* && "${post_args}" == *"milvus run standalone"* ]]; then
      sleep 30
      exit 0
    fi
    ;;
esac
exit 1
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_MILVUS_ETCD_IMAGE="{etcd_image}"; '
                f'export MAXIONBENCH_MILVUS_MINIO_IMAGE="{minio_image}"; '
                f'export MAXIONBENCH_MILVUS_IMAGE="{milvus_image}"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{apptainer_log}"; '
                f'export MAXIONBENCH_TEST_HTTP_LOG="{http_log}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_wait_named_adapter_health_timeout() { sleep 0.1; return 0; }; '
                'mb_wait_http_health_timeout() { printf "%s\\n" "$1" >> "${MAXIONBENCH_TEST_HTTP_LOG}"; sleep 0.1; return 0; }; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_milvus_services; '
                'mb_stop_engine_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    apptainer_calls = apptainer_log.read_text(encoding="utf-8")
    runtime_root = slurm_tmpdir / "maxionbench_engine_runtime" / "4242_7" / "milvus"
    assert " milvus-etcd.sif env " not in apptainer_calls
    assert " milvus-minio.sif env " not in apptainer_calls
    assert " milvus.sif env " not in apptainer_calls
    assert "milvus-etcd.sif etcd --version" in apptainer_calls
    assert "milvus-minio.sif minio --version" in apptainer_calls
    assert "milvus.sif milvus --help" in apptainer_calls
    assert "/milvus/configs/." in apptainer_calls
    assert "--env MINIO_ROOT_USER=minioadmin,MINIO_ROOT_PASSWORD=minioadmin" in apptainer_calls
    assert "--env ETCD_ENDPOINTS=127.0.0.1:" in apptainer_calls
    assert "MILVUSCONF=/milvus/configs" in apptainer_calls
    assert "MILVUS_PROXY_PORT=" not in apptainer_calls
    assert "MILVUS_METRICS_PORT=" not in apptainer_calls
    assert f"--bind {runtime_root / 'etcd'}:/etcd" in apptainer_calls
    assert f"--bind {runtime_root / 'minio'}:/minio_data" in apptainer_calls
    assert f"--bind {runtime_root / 'data'}:/var/lib/milvus" in apptainer_calls
    assert f"--bind {runtime_root / 'config'}:/milvus/configs" in apptainer_calls
    milvus_cfg = yaml.safe_load((runtime_root / "config" / "milvus.yaml").read_text(encoding="utf-8"))
    assert milvus_cfg["etcd"]["endpoints"].startswith("127.0.0.1:")
    assert milvus_cfg["minio"]["address"] == "127.0.0.1"
    assert isinstance(milvus_cfg["minio"]["port"], int)
    assert milvus_cfg["proxy"]["ip"] == "127.0.0.1"
    assert isinstance(milvus_cfg["proxy"]["port"], int)
    assert milvus_cfg["proxy"]["http"]["enabled"] is True
    assert isinstance(milvus_cfg["proxy"]["http"]["port"], int)
    assert isinstance(milvus_cfg["rootCoord"]["port"], int)
    assert isinstance(milvus_cfg["dataCoord"]["port"], int)
    assert isinstance(milvus_cfg["queryCoord"]["port"], int)
    assert milvus_cfg["proxy"]["port"] != milvus_cfg["rootCoord"]["port"]
    assert milvus_cfg["proxy"]["port"] != milvus_cfg["dataCoord"]["port"]
    assert milvus_cfg["proxy"]["port"] != milvus_cfg["queryCoord"]["port"]
    health_urls = http_log.read_text(encoding="utf-8").splitlines()
    assert len(health_urls) == 2
    assert health_urls[0].startswith("http://127.0.0.1:")
    assert health_urls[0].endswith("/readyz")
    assert health_urls[1].startswith("http://127.0.0.1:")
    assert health_urls[1].endswith("/minio/health/live")


def test_slurm_common_milvus_services_fail_before_launching_milvus_when_etcd_health_never_recovers(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    apptainer_log = tmp_path / "apptainer_milvus_fail.log"
    slurm_tmpdir = tmp_path / "slurm_tmp"

    etcd_image = tmp_path / "milvus-etcd.sif"
    minio_image = tmp_path / "milvus-minio.sif"
    milvus_image = tmp_path / "milvus.sif"
    for image_path in (etcd_image, minio_image, milvus_image):
        image_path.write_text("image\n", encoding="utf-8")

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MAXIONBENCH_TEST_APPTAINER_LOG}"
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --env)
      index=$((index + 2))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  image="${args[${index}]}"
  index=$((index + 1))
else
  image=""
fi
post_args="${args[*]:${index}}"
if [[ "${image##*/}" == "milvus.sif" && "${post_args}" == *"/milvus/configs/."* ]]; then
  target="${args[$(( ${#args[@]} - 1 ))]}"
  mkdir -p "${target}"
  cat > "${target}/milvus.yaml" <<'EOF'
etcd: {}
minio: {}
proxy:
  http: {}
EOF
  exit 0
fi
if [[ "${post_args}" == *"/bin/sh"* ]]; then
  exit 1
fi
case "${image##*/}" in
  milvus-etcd.sif)
    if [[ "${post_args}" == "etcd --version" ]]; then
      exit 0
    fi
    if [[ "${post_args}" == *"etcd -advertise-client-urls=http://127.0.0.1:"* && "${post_args}" == *"--data-dir=/etcd"* ]]; then
      sleep 30
      exit 0
    fi
    ;;
  milvus-minio.sif)
    if [[ "${post_args}" == "minio --version" ]]; then
      exit 0
    fi
    if [[ "${post_args}" == *"minio server /minio_data --address :"* && "${post_args}" == *"--console-address :"* ]]; then
      sleep 30
      exit 0
    fi
    ;;
  milvus.sif)
    if [[ "${post_args}" == "milvus --help" ]]; then
      exit 0
    fi
    ;;
esac
exit 1
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_MILVUS_ETCD_IMAGE="{etcd_image}"; '
                f'export MAXIONBENCH_MILVUS_MINIO_IMAGE="{minio_image}"; '
                f'export MAXIONBENCH_MILVUS_IMAGE="{milvus_image}"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{apptainer_log}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_wait_named_adapter_health_timeout() { sleep 0.1; return 0; }; '
                'mb_wait_http_health_timeout() { return 1; }; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_milvus_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert "managed engine service milvus-etcd failed startup HTTP health verification within 30s" in output
    apptainer_calls = apptainer_log.read_text(encoding="utf-8")
    assert "milvus-etcd.sif etcd --version" in apptainer_calls
    assert "milvus.sif milvus run standalone" not in apptainer_calls


def test_slurm_common_cleanup_local_runtime_removes_scratch_but_keeps_final_output(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    config_path = tmp_path / "cleanup_config.yaml"
    dataset_dir = tmp_path / "processed_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "base.npy").write_bytes(b"123")
    config_path.write_text(
        yaml.safe_dump(
            {
                "scenario": "cleanup_probe",
                "processed_dataset_path": str(dataset_dir),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    scratch_dir = tmp_path / "scratch"
    output_root = tmp_path / "results"

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                f'export SLURM_TMPDIR="{scratch_dir}"; '
                f'export MAXIONBENCH_OUTPUT_ROOT="{output_root}"; '
                'export MAXIONBENCH_CLEANUP_LOCAL_SCRATCH="1"; '
                'export MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI="${SLURM_TMPDIR}/lancedb/service"; '
                'mb_require_tmpdir; '
                'mb_prepare_output_paths "cleanup_probe"; '
                f'STAGED_CONFIG="$(mb_stage_config_to_tmp "{config_path}")"; '
                'export MB_STAGE_ROOT="$(dirname "${STAGED_CONFIG}")"; '
                'mkdir -p "$(mb_engine_runtime_root)/logs" "${MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI}" "${MB_OUTPUT_TMP}"; '
                'printf "service\\n" > "$(mb_engine_runtime_root)/logs/service.log"; '
                'printf "result\\n" > "${MB_OUTPUT_TMP}/results.parquet"; '
                'mb_copy_back_output; '
                'mb_cleanup_local_runtime; '
                'printf "FINAL=%s\\n" "${MB_OUTPUT_FINAL}"; '
                'printf "TMP_EXISTS=%s\\n" "$(test -e "${MB_OUTPUT_TMP}" && echo 1 || echo 0)"; '
                'printf "STAGE_EXISTS=%s\\n" "$(test -e "${MB_STAGE_ROOT}" && echo 1 || echo 0)"; '
                'printf "RUNTIME_EXISTS=%s\\n" "$(test -e "$(mb_engine_runtime_root)" && echo 1 || echo 0)"; '
                'printf "LANCEDB_EXISTS=%s\\n" "$(test -e "${MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI}" && echo 1 || echo 0)"'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    stdout_lines = dict(
        line.split("=", maxsplit=1)
        for line in completed.stdout.splitlines()
        if "=" in line
    )
    final_output = Path(stdout_lines["FINAL"])
    assert final_output.exists()
    assert (final_output / "results.parquet").read_text(encoding="utf-8") == "result\n"
    assert stdout_lines["TMP_EXISTS"] == "0"
    assert stdout_lines["STAGE_EXISTS"] == "0"
    assert stdout_lines["RUNTIME_EXISTS"] == "0"
    assert stdout_lines["LANCEDB_EXISTS"] == "0"


def test_slurm_common_finalize_job_captures_runtime_logs_before_cleanup(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    scratch_dir = tmp_path / "scratch"
    output_root = tmp_path / "results"

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                f'export SLURM_TMPDIR="{scratch_dir}"; '
                f'export MAXIONBENCH_OUTPUT_ROOT="{output_root}"; '
                'export MAXIONBENCH_CLEANUP_LOCAL_SCRATCH="1"; '
                'mb_require_tmpdir; '
                'mb_prepare_output_paths "finalize_probe"; '
                'mkdir -p "$(mb_engine_runtime_root)/logs" "${MB_OUTPUT_TMP}"; '
                'printf "service\\n" > "$(mb_engine_runtime_root)/logs/service.log"; '
                'printf "result\\n" > "${MB_OUTPUT_TMP}/results.parquet"; '
                'mb_finalize_job 9 0; '
                'printf "FINAL=%s\\n" "${MB_OUTPUT_FINAL}"; '
                'printf "RUNTIME_EXISTS=%s\\n" "$(test -e "$(mb_engine_runtime_root)" && echo 1 || echo 0)"'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    stdout_lines = dict(
        line.split("=", maxsplit=1)
        for line in completed.stdout.splitlines()
        if "=" in line
    )
    final_output = Path(stdout_lines["FINAL"])
    assert final_output.exists()
    assert (final_output / "results.parquet").read_text(encoding="utf-8") == "result\n"
    run_status = json.loads((final_output / "run_status.json").read_text(encoding="utf-8"))
    assert run_status["status"] == "failed"
    assert int(run_status["exit_code"]) == 9
    service_logs = list(final_output.glob("logs/local_runtime/engine_runtime/**/service.log"))
    assert service_logs, list(final_output.rglob("*"))
    assert service_logs[0].read_text(encoding="utf-8") == "service\n"
    assert stdout_lines["RUNTIME_EXISTS"] == "0"


def test_slurm_wrapper_scripts_source_common_from_exported_slurm_dir() -> None:
    for rel_path in (
        "maxionbench/orchestration/slurm/download_datasets.sh",
        "maxionbench/orchestration/slurm/preprocess_datasets.sh",
        "maxionbench/orchestration/slurm/conformance_matrix.sh",
        "maxionbench/orchestration/slurm/postprocess.sh",
        "maxionbench/orchestration/slurm/calibrate_d3.sh",
        "maxionbench/orchestration/slurm/prefetch_datasets.sh",
        "maxionbench/orchestration/slurm/cpu_array.sh",
        "maxionbench/orchestration/slurm/gpu_array.sh",
    ):
        text = Path(rel_path).read_text(encoding="utf-8")
        assert 'SLURM_DIR="${MAXIONBENCH_SLURM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"' in text
        assert 'source "${SLURM_DIR}/common.sh"' in text


def test_cpu_array_includes_d3_matched_s1_baseline_config() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert "configs/scenarios/s1_ann_frontier_d3.yaml" in text


def test_cpu_array_supports_partial_scenario_dir_override_fallback() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SCENARIO_CONFIG_DIR" in text
    assert "MAXIONBENCH_SLURM_RUN_MANIFEST" in text
    assert "run_manifest resolve" in text
    assert 'CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/$(basename "${DEFAULT_CONFIG_PATH}")"' in text
    assert 'if [[ -f "$(mb_resolve_config "${CANDIDATE_CONFIG_PATH}")" ]]; then' in text
    assert 'CONFIG_PATH="${DEFAULT_CONFIG_PATH}"' in text
    assert 'SCENARIO_KEY="$(mb_read_config_field "${CONFIG_PATH}" "scenario")"' in text


def test_cpu_array_supports_skip_s6_env_flag() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SKIP_S6" in text
    assert "s6_fusion.yaml" in text
    assert "skipping S6 task index" in text


def test_cpu_array_starts_and_stops_managed_engine_services() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert 'if mb_engine_requires_service "${STAGED_CONFIG}"; then' in text
    assert "mb_start_engine_services" in text
    assert "mb_wait_engine_health" in text
    assert 'trap \'status=$?; trap - EXIT; mb_finalize_job "${status}" "${SERVICE_STARTED:-0}"; exit "${status}"\' EXIT' in text
    assert "SERVICE_STARTED=1" in text
    assert 'export MB_STAGE_ROOT="$(dirname "${STAGED_CONFIG}")"' in text
    assert "mb_finalize_job" in text


def test_gpu_array_supports_partial_scenario_dir_override_fallback() -> None:
    text = Path("maxionbench/orchestration/slurm/gpu_array.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SCENARIO_CONFIG_DIR" in text
    assert "MAXIONBENCH_SLURM_RUN_MANIFEST" in text
    assert "run_manifest resolve" in text
    assert 'CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/$(basename "${DEFAULT_CONFIG_PATH}")"' in text
    assert 'if [[ -f "$(mb_resolve_config "${CANDIDATE_CONFIG_PATH}")" ]]; then' in text
    assert 'CONFIG_PATH="${DEFAULT_CONFIG_PATH}"' in text


def test_gpu_array_starts_and_stops_managed_engine_services() -> None:
    text = Path("maxionbench/orchestration/slurm/gpu_array.sh").read_text(encoding="utf-8")
    assert 'if mb_engine_requires_service "${STAGED_CONFIG}"; then' in text
    assert "mb_start_engine_services" in text
    assert "mb_wait_engine_health" in text
    assert 'trap \'status=$?; trap - EXIT; mb_finalize_job "${status}" "${SERVICE_STARTED:-0}"; exit "${status}"\' EXIT' in text
    assert "SERVICE_STARTED=1" in text
    assert 'export MB_STAGE_ROOT="$(dirname "${STAGED_CONFIG}")"' in text
    assert "mb_finalize_job" in text


def test_new_slurm_pipeline_scripts_exist() -> None:
    for path in (
        Path("maxionbench/orchestration/slurm/download_datasets.sh"),
        Path("maxionbench/orchestration/slurm/preprocess_datasets.sh"),
        Path("maxionbench/orchestration/slurm/conformance_matrix.sh"),
        Path("maxionbench/orchestration/slurm/postprocess.sh"),
        Path("run_slurm_pipeline.sh"),
        Path("maxionbench/orchestration/slurm/profiles_clusters.example.yaml"),
        Path(".env.slurm.example"),
    ):
        assert path.exists(), path


def test_gpu_array_explicitly_lists_track_b_and_track_c_entries() -> None:
    text = Path("maxionbench/orchestration/slurm/gpu_array.sh").read_text(encoding="utf-8")
    assert "s1_ann_frontier_track_b_gpu.yaml" in text
    assert "s5_rerank_track_c_gpu.yaml" in text


def test_calibrate_d3_supports_scenario_dir_override_with_explicit_override_precedence() -> None:
    text = Path("maxionbench/orchestration/slurm/calibrate_d3.sh").read_text(encoding="utf-8")
    assert 'CONFIG_PATH="${MAXIONBENCH_CALIBRATE_CONFIG:-configs/scenarios/calibrate_d3.yaml}"' in text
    assert 'if [[ -z "${MAXIONBENCH_CALIBRATE_CONFIG:-}" ]]; then' in text
    assert 'SCENARIO_CONFIG_DIR="${MAXIONBENCH_SCENARIO_CONFIG_DIR:-}"' in text
    assert 'CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml"' in text
    assert 'if [[ ! -f "$(mb_resolve_config "${CONFIG_PATH}")" ]]; then' in text
    assert "mb_source_dataset_env" in text
    assert 'export MB_STAGE_ROOT="$(dirname "${STAGED_CONFIG}")"' in text
    assert "mb_finalize_job" in text


def test_prefetch_datasets_script_exists_and_uses_prefetch_helper() -> None:
    text = Path("maxionbench/orchestration/slurm/prefetch_datasets.sh").read_text(encoding="utf-8")
    assert "dataset_prefetch" in text
    assert "MAXIONBENCH_DATASET_ENV_SH" in text
    assert "mb_source_dataset_env" in text
