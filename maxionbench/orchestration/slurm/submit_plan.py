"""Submit Slurm jobs with enforced MaxionBench dependency ordering."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml

from maxionbench.orchestration.config_schema import expand_env_placeholders
from maxionbench.orchestration.slurm.run_manifest import RunManifest, build_run_manifest


@dataclass(frozen=True)
class SubmitStep:
    key: str
    script_name: str
    array: str | None = None
    depends_on: tuple[str, ...] = ()


SLURM_PROFILE_OVERRIDES_ENV = "MAXIONBENCH_SLURM_PROFILE_OVERRIDES"
DEFAULT_LOCAL_PROFILE_OVERRIDES_PATH = Path(__file__).resolve().with_name("profiles_local.yaml")
DEFAULT_TRACKED_PROFILES_PATH = Path(__file__).resolve().with_name("profiles_clusters.example.yaml")
_LOG = logging.getLogger(__name__)
_D3_WORKLOAD_SCENARIOS = {"s2_filtered_ann", "s3_churn_smooth", "s3b_churn_bursty"}
_GPU_SPLIT_STEP_KEYS = {"gpu_d3_baseline", "gpu_d3_workloads", "gpu_non_d3"}
_EXPECTED_FULL_MATRIX_GPU_ROWS = 19


def build_submit_steps(
    *,
    include_gpu: bool = True,
    skip_s6: bool = False,
    prepare_containers: bool = False,
    prefetch_datasets: bool = False,
    download_datasets: bool = False,
    preprocess_datasets: bool = False,
    include_postprocess: bool = False,
    manifest: RunManifest | None = None,
) -> list[SubmitStep]:
    steps: list[SubmitStep] = []
    container_stage_key: str | None = None
    if prepare_containers:
        steps.append(
            SubmitStep(
                key="prepare_containers",
                script_name="prepare_containers.sh",
            )
        )
        container_stage_key = "prepare_containers"
    last_dataset_stage_key: str | None = None
    if download_datasets:
        steps.append(
            SubmitStep(
                key="download_datasets",
                script_name="download_datasets.sh",
            )
        )
        last_dataset_stage_key = "download_datasets"
    if preprocess_datasets:
        steps.append(
            SubmitStep(
                key="preprocess_datasets",
                script_name="preprocess_datasets.sh",
                depends_on=((last_dataset_stage_key,) if last_dataset_stage_key else ()),
            )
        )
        last_dataset_stage_key = "preprocess_datasets"
    if prefetch_datasets:
        steps.append(
            SubmitStep(
                key="prefetch_datasets",
                script_name="prefetch_datasets.sh",
                depends_on=((last_dataset_stage_key,) if last_dataset_stage_key else ()),
            )
        )
        last_dataset_stage_key = "prefetch_datasets"
    conformance_depends_list: list[str] = []
    if last_dataset_stage_key:
        conformance_depends_list.append(last_dataset_stage_key)
    if container_stage_key:
        conformance_depends_list.append(container_stage_key)
    conformance_depends_on = tuple(conformance_depends_list)
    steps.append(
        SubmitStep(
            key="conformance",
            script_name="conformance_matrix.sh",
            depends_on=conformance_depends_on,
        )
    )
    calibrate_depends_on: tuple[str, ...] = ("conformance",)

    if manifest is None:
        cpu_non_d3_array = "0,5" if skip_s6 else "0,5-6"
        steps.extend(
            [
                SubmitStep(
                    key="calibrate",
                    script_name="calibrate_d3.sh",
                    depends_on=calibrate_depends_on,
                ),
                SubmitStep(
                    key="cpu_d3_baseline",
                    script_name="cpu_array.sh",
                    array="1",
                    depends_on=("calibrate",),
                ),
                SubmitStep(
                    key="cpu_d3_workloads",
                    script_name="cpu_array.sh",
                    array="2-4",
                    depends_on=("calibrate", "cpu_d3_baseline"),
                ),
                SubmitStep(
                    key="cpu_non_d3",
                    script_name="cpu_array.sh",
                    array=cpu_non_d3_array,
                    depends_on=("calibrate",),
                ),
            ]
        )
        if include_gpu:
            steps.append(
                SubmitStep(
                    key="gpu_all",
                    script_name="gpu_array.sh",
                    array="0-2",
                    depends_on=("calibrate",),
                )
            )
    else:
        steps.append(
            SubmitStep(
                key="calibrate",
                script_name="calibrate_d3.sh",
                depends_on=calibrate_depends_on,
            )
        )
        cpu_baseline, cpu_d3_workloads, cpu_non_d3 = _partition_manifest_indices(manifest.cpu_rows)
        if cpu_baseline:
            steps.append(
                SubmitStep(
                    key="cpu_d3_baseline",
                    script_name="cpu_array.sh",
                    array=_indices_to_array_spec(cpu_baseline),
                    depends_on=("calibrate",),
                )
            )
        if cpu_d3_workloads:
            workload_depends = ("calibrate", "cpu_d3_baseline") if cpu_baseline else ("calibrate",)
            steps.append(
                SubmitStep(
                    key="cpu_d3_workloads",
                    script_name="cpu_array.sh",
                    array=_indices_to_array_spec(cpu_d3_workloads),
                    depends_on=workload_depends,
                )
            )
        if cpu_non_d3:
            steps.append(
                SubmitStep(
                    key="cpu_non_d3",
                    script_name="cpu_array.sh",
                    array=_indices_to_array_spec(cpu_non_d3),
                    depends_on=("calibrate",),
                )
            )
        if include_gpu and manifest.gpu_rows:
            gpu_baseline, gpu_d3_workloads, gpu_non_d3 = _partition_manifest_indices(manifest.gpu_rows)
            if gpu_baseline:
                steps.append(
                    SubmitStep(
                        key="gpu_d3_baseline",
                        script_name="gpu_array.sh",
                        array=_indices_to_array_spec(gpu_baseline),
                        depends_on=("calibrate",),
                    )
                )
            if gpu_d3_workloads:
                workload_depends = ("calibrate", "gpu_d3_baseline") if gpu_baseline else ("calibrate",)
                steps.append(
                    SubmitStep(
                        key="gpu_d3_workloads",
                        script_name="gpu_array.sh",
                        array=_indices_to_array_spec(gpu_d3_workloads),
                        depends_on=workload_depends,
                    )
                )
            if gpu_non_d3:
                steps.append(
                    SubmitStep(
                        key="gpu_non_d3",
                        script_name="gpu_array.sh",
                        array=_indices_to_array_spec(gpu_non_d3),
                        depends_on=("calibrate",),
                    )
                )
    if include_postprocess:
        benchmark_depends = tuple(
            step.key
            for step in steps
            if step.key
            in {
                "cpu_d3_baseline",
                "cpu_d3_workloads",
                "cpu_non_d3",
                "gpu_all",
                "gpu_d3_baseline",
                "gpu_d3_workloads",
                "gpu_non_d3",
            }
        )
        steps.append(
            SubmitStep(
                key="postprocess",
                script_name="postprocess.sh",
                depends_on=benchmark_depends,
            )
        )
    return steps


def submit_steps(
    *,
    slurm_dir: Path,
    steps: list[SubmitStep],
    seed: int | None = None,
    scenario_config_dir: str | None = None,
    output_root: str | None = None,
    slurm_profile: str | None = None,
    container_runtime: str | None = None,
    container_image: str | None = None,
    container_bind: list[str] | None = None,
    hf_cache_dir: str | None = None,
    conformance_matrix_path: str | None = None,
    prefetch_datasets: bool = False,
    download_datasets: bool = False,
    preprocess_datasets: bool = False,
    include_postprocess: bool = False,
    run_manifest: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    normalized_container_runtime = str(container_runtime).strip().lower() if container_runtime else None
    normalized_container_image = str(container_image).strip() if container_image else None
    normalized_hf_cache_dir = str(hf_cache_dir).strip() if hf_cache_dir else None
    normalized_conformance_matrix_path = str(conformance_matrix_path).strip() if conformance_matrix_path else None
    normalized_run_manifest = str(run_manifest).strip() if run_manifest else None
    normalized_container_bind = [str(item).strip() for item in (container_bind or []) if str(item).strip()]
    if normalized_container_runtime and normalized_container_runtime != "apptainer":
        raise ValueError(f"unsupported container runtime: {normalized_container_runtime}")
    if normalized_container_runtime and not normalized_container_image:
        raise ValueError("container_image is required when container_runtime is set")
    if normalized_container_image and not normalized_container_runtime:
        raise ValueError("container_runtime is required when container_image is set")

    resolved_dir = slurm_dir.resolve()
    normalized_slurm_profile = str(slurm_profile).strip().lower() if slurm_profile else None
    if normalized_slurm_profile:
        _warn_unknown_step_override_keys(normalized_slurm_profile, valid_step_keys={step.key for step in steps})
    job_ids: dict[str, str] = {}
    submitted: list[dict[str, Any]] = []
    for step in steps:
        script_path = (resolved_dir / step.script_name).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"missing slurm script: {script_path}")
        cmd = _build_sbatch_command(
            step_key=step.key,
            script_path=script_path,
            slurm_dir=resolved_dir,
            array=step.array,
            seed=seed,
            scenario_config_dir=scenario_config_dir,
            output_root=output_root,
            slurm_profile=slurm_profile,
            container_runtime=normalized_container_runtime,
            container_image=normalized_container_image,
            container_bind=normalized_container_bind,
            hf_cache_dir=normalized_hf_cache_dir,
            conformance_matrix_path=normalized_conformance_matrix_path,
            run_manifest=normalized_run_manifest,
            dependencies=tuple(job_ids[key] for key in step.depends_on),
        )
        if dry_run:
            job_id = f"<{step.key.upper()}_JOB_ID>"
        else:
            job_id = _run_sbatch(cmd)
        job_ids[step.key] = job_id
        submitted.append(
            {
                "key": step.key,
                "script": str(script_path),
                "array": step.array,
                "depends_on": list(step.depends_on),
                "dependencies_resolved": [job_ids[key] for key in step.depends_on],
                "command": cmd,
                "job_id": job_id,
            }
        )
    return {
        "slurm_dir": str(resolved_dir),
        "dry_run": dry_run,
        "seed": seed,
        "scenario_config_dir": scenario_config_dir,
        "output_root": output_root,
        "slurm_profile": slurm_profile,
        "container_runtime": normalized_container_runtime,
        "container_image": normalized_container_image,
        "container_bind": normalized_container_bind,
        "hf_cache_dir": normalized_hf_cache_dir,
        "conformance_matrix_path": normalized_conformance_matrix_path,
        "prefetch_datasets": bool(prefetch_datasets),
        "download_datasets": bool(download_datasets),
        "preprocess_datasets": bool(preprocess_datasets),
        "include_postprocess": bool(include_postprocess),
        "run_manifest": normalized_run_manifest,
        "steps": submitted,
        "job_ids": job_ids,
    }


def _build_sbatch_command(
    *,
    step_key: str,
    script_path: Path,
    slurm_dir: Path,
    array: str | None,
    seed: int | None,
    scenario_config_dir: str | None,
    output_root: str | None,
    slurm_profile: str | None,
    container_runtime: str | None,
    container_image: str | None,
    container_bind: list[str],
    hf_cache_dir: str | None,
    conformance_matrix_path: str | None,
    run_manifest: str | None,
    dependencies: tuple[str, ...],
) -> list[str]:
    cmd = ["sbatch", "--parsable"]
    if slurm_profile:
        cmd.extend(_profile_sbatch_args(slurm_profile, step_key=step_key))
    if dependencies:
        cmd.extend(["--dependency", f"afterok:{':'.join(dependencies)}"])
    if array:
        cmd.extend(["--array", array])
    export_vars: list[str] = ["ALL"]
    export_vars.append(f"MAXIONBENCH_SLURM_DIR={str(slurm_dir)}")
    if seed is not None:
        export_vars.append(f"MAXIONBENCH_SEED={int(seed)}")
    if scenario_config_dir:
        export_vars.append(f"MAXIONBENCH_SCENARIO_CONFIG_DIR={scenario_config_dir}")
    if output_root:
        export_vars.append(f"MAXIONBENCH_OUTPUT_ROOT={output_root}")
    if container_runtime:
        export_vars.append(f"MAXIONBENCH_CONTAINER_RUNTIME={container_runtime}")
    if container_image:
        export_vars.append(f"MAXIONBENCH_CONTAINER_IMAGE={container_image}")
    if container_bind:
        export_vars.append(f"MAXIONBENCH_CONTAINER_BIND={'|'.join(container_bind)}")
    if hf_cache_dir:
        export_vars.append(f"MAXIONBENCH_HF_CACHE_DIR={hf_cache_dir}")
    if conformance_matrix_path:
        export_vars.append(f"MAXIONBENCH_CONFORMANCE_MATRIX={conformance_matrix_path}")
    if run_manifest:
        export_vars.append(f"MAXIONBENCH_SLURM_RUN_MANIFEST={run_manifest}")
    if len(export_vars) > 1:
        cmd.extend(["--export", ",".join(export_vars)])
    cmd.append(str(script_path))
    return cmd


def _run_sbatch(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr or stdout or f"sbatch exited with code {proc.returncode}"
        raise RuntimeError(f"failed to submit {' '.join(cmd)}: {detail}")
    raw = proc.stdout.strip()
    if not raw:
        raise RuntimeError(f"sbatch returned empty job id for command: {' '.join(cmd)}")
    # Slurm --parsable may return "<jobid>" or "<jobid>;<cluster>".
    return raw.split(";", maxsplit=1)[0]


def _profile_sbatch_args(slurm_profile: str, *, step_key: str) -> list[str]:
    profiles = _load_local_profile_overrides()
    if slurm_profile not in profiles:
        raise ValueError(
            "unknown slurm profile: "
            f"{slurm_profile}. Define it in {DEFAULT_LOCAL_PROFILE_OVERRIDES_PATH}, "
            f"{DEFAULT_TRACKED_PROFILES_PATH}, or via {SLURM_PROFILE_OVERRIDES_ENV}"
        )
    local_profile = profiles[slurm_profile]
    base_specs = _normalize_flag_specs(local_profile.get("base"), label=f"{slurm_profile}.base")
    overrides = _normalize_step_override_specs(local_profile.get("step_overrides"), step_key=step_key, profile=slurm_profile)
    merged_specs = _merge_flag_specs(base_specs, overrides)
    args: list[str] = []
    for flag, value in merged_specs:
        args.extend([flag, value])
    return args


def _load_local_profile_overrides() -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for candidate in _profile_override_candidates():
        if not candidate.exists():
            continue
        payload = _load_profile_file(candidate)
        normalized.update(payload)
    return normalized


def _profile_override_candidates() -> list[Path]:
    raw_path = os.environ.get(SLURM_PROFILE_OVERRIDES_ENV)
    candidates = [DEFAULT_TRACKED_PROFILES_PATH]
    if raw_path:
        candidates.append(Path(raw_path).expanduser())
    else:
        candidates.append(DEFAULT_LOCAL_PROFILE_OVERRIDES_PATH)
    return candidates


def _load_profile_file(candidate: Path) -> dict[str, dict[str, Any]]:
    with candidate.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Slurm profile overrides must be a YAML mapping: {candidate}")
    normalized: dict[str, dict[str, Any]] = {}
    for profile_name, profile_payload in payload.items():
        profile_key = str(profile_name).strip().lower()
        if not profile_key:
            raise ValueError(f"Local Slurm profile name must be non-empty: {profile_name!r}")
        if profile_payload is None:
            normalized[profile_key] = {}
            continue
        if not isinstance(profile_payload, dict):
            raise ValueError(f"Local Slurm profile override for {profile_key!r} must be a mapping")
        normalized[profile_key] = dict(profile_payload)
    return normalized


def _manifest_indices(*, rows: list[Any], predicate) -> list[int]:  # type: ignore[no-untyped-def]
    return [idx for idx, row in enumerate(rows) if predicate(row)]


def _partition_manifest_indices(rows: list[Any]) -> tuple[list[int], list[int], list[int]]:
    baseline = _manifest_indices(
        rows=rows,
        predicate=lambda row: row.scenario == "s1_ann_frontier" and row.dataset_bundle.upper() == "D3",
    )
    workloads = _manifest_indices(
        rows=rows,
        predicate=lambda row: row.scenario in _D3_WORKLOAD_SCENARIOS,
    )
    baseline_set = set(baseline)
    workload_set = set(workloads)
    non_d3 = [
        idx
        for idx in range(len(rows))
        if idx not in baseline_set and idx not in workload_set
    ]
    return baseline, workloads, non_d3


def _indices_to_array_spec(indices: list[int]) -> str:
    if not indices:
        raise ValueError("array spec requires at least one index")
    ordered = sorted(set(int(item) for item in indices))
    ranges: list[str] = []
    start = ordered[0]
    prev = ordered[0]
    for item in ordered[1:]:
        if item == prev + 1:
            prev = item
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = item
        prev = item
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _normalize_flag_specs(raw_specs: Any, *, label: str) -> tuple[tuple[str, str], ...]:
    if raw_specs in (None, ""):
        return ()
    if not isinstance(raw_specs, list):
        raise ValueError(f"{label} must be a list of [flag, value] pairs")
    normalized: list[tuple[str, str]] = []
    for idx, item in enumerate(raw_specs):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{label}[{idx}] must be a [flag, value] pair")
        flag = str(item[0]).strip()
        expanded_value = expand_env_placeholders(item[1])
        value = "" if expanded_value is None else str(expanded_value).strip()
        if not flag.startswith("--"):
            raise ValueError(f"{label}[{idx}] flag must start with '--'")
        if not value:
            raise ValueError(f"{label}[{idx}] value must be non-empty")
        normalized.append((flag, value))
    return tuple(normalized)


def _normalize_step_override_specs(raw_payload: Any, *, step_key: str, profile: str) -> tuple[tuple[str, str], ...]:
    if raw_payload in (None, ""):
        return ()
    if not isinstance(raw_payload, dict):
        raise ValueError(f"{profile}.step_overrides must be a mapping of step_key -> [flag, value] pairs")
    for candidate_key in _step_override_lookup_keys(step_key):
        if candidate_key not in raw_payload:
            continue
        return _normalize_flag_specs(
            raw_payload.get(candidate_key),
            label=f"{profile}.step_overrides.{candidate_key}",
        )
    return ()


def _merge_flag_specs(
    base_specs: tuple[tuple[str, str], ...],
    override_specs: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    merged: dict[str, str] = {}
    for flag, value in (*base_specs, *override_specs):
        merged[flag] = value
    return tuple((flag, value) for flag, value in merged.items())


def _warn_unknown_step_override_keys(profile: str, *, valid_step_keys: set[str]) -> None:
    profiles = _load_local_profile_overrides()
    profile_payload = profiles.get(profile)
    if profile_payload is None:
        return
    raw_payload = profile_payload.get("step_overrides")
    if raw_payload in (None, "") or not isinstance(raw_payload, dict):
        return
    unknown = sorted(
        key
        for key in (str(item).strip() for item in raw_payload.keys())
        if key and key not in _accepted_step_override_keys(valid_step_keys)
    )
    if not unknown:
        return
    _LOG.warning(
        "Ignoring unknown %s.step_overrides keys: %s. Known step keys: %s",
        profile,
        ", ".join(unknown),
        ", ".join(sorted(valid_step_keys)),
    )


def _step_override_lookup_keys(step_key: str) -> tuple[str, ...]:
    normalized_key = str(step_key).strip()
    if normalized_key in _GPU_SPLIT_STEP_KEYS:
        return (normalized_key, "gpu_all")
    return (normalized_key,)


def _accepted_step_override_keys(valid_step_keys: set[str]) -> set[str]:
    accepted = set(valid_step_keys)
    if valid_step_keys & _GPU_SPLIT_STEP_KEYS:
        accepted.add("gpu_all")
    return accepted


def _conformance_matrix_path_for_run_manifest_dir(run_manifest_dir: str) -> str:
    return str((Path(str(run_manifest_dir)).expanduser().resolve() / "conformance" / "conformance_matrix.csv").resolve())


def _env_flag_true(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def validate_full_matrix_contract(
    *,
    manifest: RunManifest,
    skip_gpu: bool,
    prefetch_datasets: bool,
    allow_gpu_unavailable_env: str | None,
    allow_reduced_matrix: bool = False,
) -> None:
    if skip_gpu:
        raise ValueError("--full-matrix reruns must include all GPU jobs; remove --skip-gpu")
    if _env_flag_true(allow_gpu_unavailable_env):
        raise ValueError("MAXIONBENCH_ALLOW_GPU_UNAVAILABLE=1 is not allowed for full-matrix GPU reruns")
    if not prefetch_datasets:
        raise ValueError("--full-matrix reruns require --prefetch-datasets")
    if allow_reduced_matrix:
        if not manifest.gpu_rows:
            raise ValueError("reduced full-matrix smoke runs must still include at least one GPU row")
        return
    if len(manifest.gpu_rows) != _EXPECTED_FULL_MATRIX_GPU_ROWS:
        raise ValueError(
            f"full-matrix manifest must include all {_EXPECTED_FULL_MATRIX_GPU_ROWS} GPU rows; found {len(manifest.gpu_rows)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Submit MaxionBench Slurm plan with enforced dependencies")
    parser.add_argument(
        "--slurm-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing calibrate_d3.sh/cpu_array.sh/gpu_array.sh",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--slurm-profile",
        default=None,
        help=(
            "Optional local Slurm profile key defined in "
            "maxionbench/orchestration/slurm/profiles_local.yaml or "
            "MAXIONBENCH_SLURM_PROFILE_OVERRIDES"
        ),
    )
    parser.add_argument(
        "--scenario-config-dir",
        default=None,
        help=(
            "Optional scenario config directory override exported to Slurm jobs as "
            "MAXIONBENCH_SCENARIO_CONFIG_DIR"
        ),
    )
    parser.add_argument(
        "--engine-config-dir",
        default="configs/engines",
        help="Engine config catalog used when --full-matrix builds an engine x scenario manifest.",
    )
    parser.add_argument(
        "--run-manifest-dir",
        default="artifacts/slurm_manifests/latest",
        help="Directory used for generated Slurm run manifests and resolved config files in --full-matrix mode.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional output root exported to Slurm jobs as MAXIONBENCH_OUTPUT_ROOT. "
            "Jobs copy final run artifacts under <output-root>/<run_id>."
        ),
    )
    parser.add_argument(
        "--container-runtime",
        default=None,
        choices=["apptainer"],
        help="Optional container runtime for Slurm jobs. Host execution remains the default when omitted.",
    )
    parser.add_argument(
        "--container-image",
        default=None,
        help="Container image path exported to Slurm jobs (for example /shared/containers/maxionbench.sif).",
    )
    parser.add_argument(
        "--container-bind",
        action="append",
        dest="container_bind",
        default=None,
        help=(
            "Repeatable bind spec exported to Slurm jobs for container execution. "
            "Use host[:container[:opts]] and repeat the flag for multiple bind roots."
        ),
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help=(
            "Optional host HF cache directory to bind into the container and export as "
            "HF_HOME/TRANSFORMERS_CACHE/HUGGINGFACE_HUB_CACHE."
        ),
    )
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--prepare-containers",
        action="store_true",
        help="Submit a shared container-build job before the first containerized Slurm step.",
    )
    parser.add_argument(
        "--prefetch-datasets",
        action="store_true",
        help="Submit a dataset prefetch/cache job before calibration and benchmark arrays.",
    )
    parser.add_argument(
        "--download-datasets",
        action="store_true",
        help="Submit a shared-storage dataset download job before preprocessing/calibration.",
    )
    parser.add_argument(
        "--preprocess-datasets",
        action="store_true",
        help="Submit a shared-storage preprocessing job before calibration/benchmark arrays.",
    )
    parser.add_argument(
        "--include-postprocess",
        action="store_true",
        help="Submit a postprocess/report-generation job after all benchmark arrays succeed.",
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help=(
            "Build an engine x scenario manifest from --engine-config-dir and --scenario-config-dir, "
            "then drive cpu_array/gpu_array from that manifest."
        ),
    )
    parser.add_argument(
        "--allow-reduced-matrix",
        action="store_true",
        help="Allow a reduced smoke/debug matrix while still requiring the GPU lane and dataset prefetch.",
    )
    parser.add_argument(
        "--skip-s6",
        action="store_true",
        help="Defer S6 by removing index 6 from cpu_non_d3 array submissions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    manifest: RunManifest | None = None
    manifest_path: str | None = None
    conformance_matrix_path = _conformance_matrix_path_for_run_manifest_dir(str(args.run_manifest_dir))
    if args.full_matrix:
        if not args.scenario_config_dir:
            raise ValueError("--full-matrix requires --scenario-config-dir")
        manifest = build_run_manifest(
            repo_root=Path.cwd(),
            scenario_config_dir=Path(str(args.scenario_config_dir)),
            engine_config_dir=Path(str(args.engine_config_dir)),
            out_dir=Path(str(args.run_manifest_dir)),
            include_gpu=not bool(args.skip_gpu),
            skip_s6=bool(args.skip_s6),
        )
        validate_full_matrix_contract(
            manifest=manifest,
            skip_gpu=bool(args.skip_gpu),
            prefetch_datasets=bool(args.prefetch_datasets),
            allow_gpu_unavailable_env=os.environ.get("MAXIONBENCH_ALLOW_GPU_UNAVAILABLE"),
            allow_reduced_matrix=bool(args.allow_reduced_matrix),
        )
        manifest_path = str((Path(args.run_manifest_dir).expanduser().resolve() / "run_manifest.json").resolve())

    steps = build_submit_steps(
        include_gpu=not bool(args.skip_gpu),
        skip_s6=bool(args.skip_s6),
        prepare_containers=bool(args.prepare_containers),
        prefetch_datasets=bool(args.prefetch_datasets),
        download_datasets=bool(args.download_datasets),
        preprocess_datasets=bool(args.preprocess_datasets),
        include_postprocess=bool(args.include_postprocess),
        manifest=manifest,
    )
    summary = submit_steps(
        slurm_dir=Path(args.slurm_dir),
        steps=steps,
        seed=int(args.seed),
        scenario_config_dir=str(args.scenario_config_dir) if args.scenario_config_dir else None,
        output_root=str(args.output_root) if args.output_root else None,
        slurm_profile=str(args.slurm_profile) if args.slurm_profile else None,
        container_runtime=str(args.container_runtime) if args.container_runtime else None,
        container_image=str(args.container_image) if args.container_image else None,
        container_bind=[str(item) for item in (args.container_bind or [])],
        hf_cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
        conformance_matrix_path=conformance_matrix_path,
        prefetch_datasets=bool(args.prefetch_datasets),
        download_datasets=bool(args.download_datasets),
        preprocess_datasets=bool(args.preprocess_datasets),
        include_postprocess=bool(args.include_postprocess),
        run_manifest=manifest_path,
        dry_run=bool(args.dry_run),
    )
    summary["prepare_containers"] = bool(args.prepare_containers)
    if manifest_path is not None:
        summary["full_matrix"] = True
        summary["allow_reduced_matrix"] = bool(args.allow_reduced_matrix)
        summary["run_manifest"] = manifest_path
        summary["run_manifest_dir"] = str(Path(args.run_manifest_dir).expanduser().resolve())
        summary["engine_config_dir"] = str(args.engine_config_dir)
    else:
        summary["full_matrix"] = False

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for step in summary["steps"]:
            print(f"{step['key']}: job_id={step['job_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
