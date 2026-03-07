"""Submit Slurm jobs with enforced MaxionBench dependency ordering."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml


@dataclass(frozen=True)
class SubmitStep:
    key: str
    script_name: str
    array: str | None = None
    depends_on: tuple[str, ...] = ()


SLURM_PROFILE_OVERRIDES_ENV = "MAXIONBENCH_SLURM_PROFILE_OVERRIDES"
DEFAULT_LOCAL_PROFILE_OVERRIDES_PATH = Path(__file__).resolve().with_name("profiles_local.yaml")


def build_submit_steps(*, include_gpu: bool = True, skip_s6: bool = False) -> list[SubmitStep]:
    cpu_non_d3_array = "0,5" if skip_s6 else "0,5-6"
    steps: list[SubmitStep] = [
        SubmitStep(
            key="calibrate",
            script_name="calibrate_d3.sh",
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
    if include_gpu:
        steps.append(
            SubmitStep(
                key="gpu_all",
                script_name="gpu_array.sh",
                array="0-2",
                depends_on=("calibrate",),
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
    dry_run: bool = False,
) -> dict[str, Any]:
    normalized_container_runtime = str(container_runtime).strip().lower() if container_runtime else None
    normalized_container_image = str(container_image).strip() if container_image else None
    normalized_hf_cache_dir = str(hf_cache_dir).strip() if hf_cache_dir else None
    normalized_container_bind = [str(item).strip() for item in (container_bind or []) if str(item).strip()]
    if normalized_container_runtime and normalized_container_runtime != "apptainer":
        raise ValueError(f"unsupported container runtime: {normalized_container_runtime}")
    if normalized_container_runtime and not normalized_container_image:
        raise ValueError("container_image is required when container_runtime is set")
    if normalized_container_image and not normalized_container_runtime:
        raise ValueError("container_runtime is required when container_image is set")

    resolved_dir = slurm_dir.resolve()
    job_ids: dict[str, str] = {}
    submitted: list[dict[str, Any]] = []
    for step in steps:
        script_path = (resolved_dir / step.script_name).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"missing slurm script: {script_path}")
        cmd = _build_sbatch_command(
            step_key=step.key,
            script_path=script_path,
            array=step.array,
            seed=seed,
            scenario_config_dir=scenario_config_dir,
            output_root=output_root,
            slurm_profile=slurm_profile,
            container_runtime=normalized_container_runtime,
            container_image=normalized_container_image,
            container_bind=normalized_container_bind,
            hf_cache_dir=normalized_hf_cache_dir,
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
        "steps": submitted,
        "job_ids": job_ids,
    }


def _build_sbatch_command(
    *,
    step_key: str,
    script_path: Path,
    array: str | None,
    seed: int | None,
    scenario_config_dir: str | None,
    output_root: str | None,
    slurm_profile: str | None,
    container_runtime: str | None,
    container_image: str | None,
    container_bind: list[str],
    hf_cache_dir: str | None,
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
            f"{slurm_profile}. Define it in {DEFAULT_LOCAL_PROFILE_OVERRIDES_PATH} "
            f"or via {SLURM_PROFILE_OVERRIDES_ENV}"
        )
    local_profile = profiles[slurm_profile]
    base_specs = _normalize_flag_specs(local_profile.get("base"), label=f"{slurm_profile}.base")
    overrides = _normalize_step_override_specs(local_profile.get("step_overrides"), step_key=step_key, profile=slurm_profile)
    args: list[str] = []
    for flag, value in (*base_specs, *overrides):
        args.extend([flag, value])
    return args


def _load_local_profile_overrides() -> dict[str, dict[str, Any]]:
    raw_path = os.environ.get(SLURM_PROFILE_OVERRIDES_ENV)
    candidate = Path(raw_path).expanduser() if raw_path else DEFAULT_LOCAL_PROFILE_OVERRIDES_PATH
    if not candidate.exists():
        return {}
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
        value = str(item[1]).strip()
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
    return _normalize_flag_specs(raw_payload.get(step_key), label=f"{profile}.step_overrides.{step_key}")


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
        "--skip-s6",
        action="store_true",
        help="Defer S6 by removing index 6 from cpu_non_d3 array submissions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    steps = build_submit_steps(
        include_gpu=not bool(args.skip_gpu),
        skip_s6=bool(args.skip_s6),
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
        dry_run=bool(args.dry_run),
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for step in summary["steps"]:
            print(f"{step['key']}: job_id={step['job_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
