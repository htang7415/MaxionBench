"""Submit Slurm jobs with enforced MaxionBench dependency ordering."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any


@dataclass(frozen=True)
class SubmitStep:
    key: str
    script_name: str
    array: str | None = None
    depends_on: tuple[str, ...] = ()


SLURM_PROFILE_BASE_SPECS: dict[str, tuple[tuple[str, str], ...]] = {
    "your_cluster": (
        ("--job-name", "maxion"),
        ("--output", "logs/%x_%j.out"),
        ("--error", "logs/%x_%j.err"),
        ("--nodes", "1"),
        ("--ntasks-per-node", "1"),
        ("--cpus-per-task", "64"),
        ("--mem", "164G"),
        ("--partition", "pdelab"),
        ("--time", "8-00:00:00"),
    ),
    "your_cluster": (
        ("--account", "nawimem"),
        ("--time", "2-00:00:00"),
        ("--job-name", "maxion"),
        ("--output", "logs/%x_%j.out"),
        ("--error", "logs/%x_%j.err"),
        ("--mem", "256G"),
        ("--nodes", "1"),
        ("--ntasks-per-node", "1"),
        ("--cpus-per-task", "64"),
    ),
}

SLURM_PROFILE_STEP_OVERRIDES: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "your_cluster": {
        "calibrate": (("--gres", "gpu:1"),),
        "gpu_all": (("--gres", "gpu:1"),),
    },
    "your_cluster": {
        "calibrate": (("--gres", "gpu:1"),),
        "gpu_all": (("--gres", "gpu:1"),),
    },
}


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
    dry_run: bool = False,
) -> dict[str, Any]:
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
    if slurm_profile not in SLURM_PROFILE_BASE_SPECS:
        raise ValueError(f"unknown slurm profile: {slurm_profile}")
    base_specs = SLURM_PROFILE_BASE_SPECS[slurm_profile]
    overrides = SLURM_PROFILE_STEP_OVERRIDES.get(slurm_profile, {}).get(step_key, ())
    args: list[str] = []
    for flag, value in (*base_specs, *overrides):
        args.extend([flag, value])
    return args


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
        choices=sorted(SLURM_PROFILE_BASE_SPECS.keys()),
        help="Optional Slurm profile preset for cluster-specific sbatch flags",
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
