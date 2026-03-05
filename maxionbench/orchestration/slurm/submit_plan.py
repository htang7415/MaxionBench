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


def build_submit_steps(*, include_gpu: bool = True) -> list[SubmitStep]:
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
            array="0,5-6",
            depends_on=("calibrate",),
        ),
    ]
    if include_gpu:
        steps.append(
            SubmitStep(
                key="gpu_all",
                script_name="gpu_array.sh",
                array="0-1",
                depends_on=("calibrate",),
            )
        )
    return steps


def submit_steps(
    *,
    slurm_dir: Path,
    steps: list[SubmitStep],
    seed: int | None = None,
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
            script_path=script_path,
            array=step.array,
            seed=seed,
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
        "steps": submitted,
        "job_ids": job_ids,
    }


def _build_sbatch_command(
    *,
    script_path: Path,
    array: str | None,
    seed: int | None,
    dependencies: tuple[str, ...],
) -> list[str]:
    cmd = ["sbatch", "--parsable"]
    if dependencies:
        cmd.extend(["--dependency", f"afterok:{':'.join(dependencies)}"])
    if array:
        cmd.extend(["--array", array])
    if seed is not None:
        cmd.extend(["--export", f"ALL,MAXIONBENCH_SEED={int(seed)}"])
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


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Submit MaxionBench Slurm plan with enforced dependencies")
    parser.add_argument(
        "--slurm-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing calibrate_d3.sh/cpu_array.sh/gpu_array.sh",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    steps = build_submit_steps(include_gpu=not bool(args.skip_gpu))
    summary = submit_steps(
        slurm_dir=Path(args.slurm_dir),
        steps=steps,
        seed=int(args.seed),
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
