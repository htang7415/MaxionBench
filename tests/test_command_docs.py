from __future__ import annotations

from pathlib import Path


def test_command_md_is_concise_slurm_operator_doc() -> None:
    text = Path("command.md").read_text(encoding="utf-8")

    assert "# MaxionBench Slurm Commands" in text
    assert "download_datasets -> preprocess_datasets -> prefetch_datasets -> conformance -> calibrate_d3 -> benchmark arrays -> postprocess" in text
    assert "maxionbench/orchestration/slurm/profiles_local.yaml" in text
    assert ".env.slurm.euler" in text
    assert ".env.slurm.nrel" in text
    assert "run_slurm_pipeline.sh` auto-loads `.env.slurm.<cluster>` when present and refreshes that file with the resolved values for the current run." in text
    assert "The script derives shared paths automatically from `MAXIONBENCH_SHARED_ROOT` or from the repository root that contains `run_slurm_pipeline.sh`" in text
    assert "- `dataset/`" in text
    assert "- `.cache/`" in text
    assert "- `results/`" in text
    assert "- `figures/`" in text
    assert "- `.cache/huggingface/`" in text
    assert "--shared-root /shared/path/maxionbench" in text
    assert "bash run_slurm_pipeline.sh --cluster euler" in text
    assert "bash run_slurm_pipeline.sh --cluster nrel" in text
    assert "bash test_slrum_pipeline.sh --cluster nrel" in text
    assert "Dry-run only prints the submit plan." in text
    assert "`--launch` prepares the shared directory tree and ensures the required Apptainer images exist before it calls `submit-slurm-plan`." in text
    assert "run_slurm_pipeline.sh` rejects `--skip-gpu` and `MAXIONBENCH_ALLOW_GPU_UNAVAILABLE=1`" in text
    assert "It stages a reduced matrix and then calls `run_slurm_pipeline.sh` with `--allow-reduced-matrix`." in text
    assert "Copied example values such as `your-account`, `YOUR_PRIVATE_PARTITION`, or `/shared/containers/...` are rejected before submission." in text
    assert "Large Apptainer build cache/tmp data defaults to `${MAXIONBENCH_SHARED_ROOT}/.cache/apptainer`" in text
    assert "${MAXIONBENCH_SHARED_ROOT}/containers/maxionbench.sif" in text
    assert "--launch" in text
    assert 'squeue -u "$USER"' in text
    assert "sacct -j <job_id>" in text
    assert "tail -f logs/maxion_*.out" in text

    assert "run_workstation.sh" not in text
    assert "run_docker_scenario.sh" not in text


def test_command_mac_md_is_concise_local_terminal_doc() -> None:
    text = Path("command-mac.md").read_text(encoding="utf-8")

    assert "# MaxionBench Mac Mini M4 Commands" in text
    assert 'python -m pip install -e ".[dev,engines,reporting,datasets]"' in text
    assert "python -m maxionbench.cli download-datasets --root dataset --cache-dir .cache --crag-examples 500 --json" in text
    assert "## 3. Prepare datasets" in text
    assert "bash preprocess_all_datasets.sh" in text
    assert "python -m maxionbench.cli run --config configs/scenarios/calibrate_d3.yaml --seed 42 --repeats 1 --no-retry" in text
    assert "python -m maxionbench.cli run --config configs/scenarios/s1_ann_frontier_qdrant_local.yaml --seed 42 --repeats 1 --no-retry" in text
    assert "python -m maxionbench.cli run --config configs/scenarios/s2_filtered_ann.yaml --seed 42 --repeats 1 --no-retry --d3-params artifacts/calibration/d3_params.yaml" in text
    assert "python -m maxionbench.cli run --config configs/scenarios/s3_churn_smooth.yaml --seed 42 --repeats 1 --no-retry --d3-params artifacts/calibration/d3_params.yaml" in text
    assert "python -m maxionbench.cli run --config configs/scenarios/s3b_churn_bursty.yaml --seed 42 --repeats 1 --no-retry --d3-params artifacts/calibration/d3_params.yaml" in text
    assert "python -m maxionbench.cli run --config configs/scenarios/s4_hybrid.yaml --seed 42 --repeats 1 --no-retry" in text
    assert "python -m maxionbench.cli run --config configs/scenarios/s6_fusion.yaml --seed 42 --repeats 1 --no-retry" in text
    assert "python -m maxionbench.cli report --input artifacts/runs --mode milestones --milestone-id M1" in text
    assert "python -m maxionbench.cli report --input artifacts/runs --mode final --out artifacts/figures/final" in text

    assert "submit-slurm-plan" not in text
    assert "run_slurm_pipeline.sh" not in text
