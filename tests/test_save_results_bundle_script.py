from __future__ import annotations

from pathlib import Path
import subprocess


def test_save_results_bundle_script_exists_and_mentions_archive_contract() -> None:
    script = Path("save_results_bundle.sh")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert 'RUN_BUNDLE="artifacts/workstation_runs/latest"' in text
    assert 'RESULTS_ROOT="results"' in text
    assert 'DATASET_ROOT="dataset"' in text
    assert "--copy-datasets" in text
    assert "archive_manifest.txt" in text
    assert 'ln -sfn "${RUN_ID}" "${RESULTS_ROOT_ABS}/latest"' in text


def test_save_results_bundle_script_is_bash_parseable() -> None:
    completed = subprocess.run(
        ["bash", "-n", "save_results_bundle.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
