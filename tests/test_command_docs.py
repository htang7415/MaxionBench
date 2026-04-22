from __future__ import annotations

from pathlib import Path


def test_command_md_is_portable_agentic_operator_doc() -> None:
    text = Path("command.md").read_text(encoding="utf-8")

    assert "# MaxionBench Commands" in text
    assert "The Mac mini is the execution host only, not part of the benchmark storyline." in text
    assert "Target: complete the local workflow within one day." in text
    assert 'pip install -e ".[dev,engines,reporting,datasets,embeddings]"' in text
    assert 'pip install -e ".[dev,engines,reporting,datasets,embeddings]" "numpy==1.26.4" "transformers<5"' in text
    assert "Intel macOS only:" in text
    assert "maxionbench --help" in text
    assert "maxionbench verify-pins --json" in text
    assert "maxionbench verify-dataset-manifests --json" in text
    assert "maxionbench verify-conformance-configs --json" in text
    assert "maxionbench portable-workflow setup --json" in text
    assert "maxionbench portable-workflow data --json" in text
    assert "maxionbench submit-portable --budget b0 --json" in text
    assert "maxionbench submit-portable --budget b1 --json" in text
    assert "maxionbench submit-portable --budget b2 --json" in text
    assert "maxionbench portable-workflow finalize --json" in text

    assert "run_docker_scenario.sh" not in text
    assert "run_workstation.sh" not in text
    assert "save_results_bundle.sh" not in text
