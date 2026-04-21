from __future__ import annotations

from pathlib import Path


def test_command_md_is_portable_agentic_operator_doc() -> None:
    text = Path("command.md").read_text(encoding="utf-8")

    assert "# MaxionBench Portable-Agentic Commands" in text
    assert "install -> services -> conformance -> data+embeddings -> submit -> report -> archive" in text
    assert "All run arguments live in config." in text
    assert "Docker is still used only for the service-backed engines in the Mac mini lane: `qdrant` and `pgvector`." in text
    assert "Use this command once:" in text
    assert 'pip install -e ".[dev,engines,reporting,datasets,embeddings]"' in text
    assert "maxionbench portable-workflow setup --json" in text
    assert "maxionbench portable-workflow data --json" in text
    assert "maxionbench submit-portable --budget b0 --json" in text
    assert "maxionbench submit-portable --budget b1 --json" in text
    assert "maxionbench submit-portable --budget b2 --json" in text
    assert "Use one command per budget:" in text
    assert "`submit-portable` wraps `run-matrix` + `execute-run-matrix` for the Mac mini lane." in text
    assert "configs/scenarios_portable" in text
    assert "configs/engines_portable" in text
    assert "maxionbench portable-workflow finalize --json" in text

    assert "run_docker_scenario.sh" not in text
    assert "run_workstation.sh" not in text
    assert "save_results_bundle.sh" not in text
