from __future__ import annotations

import json
from pathlib import Path

from maxionbench.cli import main as cli_main


def test_archive_cli_uses_hotpot_override(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    for name in ("command.md", "project.md", "prompt.md", "document.md"):
        (repo_root / name).write_text(f"# {name}\n", encoding="utf-8")
    hotpot_dir = repo_root / "dataset" / "processed" / "hotpot_portable"
    hotpot_dir.mkdir(parents=True)
    (hotpot_dir / "meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(repo_root)
    code = cli_main(
        [
            "archive",
            "--results-dir",
            str(repo_root / "results"),
            "--hotpot-maxionbench-dir",
            str(hotpot_dir.relative_to(repo_root)),
            "--json",
            "--no-tar",
        ]
    )
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    items = {str(item.get("label")): item for item in parsed["items"]}
    assert "hotpot_maxionbench" in items
    assert items["hotpot_maxionbench"]["src"].endswith("dataset/processed/hotpot_portable")
