from __future__ import annotations

import json
from pathlib import Path
import shutil

from maxionbench.cli import main as cli_main
from maxionbench.tools.verify_behavior_cards import verify_behavior_cards


def test_verify_behavior_cards_passes_for_repo_behavior_docs() -> None:
    summary = verify_behavior_cards(Path("docs/behavior"))
    assert summary["pass"] is True
    assert int(summary["files_checked"]) >= 1
    assert int(summary["error_count"]) == 0


def test_verify_behavior_cards_detects_missing_required_section(tmp_path: Path) -> None:
    src = Path("docs/behavior")
    dst = tmp_path / "behavior"
    shutil.copytree(src, dst)

    qdrant_card = dst / "qdrant.md"
    text = qdrant_card.read_text(encoding="utf-8")
    text = text.replace("## Persistence", "## Durability")
    qdrant_card.write_text(text, encoding="utf-8")

    summary = verify_behavior_cards(dst)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("persistence" in msg.lower() for msg in messages)


def test_verify_behavior_cards_detects_lancedb_split_card_violation(tmp_path: Path) -> None:
    src = Path("docs/behavior")
    dst = tmp_path / "behavior"
    shutil.copytree(src, dst)
    (dst / "lancedb_service.md").write_text("# invalid split card\n", encoding="utf-8")

    summary = verify_behavior_cards(dst)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("lancedb policy violation" in msg.lower() for msg in messages)


def test_verify_behavior_cards_cli_reports_failure_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src = Path("docs/behavior")
    dst = tmp_path / "behavior"
    shutil.copytree(src, dst)
    (dst / "lancedb.md").write_text("# broken card\n", encoding="utf-8")

    code = cli_main(["verify-behavior-cards", "--behavior-dir", str(dst), "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert int(parsed["error_count"]) >= 1
