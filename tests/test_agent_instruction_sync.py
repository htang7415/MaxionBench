from __future__ import annotations

from pathlib import Path


def test_agents_and_claude_instruction_files_are_in_sync() -> None:
    agents = Path("AGENTS.md")
    claude = Path("CLAUDE.md")
    assert agents.exists(), "AGENTS.md must exist"
    assert claude.exists(), "CLAUDE.md must exist"
    assert agents.read_text(encoding="utf-8") == claude.read_text(encoding="utf-8")
