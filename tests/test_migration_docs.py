from __future__ import annotations

from pathlib import Path


def _readme_migration_entries(readme_text: str) -> list[str]:
    lines = readme_text.splitlines()
    try:
        start = lines.index("Migration details:")
    except ValueError:
        raise AssertionError("README.md missing `Migration details:` section")

    entries: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            if entries:
                break
            continue
        if not line.startswith("- "):
            if entries:
                break
            continue
        tick_start = line.find("`")
        tick_end = line.rfind("`")
        if tick_start == -1 or tick_end <= tick_start:
            continue
        entries.append(line[tick_start + 1 : tick_end])
    return entries


def test_readme_migration_index_matches_docs_directory() -> None:
    readme = Path("README.md")
    assert readme.exists()
    migration_dir = Path("docs/migrations")
    assert migration_dir.exists()

    readme_text = readme.read_text(encoding="utf-8")
    indexed = _readme_migration_entries(readme_text)
    indexed_migration = sorted([item for item in indexed if item.startswith("docs/migrations/")])

    on_disk = sorted([str(path.as_posix()) for path in migration_dir.glob("*.md")])

    assert indexed_migration == on_disk


def test_readme_migration_section_references_branch_protection_doc() -> None:
    readme_text = Path("README.md").read_text(encoding="utf-8")
    indexed = _readme_migration_entries(readme_text)
    assert "docs/ci/branch_protection.md" in indexed
