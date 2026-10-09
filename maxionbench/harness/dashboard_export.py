"""Copy the latest results.json of each published experiment into the dashboard's static data dir.

The dashboard reads only these files (plus index.json); nothing is fetched at runtime. Each file is
validated against the result model first, so a stale or hand-edited bundle fails the export.

    python -m maxionbench.harness.dashboard_export [--out dashboard/public/data]
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

from maxionbench.harness.results import ExperimentResult, from_dict

# experiment name -> dashboard page
EXPERIMENTS = {
    "e1-engines-gpu": "engines",
    "e1-engines-cpu": "engines",
    "e2-prefix-caching": "caching",
    "e6-gemini-caching": "caching",
    "e3-llmd-sim": "scheduling",
    "e3-llmd-metal": "scheduling",
    "e4-hybrid-gateway": "hybrid",
    "e4b-slo-overflow": "hybrid",
    "e5-gemini": "quality",
    "e5-qwen3-4b": "quality",
    "c1-context-policies": "context",
    "k6-copilot-context-policies": "context",
    "k7-llmd-copilot-context-policies": "context",
    "k8-vllm-metal-copilot-context-policies": "context",
    "k9-gateway-context-qwen3-8b": "gateway",
    "k9b-gateway-mask-min-growth-qwen3-8b": "gateway",
    "c2a-gateway-context": "gateway",
    "c2b-gateway-context": "gateway",
}
SEARCH_DIRS = (Path("artifacts/harness"), Path("artifacts/e5"), Path("artifacts/e6"), Path("artifacts/kvsim"),
               Path("artifacts/context_eval"))


def latest_results(search_dirs: tuple[Path, ...] = SEARCH_DIRS) -> dict[str, Path]:
    """Newest complete results.json per experiment name (run ids start with a UTC timestamp)."""
    found: dict[str, tuple[str, Path]] = {}
    for root in search_dirs:
        for path in sorted(root.glob("*/results.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            tools = data["provenance"]["tools"]
            if "trials_planned" in tools and tools.get("trials_completed") != tools["trials_planned"]:
                continue  # interrupted or still running (runners without a plan write results only when done)
            name, run_id = data["name"], data["run_id"]
            if name in EXPERIMENTS and (name not in found or run_id > found[name][0]):
                found[name] = (run_id, path)
    return {name: path for name, (_, path) in found.items()}


def export(out_dir: Path, search_dirs: tuple[Path, ...] = SEARCH_DIRS) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, path in sorted(latest_results(search_dirs).items()):
        data = json.loads(path.read_text(encoding="utf-8"))
        result: ExperimentResult = from_dict(ExperimentResult, data)  # strict schema check
        (out_dir / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
        entries.append({"name": name, "page": EXPERIMENTS[name], "run_id": result.run_id, "file": f"{name}.json",
                        "git_commit": result.provenance.git_commit, "git_dirty": result.provenance.git_dirty,
                        "finished_at": result.provenance.finished_at})
    index = {"experiments": entries}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("dashboard/public/data"))
    args = parser.parse_args(argv)
    index = export(args.out)
    for e in index["experiments"]:
        print(f"{e['page']:<11} {e['name']:<20} {e['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
