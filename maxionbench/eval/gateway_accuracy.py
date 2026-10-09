"""Step 3: the gateway's window+cache arm against `full` on up to 100 paired BrowseComp-Plus tasks.

c2a ran the gateway arm on the C1 tasks (paired with C1's `full` and other policies); c2b ran `full` and the
gateway arm on 50 new tasks. Both gradings (judge, strict string match), exact McNemar with Holm across the
comparisons, paired cost differences with 95% CIs, and $ per solved task. Writes ids and numbers only.

    python -m maxionbench.eval.gateway_accuracy <c1 run> <c2a run> <c2b run> --out <summary.json>
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import yaml

from maxionbench.eval.context_eval import _read_jsonl, spec_tasks
from maxionbench.eval.context_regrade import holm, mcnemar_exact
from maxionbench.graders.qa import agent_success
from maxionbench.harness.results import mean_ci

ARM = "gateway-window+cache"


def _load(run: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], str], dict[str, str]]:
    spec = yaml.safe_load((run / "spec.yaml").read_text(encoding="utf-8"))
    items = {(it["task_id"], it["policy"]): it for it in _read_jsonl(run / "items.jsonl")}
    answers = {(a["task_id"], a["policy"]): a["answer"] for a in _read_jsonl(run / "answers.jsonl")}
    gold = {t.task_id: t.answer for t in spec_tasks(spec, int(spec["tasks"]))}
    return items, answers, gold


def compare(rows: dict[tuple[str, str], dict[str, Any]], arm: str, base: str, tasks: list[str]) -> dict[str, Any]:
    pairs = [(rows[(t, arm)], rows[(t, base)]) for t in tasks if (t, arm) in rows and (t, base) in rows]
    out: dict[str, Any] = {"tasks": len(pairs)}
    for grading in ("judge", "strict"):
        a = [p[0][grading] for p in pairs]
        b = [p[1][grading] for p in pairs]
        wins = sum(x and not y for x, y in zip(a, b))
        losses = sum(y and not x for x, y in zip(a, b))
        out[grading] = {"accuracy": round(sum(a) / len(a), 4), "base_accuracy": round(sum(b) / len(b), 4),
                        "wins": wins, "losses": losses, "p": round(mcnemar_exact(wins, losses), 4)}
    for field in ("cost_usd", "prompt_tokens", "model_calls"):
        ci = mean_ci([float(x[field]) - float(y[field]) for x, y in pairs])
        base = sum(float(y[field]) for _, y in pairs) / len(pairs)
        out[field] = {"base_mean": round(base, 6), "diff_mean": round(ci.mean, 6),
                      "ci95": [round(ci.ci_low, 6), round(ci.ci_high, 6)], "rel": round(ci.mean / base, 4)}
    for name, side in (("arm", 0), ("base", 1)):
        cost = sum(p[side]["cost_usd"] for p in pairs)
        solved = sum(p[side]["judge"] for p in pairs)
        out[f"usd_per_solved_{name}"] = round(cost / solved, 4) if solved else None
        out[f"cached_share_{name}"] = round(sum(p[side]["cached_tokens"] for p in pairs)
                                            / sum(p[side]["prompt_tokens"] for p in pairs), 4)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Gateway window+cache arm vs full (Step 3)")
    parser.add_argument("c1", type=Path)
    parser.add_argument("c2a", type=Path)
    parser.add_argument("c2b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    sets: dict[str, list[str]] = {}
    for label, run in (("c1", args.c1), ("c2a", args.c2a), ("c2b", args.c2b)):
        items, answers, gold = _load(run)
        for key, it in items.items():
            rows[key] = {**it, "judge": bool(it["correct"]),
                                      "strict": agent_success(answers.get(key), gold[key[0]])}
        sets[label] = sorted(gold)
    c1_tasks, new_tasks = sets["c1"], sets["c2b"]
    comparisons = {
        "all_vs_full": compare(rows, ARM, "full", c1_tasks + new_tasks),
        "c1_tasks_vs_full": compare(rows, ARM, "full", c1_tasks),
        "new_tasks_vs_full": compare(rows, ARM, "full", new_tasks),
        "c1_tasks_vs_inprocess_window+cache": compare(rows, ARM, "window+cache", c1_tasks),
        "c1_tasks_vs_summarize": compare(rows, ARM, "summarize", c1_tasks),
    }
    tested = ["all_vs_full", "c1_tasks_vs_inprocess_window+cache", "c1_tasks_vs_summarize"]  # disjoint questions
    for grading in ("judge", "strict"):
        for name, p in holm({n: comparisons[n][grading]["p"] for n in tested}).items():
            comparisons[name][grading]["p_holm"] = round(p, 4)
    result = {"arm": ARM, "runs": [args.c1.name, args.c2a.name, args.c2b.name], "comparisons": comparisons}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
