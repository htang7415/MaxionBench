"""Regrade a context-policy run two ways and test each policy against `full`.

Judge grading is what the run recorded; strict grading is the offline `agent_success` string match against
the gold answer. Each policy is compared with `full` on the same tasks with an exact two-sided McNemar test
on the discordant tasks, Holm-adjusted across policies. Needs the run's local-only answers.jsonl; writes
`regrade.json` with ids and numbers only.

    python -m maxionbench.eval.context_regrade artifacts/context_eval/<run>
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from math import comb
from pathlib import Path
from typing import Any, Sequence

import yaml

from maxionbench.agents.browsecomp_env import load_tasks
from maxionbench.graders.qa import agent_success


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of `wins` among the discordant pairs at 1/2."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    ordered = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for i, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * pvalues[name]))
        out[name] = running
    return out


def compare(correct: dict[tuple[str, str], bool], policies: Sequence[str]) -> dict[str, Any]:
    """Accuracy, accuracy difference vs full, wins/losses, raw and Holm-adjusted p per policy."""
    tasks = sorted({t for t, _ in correct})
    out: dict[str, Any] = {}
    for name in policies:
        paired = [(correct[(t, name)], correct[(t, "full")]) for t in tasks if (t, name) in correct and (t, "full") in correct]
        acc = sum(a for a, _ in paired) / len(paired)
        out[name] = {"tasks": len(paired), "accuracy": round(acc, 4)}
        if name != "full":
            wins, losses = sum(a and not b for a, b in paired), sum(b and not a for a, b in paired)
            out[name].update(delta=round((wins - losses) / len(paired), 4), wins=wins, losses=losses,
                             p=round(mcnemar_exact(wins, losses), 4))
    adjusted = holm({n: out[n]["p"] for n in policies if n != "full"})
    for name, p in adjusted.items():
        out[name]["p_holm"] = round(p, 4)
    return out


def regrade(run_dir: Path) -> dict[str, Any]:
    spec = yaml.safe_load((run_dir / "spec.yaml").read_text(encoding="utf-8"))
    items = [json.loads(line) for line in (run_dir / "items.jsonl").read_text(encoding="utf-8").splitlines() if line]
    answers = {(a["task_id"], a["policy"]): a["answer"] for line in
               (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line for a in [json.loads(line)]}
    n = int(spec["tasks"]) if spec.get("_task_limit") is None else min(int(spec["tasks"]), int(spec["_task_limit"]))
    gold = {t.task_id: t.answer for t in load_tasks(n, int(spec["seed"]), pool=int(spec["pool"]))}
    policies = list(spec["policies"])
    judge = {(it["task_id"], it["policy"]): bool(it["correct"]) for it in items}
    strict = {k: agent_success(answers.get(k), gold[k[0]]) for k in judge}
    agree = sum(judge[k] == strict[k] for k in judge) / len(judge)
    return {"run": run_dir.name, "answers": len(judge), "judge_strict_agreement": round(agree, 4),
            "judge": compare(judge, policies), "strict": compare(strict, policies)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Strict regrade and Holm-corrected McNemar tests of a context-policy run")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    result = regrade(args.run_dir)
    (args.run_dir / "regrade.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
