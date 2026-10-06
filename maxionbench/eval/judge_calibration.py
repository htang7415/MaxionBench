"""Judge calibration: a frozen set of 100 model answers, reference labels, and judge agreement.

    python -m maxionbench.eval.judge_calibration answers   # 60 CRAG + 40 HotpotQA answers (Gemini)
    (reference labels are then added to each row as "reference_label", before any judging)
    python -m maxionbench.eval.judge_calibration judge --repeats 3
    python -m maxionbench.eval.judge_calibration agreement

The reference labels were written by Claude (the coding assistant), not by independent human
annotators; the answering model and the judge are the same Gemini model. Both are disclosed with
the results. Question text comes from CRAG (CC BY-NC 4.0) and HotpotQA (CC BY-SA 4.0).
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

from maxionbench.datasets.loaders.v03 import load_crag
from maxionbench.eval.batch import Call, run_calls
from maxionbench.eval.qa import crag_item, hotpot_items
from maxionbench.graders.judge import RUBRIC_VERSION, agreement, cohen_kappa, judge_messages, parse_label
from maxionbench.harness.targets import GeminiTarget

CALIBRATION_SET = Path(__file__).resolve().parents[1] / "graders" / "calibration" / "qa_judge_v1.jsonl"
JUDGE_OUT = Path("artifacts/judge/qa_judge_v1.judgements.jsonl")
MODEL = "gemini-3.5-flash-lite"


def make_answers(n_crag: int, n_hotpot: int, seed: int, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"{out} exists; the calibration set is frozen once labelled")
    items = [crag_item(e) for e in random.Random(seed).sample(load_crag(), n_crag)]
    items += hotpot_items(Path("dataset/processed/hotpot_portable"), n_hotpot, seed)
    target = GeminiTarget({"model": MODEL, "reasoning_effort": "minimal"})
    calls = [Call(i.id, i.messages, max_tokens=64) for i in items]
    results, spend = run_calls(target, calls, "p4e/judge-calibration-answers")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            r = results[item.id]
            fh.write(json.dumps({"id": item.id, "source": item.source, "question": item.question,
                                 "golds": list(item.golds), "answer": r.text.strip(), "answer_status": r.status,
                                 "answer_model": MODEL}, ensure_ascii=False) + "\n")
    return spend


def run_judge(rows: list[dict[str, Any]], repeats: int, out: Path) -> dict[str, Any]:
    target = GeminiTarget({"model": MODEL, "reasoning_effort": "low"})
    calls = [Call(f"{row['id']}#{rep}", judge_messages(row["question"], row["golds"], row["answer"]), max_tokens=512)
             for rep in range(repeats) for row in rows]
    results, spend = run_calls(target, calls, "p4e/judge-calibration")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for call in calls:
            r = results[call.id]
            item_id, rep = call.id.rsplit("#", 1)
            fh.write(json.dumps({"id": item_id, "repeat": int(rep), "rubric": RUBRIC_VERSION, "judge_model": MODEL,
                                 "label": parse_label(r.text), "raw": r.text, "status": r.status}) + "\n")
    return spend


def report(rows: list[dict[str, Any]], judgements: list[dict[str, Any]]) -> dict[str, Any]:
    reference = {r["id"]: r["reference_label"] for r in rows}
    by_rep: dict[int, dict[str, str | None]] = {}
    for j in judgements:
        by_rep.setdefault(j["repeat"], {})[j["id"]] = j["label"]
    ids = sorted(reference)
    out: dict[str, Any] = {"rubric": RUBRIC_VERSION, "judge_model": MODEL, "reference": "Claude (not independent)",
                           "reference_counts": dict(Counter(reference.values())), "per_repeat": {}}
    for rep, labels in sorted(by_rep.items()):
        judged = [labels.get(i) or "unparsed" for i in ids]
        usable = [(reference[i], j) for i, j in zip(ids, judged) if j != "unparsed"]
        out["per_repeat"][rep] = {"unparsed": judged.count("unparsed"),
                                  **agreement([r for r, _ in usable], [j for _, j in usable])}
        for source in ("crag", "hotpot"):
            pairs = [(reference[i], labels[i]) for i in ids if i.startswith(source) and labels.get(i)]
            out["per_repeat"][rep][f"kappa_{source}"] = round(cohen_kappa(*zip(*pairs)), 4) if pairs else None
    reps = sorted(by_rep)
    if len(reps) > 1:  # judge self-consistency: kappa between repeat 0 and each later repeat
        base = by_rep[reps[0]]
        out["self_kappa"] = {
            rep: round(cohen_kappa(*zip(*[(base[i], by_rep[rep][i]) for i in ids if base.get(i) and by_rep[rep].get(i)])), 4)
            for rep in reps[1:]
        }
    return out


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("answers", "judge", "agreement"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "answers":
        print(json.dumps(make_answers(60, 40, args.seed, CALIBRATION_SET)))
        return 0
    rows = _read(CALIBRATION_SET)
    if any("reference_label" not in r for r in rows):
        raise SystemExit("label every row (reference_label) before judging, so labels stay blind to the judge")
    if args.command == "judge":
        print(json.dumps(run_judge(rows, args.repeats, JUDGE_OUT)))
        return 0
    print(json.dumps(report(rows, _read(JUDGE_OUT)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
