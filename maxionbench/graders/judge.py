"""LLM-as-judge for short-answer QA (CRAG/HotpotQA), with a fixed rubric and agreement statistics.

Labels follow CRAG's scoring: "correct" (+1), "missing" (0, the answer abstains), "incorrect" (-1).
The rubric is versioned so judge results can be tied to the exact instructions that produced them.
"""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Mapping, Sequence

LABELS = ("correct", "incorrect", "missing")
RUBRIC_VERSION = "qa-judge-v1"
RUBRIC = """You grade a model's answer to a question against the gold answer(s).

Labels:
- correct: the answer states the same fact as at least one gold answer. Paraphrases, equivalent
  names, formats and units (e.g. "2.1 million" vs "2,100,000"), and extra details that do not
  contradict the gold are fine. For a list, every gold item must be present. If the gold answer is
  "invalid question", the answer is correct only if it says the question's premise is false.
- missing: the answer abstains without committing to an answer ("I don't know", cannot find it,
  asks for clarification).
- incorrect: anything else: a different or contradicting fact, a partially correct answer, several
  hedged candidates, or a commitment to the premise of an invalid question.

Judge only against the gold answers, not your own knowledge. Reply with JSON only:
{"label": "correct" | "incorrect" | "missing", "reason": "<one short sentence>"}"""


def judge_messages(question: str, golds: Sequence[str], answer: str) -> tuple[dict[str, str], ...]:
    gold_text = "\n".join(f"- {g}" for g in golds)
    return (
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"Question: {question}\n\nGold answer(s):\n{gold_text}\n\nModel answer: {answer}"},
    )


def parse_label(text: str) -> str | None:
    """The label from the judge's JSON reply (tolerating code fences or stray text), else None."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            label = str(json.loads(match.group(0)).get("label", "")).strip().lower()
        except (json.JSONDecodeError, AttributeError):
            label = ""
        if label in LABELS:
            return label
    return None


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label sequences")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in labels) / (n * n)
    return 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)


def agreement(reference: Sequence[str], judged: Sequence[str]) -> dict[str, object]:
    """Accuracy, Cohen's kappa and the confusion matrix (rows: reference, columns: judge)."""
    confusion: Mapping[str, dict[str, int]] = {r: {j: 0 for j in LABELS} for r in LABELS}
    for r, j in zip(reference, judged):
        confusion[r][j] += 1
    return {
        "n": len(reference),
        "accuracy": round(sum(r == j for r, j in zip(reference, judged)) / len(reference), 4),
        "cohen_kappa": round(cohen_kappa(reference, judged), 4),
        "confusion": confusion,
    }
