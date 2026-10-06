"""Answer graders: HotpotQA/CRAG exact match and token F1, CRAG's three-way score, agent task success."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from maxionbench.rag.answer_metrics import exact_match, normalize_answer, token_f1

_MISSING = {"i dont know", "i do not know", "unknown", "not sure", "cannot answer", "i cannot answer"}


@dataclass(frozen=True)
class QAGrade:
    em: float
    f1: float


def grade_qa(prediction: str, golds: Sequence[str]) -> QAGrade:
    """Best EM and best F1 over all gold answers (SQuAD/HotpotQA convention)."""
    if not golds:
        raise ValueError("need at least one gold answer")
    return QAGrade(max(exact_match(prediction, g) for g in golds), max(token_f1(prediction, g) for g in golds))


def is_missing(prediction: str) -> bool:
    return normalize_answer(prediction) in _MISSING or not normalize_answer(prediction)


def crag_label(prediction: str, answer: str, alt_ans: Sequence[str] = ()) -> str:
    """Offline CRAG label: "correct" (exact match), "missing" (abstained), else "incorrect".

    CRAG scores correct +1, missing 0, incorrect -1. Exact match under-credits paraphrases, so this
    is a strict lower bound; the Gemini judge re-labels the "incorrect" bucket.
    """
    if is_missing(prediction):
        return "missing"
    return "correct" if grade_qa(prediction, [answer, *alt_ans]).em == 1.0 else "incorrect"


def crag_score(labels: Sequence[str]) -> float:
    points = {"correct": 1.0, "missing": 0.0, "incorrect": -1.0}
    return sum(points[label] for label in labels) / len(labels) if labels else 0.0


def agent_success(prediction: str | None, gold: str) -> bool:
    """Exact match, or a short answer that contains the gold answer as a whole-token span.

    Containment accepts "Chief of Protocol of the United States" for "Chief of Protocol" while the
    length cap rejects answers that hedge by listing many candidates. yes/no needs exact match.
    """
    if prediction is None:
        return False
    if exact_match(prediction, gold):
        return True
    pred, want = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not want or want in (["yes"], ["no"]) or len(pred) > 2 * len(want) + 5:
        return False
    return any(pred[i:i + len(want)] == want for i in range(len(pred) - len(want) + 1))
