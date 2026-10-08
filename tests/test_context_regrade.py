from __future__ import annotations

import pytest

from maxionbench.eval.context_regrade import compare, holm, mcnemar_exact


def test_mcnemar_exact_matches_binomial() -> None:
    assert mcnemar_exact(9, 1) == pytest.approx(2 * 11 / 1024)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(1, 9) == mcnemar_exact(9, 1)


def test_holm_is_monotone_step_down() -> None:
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_compare_pairs_on_tasks() -> None:
    correct = {("t1", "full"): False, ("t1", "x"): True, ("t2", "full"): True, ("t2", "x"): True,
               ("t3", "full"): True, ("t3", "x"): False, ("t4", "full"): False, ("t4", "x"): True}
    out = compare(correct, ["full", "x"])
    assert out["full"]["accuracy"] == 0.5
    assert out["x"] == {"tasks": 4, "accuracy": 0.75, "delta": 0.25, "wins": 2, "losses": 1,
                        "p": 1.0, "p_holm": 1.0}
