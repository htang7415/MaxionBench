from __future__ import annotations

from typing import Any

import pytest

from maxionbench.eval.gateway_accuracy import ARM, compare


def test_compare_counts_discordant_pairs_and_cost_differences() -> None:
    def row(judge: bool, strict: bool, cost: float) -> dict[str, Any]:
        return {"judge": judge, "strict": strict, "cost_usd": cost, "prompt_tokens": 100, "cached_tokens": 50,
                "model_calls": 4}

    rows = {("a", ARM): row(True, True, 0.01), ("a", "full"): row(False, False, 0.03),
            ("b", ARM): row(True, False, 0.02), ("b", "full"): row(True, True, 0.02),
            ("c", ARM): row(False, False, 0.01)}  # c has no full run: not paired
    out = compare(rows, ARM, "full", ["a", "b", "c"])
    assert out["tasks"] == 2
    assert out["judge"] == {"accuracy": 1.0, "base_accuracy": 0.5, "wins": 1, "losses": 0, "p": 1.0}
    assert out["strict"]["wins"] == 1 and out["strict"]["losses"] == 1
    assert out["cost_usd"]["diff_mean"] == pytest.approx(-0.01) and out["cost_usd"]["rel"] == pytest.approx(-0.4)
    assert out["usd_per_solved_arm"] == pytest.approx(0.015) and out["usd_per_solved_base"] == pytest.approx(0.05)
