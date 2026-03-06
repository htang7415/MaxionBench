from __future__ import annotations

from maxionbench.datasets.d3_calibrate import is_trivial_curve


def test_is_trivial_curve_matches_documented_or_criterion() -> None:
    assert is_trivial_curve(p99_ratio_1pct_to_50pct=1.99, recall_gap_50_minus_1=0.10) is True
    assert is_trivial_curve(p99_ratio_1pct_to_50pct=2.10, recall_gap_50_minus_1=0.049) is True
    assert is_trivial_curve(p99_ratio_1pct_to_50pct=2.10, recall_gap_50_minus_1=0.10) is False


def test_is_trivial_curve_thresholds_are_strictly_less_than() -> None:
    assert is_trivial_curve(p99_ratio_1pct_to_50pct=2.0, recall_gap_50_minus_1=0.05) is False
    assert is_trivial_curve(p99_ratio_1pct_to_50pct=2.0, recall_gap_50_minus_1=0.049999) is True
    assert is_trivial_curve(p99_ratio_1pct_to_50pct=1.999999, recall_gap_50_minus_1=0.05) is True
