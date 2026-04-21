"""RHU-hour normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RHUReferences:
    c_ref_vcpu: float = 96.0
    g_ref_gpu: float = 1.0
    r_ref_gib: float = 512.0
    d_ref_tb: float = 7.68


@dataclass(frozen=True)
class RHUWeights:
    w_c: float = 0.25
    w_g: float = 0.25
    w_r: float = 0.25
    w_d: float = 0.25


def rhu_rate(
    *,
    cpu_vcpu: float,
    gpu_count: float,
    ram_gib: float,
    disk_tb: float,
    refs: RHUReferences,
    weights: RHUWeights,
) -> float:
    return (
        weights.w_c * (cpu_vcpu / refs.c_ref_vcpu)
        + weights.w_g * (gpu_count / refs.g_ref_gpu)
        + weights.w_r * (ram_gib / refs.r_ref_gib)
        + weights.w_d * (disk_tb / refs.d_ref_tb)
    )


def rhu_hours(duration_s: float, rate: float) -> float:
    if duration_s <= 0:
        return 0.0
    return (duration_s / 3600.0) * rate
