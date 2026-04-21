"""Resource normalization helpers for RHU accounting."""

from __future__ import annotations

from dataclasses import dataclass

from maxionbench.metrics.cost_rhu import RHUReferences, RHUWeights, rhu_rate
from maxionbench.schemas.adapter_contract import AdapterStats

_GIB = 1024.0**3
_TIB = 1024.0**4


@dataclass(frozen=True)
class ResourceProfile:
    """Normalized resource inputs used for RHU-rate computation."""

    cpu_vcpu: float
    gpu_count: float
    ram_gib: float
    disk_tb: float


def profile_from_adapter_stats(
    *,
    stats: AdapterStats,
    client_count: int,
    gpu_count: float = 0.0,
    min_cpu_vcpu: float = 1.0,
) -> ResourceProfile:
    """Derive a stable resource profile from adapter stats and run load."""

    if client_count < 0:
        raise ValueError("client_count must be non-negative")
    if min_cpu_vcpu <= 0:
        raise ValueError("min_cpu_vcpu must be positive")
    return ResourceProfile(
        cpu_vcpu=max(float(client_count), float(min_cpu_vcpu)),
        gpu_count=max(0.0, float(gpu_count)),
        ram_gib=max(0.0, float(stats.ram_usage_bytes) / _GIB),
        disk_tb=max(0.0, float(stats.disk_usage_bytes) / _TIB),
    )


def rhu_rate_for_profile(*, profile: ResourceProfile, refs: RHUReferences, weights: RHUWeights) -> float:
    """Compute RHU rate for a previously derived resource profile."""

    return rhu_rate(
        cpu_vcpu=profile.cpu_vcpu,
        gpu_count=profile.gpu_count,
        ram_gib=profile.ram_gib,
        disk_tb=profile.disk_tb,
        refs=refs,
        weights=weights,
    )
