"""Declarative experiment specs (YAML) with strict validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

SPEC_SCHEMA_VERSION = "maxionbench-experiment-v1"
SEED_STRATEGIES = ("per_repeat", "fixed")


@dataclass(frozen=True)
class ComponentSpec:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SLOPolicy:
    ttft_s: float
    e2e_s: float


@dataclass(frozen=True)
class QuietHost:
    max_load_1m: float = 3.0
    wait_s: float = 60.0
    min_available_gb: float = 0.0  # 0 disables; checked before starting a fresh engine


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    seed: int
    repeats: int
    seed_strategy: str
    target: ComponentSpec
    workload: ComponentSpec
    slo: SLOPolicy
    matrix: dict[str, list[Any]]
    quiet_host: QuietHost
    target_variants: dict[str, ComponentSpec] = field(default_factory=dict)
    reuse_targets: bool = False  # keep a started target across consecutive trials with identical config

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SPEC_SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "repeats": self.repeats,
            "seed_strategy": self.seed_strategy,
            "target": {"kind": self.target.kind, "params": dict(self.target.params)},
            "workload": {"kind": self.workload.kind, "params": dict(self.workload.params)},
            "slo": {"ttft_s": self.slo.ttft_s, "e2e_s": self.slo.e2e_s},
            "matrix": {k: list(v) for k, v in self.matrix.items()},
            "quiet_host": {
                "max_load_1m": self.quiet_host.max_load_1m,
                "wait_s": self.quiet_host.wait_s,
                "min_available_gb": self.quiet_host.min_available_gb,
            },
            "reuse_targets": self.reuse_targets,
            "target_variants": {
                name: {"kind": v.kind, "params": dict(v.params)} for name, v in self.target_variants.items()
            },
        }


def load_spec(path: Path) -> ExperimentSpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: experiment spec must be a mapping")
    return parse_spec(payload)


def parse_spec(payload: Mapping[str, Any]) -> ExperimentSpec:
    allowed = {
        "schema_version", "name", "description", "seed", "repeats", "seed_strategy",
        "target", "workload", "slo", "matrix", "quiet_host", "target_variants", "reuse_targets",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown spec keys: {sorted(unknown)}")
    if payload.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SPEC_SCHEMA_VERSION!r}")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    repeats = int(payload.get("repeats", 3))
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    seed_strategy = str(payload.get("seed_strategy", "per_repeat"))
    if seed_strategy not in SEED_STRATEGIES:
        raise ValueError(f"seed_strategy must be one of {SEED_STRATEGIES}")
    target = _component(payload, "target")
    workload = _component(payload, "workload")
    slo_raw = payload.get("slo") or {}
    slo = SLOPolicy(ttft_s=float(slo_raw["ttft_s"]), e2e_s=float(slo_raw["e2e_s"]))
    if slo.ttft_s <= 0 or slo.e2e_s <= 0:
        raise ValueError("slo thresholds must be > 0")
    variants_raw = payload.get("target_variants") or {}
    if not isinstance(variants_raw, Mapping):
        raise ValueError("target_variants must be a mapping of name -> {kind, params}")
    target_variants = {str(name): _component(variants_raw, str(name)) for name in variants_raw}
    matrix_raw = payload.get("matrix") or {}
    matrix: dict[str, list[Any]] = {}
    for key, values in matrix_raw.items():
        section, _, param = str(key).partition(".")
        if section not in ("target", "workload") or not param:
            raise ValueError(f"matrix key {key!r} must look like 'target.<param>' or 'workload.<param>'")
        if not isinstance(values, list) or not values:
            raise ValueError(f"matrix axis {key!r} must be a non-empty list")
        if key == "target.variant" and set(map(str, values)) - set(target_variants):
            raise ValueError(f"target.variant values must be defined in target_variants: {sorted(target_variants)}")
        matrix[str(key)] = list(values)
    qh = payload.get("quiet_host") or {}
    return ExperimentSpec(
        name=name,
        description=str(payload.get("description") or ""),
        seed=int(payload.get("seed", 42)),
        repeats=repeats,
        seed_strategy=seed_strategy,
        target=target,
        workload=workload,
        slo=slo,
        matrix=matrix,
        quiet_host=QuietHost(
            max_load_1m=float(qh.get("max_load_1m", QuietHost.max_load_1m)),
            wait_s=float(qh.get("wait_s", QuietHost.wait_s)),
            min_available_gb=float(qh.get("min_available_gb", QuietHost.min_available_gb)),
        ),
        target_variants=target_variants,
        reuse_targets=bool(payload.get("reuse_targets", False)),
    )


def _component(payload: Mapping[str, Any], key: str) -> ComponentSpec:
    raw = payload.get(key)
    if not isinstance(raw, Mapping) or not str(raw.get("kind") or "").strip():
        raise ValueError(f"{key}.kind is required")
    params = raw.get("params") or {}
    if not isinstance(params, Mapping):
        raise ValueError(f"{key}.params must be a mapping")
    return ComponentSpec(kind=str(raw["kind"]), params=dict(params))
