"""Conformance matrix orchestration across adapter configs."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ConformanceMatrixRow:
    adapter: str
    config_file: str
    status: str
    exit_code: int
    duration_s: float
    command: str
    note: str | None = None


def run_conformance_matrix(*, config_dir: Path, out_dir: Path, timeout_s: float = 300.0) -> list[ConformanceMatrixRow]:
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        raise FileNotFoundError(f"No conformance configs found in {config_dir}")

    rows: list[ConformanceMatrixRow] = []
    for cfg_path in configs:
        payload = _read_json_mapping(cfg_path)
        adapter = str(payload.get("adapter", "")).strip()
        if not adapter:
            rows.append(
                ConformanceMatrixRow(
                    adapter="",
                    config_file=str(cfg_path),
                    status="invalid_config",
                    exit_code=2,
                    duration_s=0.0,
                    command="",
                    note="missing adapter key",
                )
            )
            continue

        argv = [
            sys.executable,
            "-m",
            "maxionbench.conformance.run",
            "--adapter",
            adapter,
            "--adapter-options-json",
            str(payload.get("adapter_options_json", "{}")),
            "--collection",
            str(payload.get("collection", "conformance")),
            "--dimension",
            str(payload.get("dimension", 4)),
            "--metric",
            str(payload.get("metric", "ip")),
        ]
        command = " ".join(argv)
        started = datetime.now(tz=timezone.utc)
        try:
            proc = subprocess.run(
                argv,
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            ended = datetime.now(tz=timezone.utc)
            duration_s = max((ended - started).total_seconds(), 0.0)
            status = "pass" if proc.returncode == 0 else "fail"
            note = _truncate((proc.stdout or "") + "\n" + (proc.stderr or ""))
            rows.append(
                ConformanceMatrixRow(
                    adapter=adapter,
                    config_file=str(cfg_path),
                    status=status,
                    exit_code=int(proc.returncode),
                    duration_s=duration_s,
                    command=command,
                    note=note,
                )
            )
        except subprocess.TimeoutExpired:
            ended = datetime.now(tz=timezone.utc)
            duration_s = max((ended - started).total_seconds(), 0.0)
            rows.append(
                ConformanceMatrixRow(
                    adapter=adapter,
                    config_file=str(cfg_path),
                    status="timeout",
                    exit_code=124,
                    duration_s=duration_s,
                    command=command,
                    note=f"timeout after {timeout_s}s",
                )
            )

    _write_outputs(rows=rows, out_dir=out_dir, config_dir=config_dir)
    return rows


def _write_outputs(*, rows: list[ConformanceMatrixRow], out_dir: Path, config_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in rows])
    frame = frame.sort_values(["adapter", "config_file"], kind="stable").reset_index(drop=True)

    csv_path = out_dir / "conformance_matrix.csv"
    frame.to_csv(csv_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config_dir": str(config_dir),
        "rows": int(len(frame)),
        "status_counts": frame["status"].value_counts(dropna=False).to_dict() if not frame.empty else {},
    }
    json_path = out_dir / "conformance_matrix.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "rows": [asdict(row) for row in rows],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _read_json_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(payload)


def _truncate(text: str, max_len: int = 4000) -> str:
    clean = text.strip()
    if len(clean) <= max_len:
        return clean
    return clean[:max_len] + "...(truncated)"


def parse_args(argv: list[str] | None = None) -> ArgumentParser:
    parser = ArgumentParser(description="Run MaxionBench adapter conformance matrix")
    parser.add_argument("--config-dir", default="configs/conformance")
    parser.add_argument("--out-dir", default="artifacts/conformance")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = parse_args(argv)
    args = parser.parse_args(argv)
    run_conformance_matrix(
        config_dir=Path(args.config_dir).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        timeout_s=float(args.timeout_s),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
