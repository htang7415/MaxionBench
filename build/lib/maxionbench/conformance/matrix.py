"""Conformance matrix orchestration across adapter configs."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

import pandas as pd

from maxionbench.conformance.provenance import build_conformance_provenance, conformance_provenance_path


@dataclass(frozen=True)
class ConformanceMatrixRow:
    adapter: str
    config_file: str
    status: str
    exit_code: int
    duration_s: float
    command: str
    note: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None


def run_conformance_matrix(*, config_dir: Path, out_dir: Path, timeout_s: float = 300.0) -> list[ConformanceMatrixRow]:
    return _run_conformance_matrix(config_dir=config_dir, out_dir=out_dir, timeout_s=timeout_s, adapters=None)


def _run_conformance_matrix(
    *,
    config_dir: Path,
    out_dir: Path,
    timeout_s: float,
    adapters: Sequence[str] | None,
) -> list[ConformanceMatrixRow]:
    resolved_config_dir = config_dir.resolve()
    resolved_out_dir = out_dir.resolve()
    artifacts_dir = resolved_out_dir / "adapter_logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if not resolved_config_dir.exists():
        raise FileNotFoundError(f"Conformance config directory does not exist: {resolved_config_dir}")
    if not resolved_config_dir.is_dir():
        raise FileNotFoundError(f"Conformance config path is not a directory: {resolved_config_dir}")

    configs = sorted(resolved_config_dir.glob("*.json"))
    if not configs:
        raise FileNotFoundError(f"No conformance config files (*.json) found in: {resolved_config_dir}")

    rows: list[ConformanceMatrixRow] = []
    adapter_filter = _normalize_adapters(adapters)
    for cfg_path in configs:
        try:
            payload = _read_json_mapping(cfg_path)
        except (ValueError, json.JSONDecodeError) as exc:
            stdout_path, stderr_path = _write_adapter_artifacts(
                artifacts_dir=artifacts_dir,
                adapter="invalid-config",
                config_file=cfg_path,
                stdout="",
                stderr=str(exc),
            )
            rows.append(
                ConformanceMatrixRow(
                    adapter="",
                    config_file=str(cfg_path),
                    status="invalid_config",
                    exit_code=2,
                    duration_s=0.0,
                    command="",
                    note=_truncate(str(exc)),
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            )
            continue
        adapter = str(payload.get("adapter", "")).strip()
        if not adapter:
            stdout_path, stderr_path = _write_adapter_artifacts(
                artifacts_dir=artifacts_dir,
                adapter="missing-adapter",
                config_file=cfg_path,
                stdout="",
                stderr="missing adapter key",
            )
            rows.append(
                ConformanceMatrixRow(
                    adapter="",
                    config_file=str(cfg_path),
                    status="invalid_config",
                    exit_code=2,
                    duration_s=0.0,
                    command="",
                    note="missing adapter key",
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            )
            continue
        if adapter_filter is not None and adapter not in adapter_filter:
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
            stdout_text = _coerce_text(proc.stdout)
            stderr_text = _coerce_text(proc.stderr)
            stdout_path, stderr_path = _write_adapter_artifacts(
                artifacts_dir=artifacts_dir,
                adapter=adapter,
                config_file=cfg_path,
                stdout=stdout_text,
                stderr=stderr_text,
            )
            note = _truncate(stdout_text + "\n" + stderr_text)
            rows.append(
                ConformanceMatrixRow(
                    adapter=adapter,
                    config_file=str(cfg_path),
                    status=status,
                    exit_code=int(proc.returncode),
                    duration_s=duration_s,
                    command=command,
                    note=note,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            )
        except subprocess.TimeoutExpired as exc:
            ended = datetime.now(tz=timezone.utc)
            duration_s = max((ended - started).total_seconds(), 0.0)
            stdout_text = _coerce_text(exc.stdout)
            stderr_text = _coerce_text(exc.stderr)
            stdout_path, stderr_path = _write_adapter_artifacts(
                artifacts_dir=artifacts_dir,
                adapter=adapter,
                config_file=cfg_path,
                stdout=stdout_text,
                stderr=stderr_text,
            )
            rows.append(
                ConformanceMatrixRow(
                    adapter=adapter,
                    config_file=str(cfg_path),
                    status="timeout",
                    exit_code=124,
                    duration_s=duration_s,
                    command=command,
                    note=f"timeout after {timeout_s}s",
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            )

    _write_outputs(rows=rows, out_dir=resolved_out_dir, config_dir=resolved_config_dir)
    return rows


def _write_outputs(*, rows: list[ConformanceMatrixRow], out_dir: Path, config_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in rows], columns=[field.name for field in fields(ConformanceMatrixRow)])
    if not frame.empty:
        frame = frame.sort_values(["adapter", "config_file"], kind="stable").reset_index(drop=True)

    csv_path = out_dir / "conformance_matrix.csv"
    frame.to_csv(csv_path, index=False)
    provenance = build_conformance_provenance(config_dir=config_dir, matrix_path=csv_path)
    provenance_path = conformance_provenance_path(csv_path)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config_dir": str(config_dir),
        "rows": int(len(frame)),
        "status_counts": frame["status"].value_counts(dropna=False).to_dict() if not frame.empty else {},
        "provenance_path": str(provenance_path.resolve()),
    }
    json_path = out_dir / "conformance_matrix.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "provenance": provenance,
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


def _write_adapter_artifacts(
    *,
    artifacts_dir: Path,
    adapter: str,
    config_file: Path,
    stdout: str,
    stderr: str,
) -> tuple[str, str]:
    stem = f"{_slug(config_file.stem)}__{_slug(adapter)}"
    stdout_path = artifacts_dir / f"{stem}.stdout.txt"
    stderr_path = artifacts_dir / f"{stem}.stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return str(stdout_path.resolve()), str(stderr_path.resolve())


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def parse_args(argv: list[str] | None = None) -> ArgumentParser:
    parser = ArgumentParser(description="Run MaxionBench adapter conformance matrix")
    parser.add_argument("--config-dir", default="configs/conformance")
    parser.add_argument("--out-dir", default="artifacts/conformance")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--adapters", default="", help="Optional comma-separated adapter allowlist")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = parse_args(argv)
    args = parser.parse_args(argv)
    try:
        _run_conformance_matrix(
            config_dir=Path(args.config_dir),
            out_dir=Path(args.out_dir),
            timeout_s=float(args.timeout_s),
            adapters=[item for item in str(args.adapters).split(",") if item.strip()],
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"conformance-matrix failed: {exc}", file=sys.stderr)
        return 2
    return 0


def _normalize_adapters(adapters: Sequence[str] | None) -> set[str] | None:
    if adapters is None:
        return None
    normalized = {str(value).strip() for value in adapters if str(value).strip()}
    return normalized or None


if __name__ == "__main__":
    raise SystemExit(main())
