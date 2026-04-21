"""Verify conformance config catalog shape and adapter coverage."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

from maxionbench.tools.verify_engine_readiness import REQUIRED_ADAPTERS

SUPPORTED_METRICS = {"ip", "l2", "cosine"}


def verify_conformance_config_dir(
    *,
    config_dir: Path,
    allow_gpu_unavailable: bool = False,
) -> dict[str, Any]:
    resolved = config_dir.resolve()
    errors: list[dict[str, Any]] = []
    required_adapters = sorted(set(REQUIRED_ADAPTERS) | {"mock"})
    adapters_seen: set[str] = set()

    if not resolved.exists():
        errors.append(
            {
                "source": "config_dir",
                "field": "path",
                "message": f"conformance config directory does not exist: {resolved}",
            }
        )
        return _summary(
            config_dir=resolved,
            files_checked=0,
            adapters_seen=adapters_seen,
            required_adapters=required_adapters,
            allow_gpu_unavailable=allow_gpu_unavailable,
            errors=errors,
        )
    if not resolved.is_dir():
        errors.append(
            {
                "source": "config_dir",
                "field": "path",
                "message": f"conformance config path is not a directory: {resolved}",
            }
        )
        return _summary(
            config_dir=resolved,
            files_checked=0,
            adapters_seen=adapters_seen,
            required_adapters=required_adapters,
            allow_gpu_unavailable=allow_gpu_unavailable,
            errors=errors,
        )

    files = sorted(resolved.glob("*.json"))
    if not files:
        errors.append(
            {
                "source": "config_dir",
                "field": "files",
                "message": f"no conformance config files (*.json) found in: {resolved}",
            }
        )
        return _summary(
            config_dir=resolved,
            files_checked=0,
            adapters_seen=adapters_seen,
            required_adapters=required_adapters,
            allow_gpu_unavailable=allow_gpu_unavailable,
            errors=errors,
        )

    for path in files:
        payload = _load_mapping(path=path, errors=errors)
        if payload is None:
            continue

        adapter = str(payload.get("adapter", "")).strip()
        if not adapter:
            errors.append(
                {
                    "source": "config_file",
                    "file": str(path),
                    "field": "adapter",
                    "message": "adapter must be a non-empty string",
                }
            )
        else:
            adapters_seen.add(adapter)

        adapter_options_raw = payload.get("adapter_options_json")
        if not isinstance(adapter_options_raw, str):
            errors.append(
                {
                    "source": "config_file",
                    "file": str(path),
                    "field": "adapter_options_json",
                    "message": "adapter_options_json must be a JSON object string",
                }
            )
        else:
            try:
                options_payload = json.loads(adapter_options_raw)
            except json.JSONDecodeError:
                errors.append(
                    {
                        "source": "config_file",
                        "file": str(path),
                        "field": "adapter_options_json",
                        "message": "adapter_options_json is not valid JSON",
                    }
                )
            else:
                if not isinstance(options_payload, dict):
                    errors.append(
                        {
                            "source": "config_file",
                            "file": str(path),
                            "field": "adapter_options_json",
                            "message": "adapter_options_json must decode to a JSON object",
                        }
                    )

        collection = payload.get("collection")
        if not isinstance(collection, str) or not collection.strip():
            errors.append(
                {
                    "source": "config_file",
                    "file": str(path),
                    "field": "collection",
                    "message": "collection must be a non-empty string",
                }
            )

        dimension = payload.get("dimension")
        if not isinstance(dimension, int) or dimension <= 0:
            errors.append(
                {
                    "source": "config_file",
                    "file": str(path),
                    "field": "dimension",
                    "message": "dimension must be a positive integer",
                }
            )

        metric = payload.get("metric")
        if not isinstance(metric, str) or metric not in SUPPORTED_METRICS:
            errors.append(
                {
                    "source": "config_file",
                    "file": str(path),
                    "field": "metric",
                    "message": f"metric must be one of {sorted(SUPPORTED_METRICS)}",
                }
            )

    missing_required = sorted(set(required_adapters) - adapters_seen)
    for adapter in missing_required:
        errors.append(
            {
                "source": "required_coverage",
                "field": "adapter",
                "message": f"missing required conformance adapter config for `{adapter}`",
            }
        )

    return _summary(
        config_dir=resolved,
        files_checked=len(files),
        adapters_seen=adapters_seen,
        required_adapters=required_adapters,
        allow_gpu_unavailable=allow_gpu_unavailable,
        errors=errors,
    )


def _summary(
    *,
    config_dir: Path,
    files_checked: int,
    adapters_seen: set[str],
    required_adapters: list[str],
    allow_gpu_unavailable: bool,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    missing_required = sorted(set(required_adapters) - adapters_seen)
    return {
        "config_dir": str(config_dir),
        "allow_gpu_unavailable": bool(allow_gpu_unavailable),
        "files_checked": int(files_checked),
        "adapters_seen": sorted(adapters_seen),
        "required_adapters": list(required_adapters),
        "missing_required_adapters": missing_required,
        "error_count": len(errors),
        "errors": errors,
        "pass": len(errors) == 0,
    }


def _load_mapping(*, path: Path, errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(
            {
                "source": "config_file",
                "file": str(path),
                "field": "json",
                "message": f"file is not valid JSON: {exc.msg}",
            }
        )
        return None
    if not isinstance(payload, dict):
        errors.append(
            {
                "source": "config_file",
                "file": str(path),
                "field": "json",
                "message": "file root must be a JSON object",
            }
        )
        return None
    return dict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify conformance config directory structure and coverage")
    parser.add_argument("--config-dir", default="configs/conformance")
    parser.add_argument("--allow-gpu-unavailable", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = verify_conformance_config_dir(
        config_dir=Path(args.config_dir),
        allow_gpu_unavailable=bool(args.allow_gpu_unavailable),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        if summary["pass"]:
            print(f"conformance config verification passed: {summary['files_checked']} files checked")
        else:
            print(f"conformance config verification failed: {summary['error_count']} issue(s)")
            for item in summary["errors"]:
                print(f"- {item['message']}")
    return 0 if bool(summary["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
