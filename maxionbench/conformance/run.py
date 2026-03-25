"""CLI runner for adapter conformance tests."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from typing import Any

import pytest

from maxionbench.orchestration.config_schema import expand_env_placeholders


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Run MaxionBench adapter conformance tests")
    parser.add_argument("--adapter", default="mock", help="Adapter name (mock, qdrant, ...)")
    parser.add_argument(
        "--adapter-options-json",
        default="{}",
        help="JSON object with adapter options",
    )
    parser.add_argument("--collection", default="conformance")
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--metric", default="ip")
    args = parser.parse_args(argv)

    try:
        options: dict[str, Any] = json.loads(args.adapter_options_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--adapter-options-json must be valid JSON object") from exc
    if not isinstance(options, dict):
        raise ValueError("--adapter-options-json must decode to a JSON object")
    options = dict(expand_env_placeholders(options))

    env = os.environ.copy()
    env["MAXIONBENCH_CONFORMANCE_ADAPTER"] = args.adapter
    env["MAXIONBENCH_CONFORMANCE_ADAPTER_OPTIONS_JSON"] = json.dumps(options)
    env["MAXIONBENCH_CONFORMANCE_COLLECTION"] = args.collection
    env["MAXIONBENCH_CONFORMANCE_DIMENSION"] = str(args.dimension)
    env["MAXIONBENCH_CONFORMANCE_METRIC"] = args.metric

    old_env = os.environ.copy()
    try:
        os.environ.update(env)
        return pytest.main(
            [
                "-q",
                "-s",
                str(Path("maxionbench/conformance/test_conformance.py")),
            ]
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)


if __name__ == "__main__":
    raise SystemExit(main())
