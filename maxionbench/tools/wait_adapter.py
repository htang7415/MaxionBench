"""Poll adapter health until a config or adapter endpoint becomes ready."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from maxionbench.adapters import create_adapter
from maxionbench.orchestration.config_schema import expand_env_placeholders, load_run_config
from maxionbench.runtime.healthcheck import wait_for_healthy, wait_for_port_ready


def wait_for_adapter(
    *,
    adapter_name: str,
    adapter_options: dict[str, Any],
    timeout_s: float = 120.0,
    poll_interval_s: float = 1.0,
) -> dict[str, Any]:
    tcp_probe = _service_tcp_probe(adapter_name=adapter_name, adapter_options=adapter_options)
    if tcp_probe is not None:
        wait_for_port_ready(
            tcp_probe["host"],
            tcp_probe["port"],
            timeout_s=float(timeout_s),
            poll_interval_s=float(poll_interval_s),
        )
    adapter = create_adapter(adapter_name, **adapter_options)
    wait_for_healthy(
        adapter.healthcheck,
        timeout_s=float(timeout_s),
        poll_interval_s=float(poll_interval_s),
    )
    return {
        "adapter": adapter_name,
        "adapter_options": dict(adapter_options),
        "timeout_s": float(timeout_s),
        "poll_interval_s": float(poll_interval_s),
        "tcp_probe": tcp_probe,
        "ready": True,
    }


def _service_tcp_probe(*, adapter_name: str, adapter_options: dict[str, Any]) -> dict[str, Any] | None:
    normalized_name = str(adapter_name).strip().lower()
    if normalized_name == "qdrant":
        host = str(adapter_options.get("host", "")).strip()
        port = adapter_options.get("port")
        if host and port is not None:
            return {"host": host, "port": int(port)}
        return None
    if normalized_name == "pgvector":
        dsn = str(adapter_options.get("dsn", "")).strip()
        if not dsn:
            return None
        parsed = urlparse(dsn)
        if not parsed.hostname or parsed.port is None:
            return None
        return {"host": parsed.hostname, "port": int(parsed.port)}
    if normalized_name == "lancedb-service":
        base_url = str(adapter_options.get("base_url", "")).strip()
        if not base_url or adapter_options.get("inproc_uri"):
            return None
        parsed = urlparse(base_url)
        if not parsed.hostname or parsed.port is None:
            return None
        return {"host": parsed.hostname, "port": int(parsed.port)}
    return None


def parse_args(argv: list[str] | None = None) -> ArgumentParser:
    parser = ArgumentParser(description="Poll adapter health until it becomes ready")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", default=None, help="Run config whose engine + adapter_options should be polled")
    source.add_argument("--adapter", default=None, help="Adapter name when not using --config")
    parser.add_argument("--adapter-options-json", default="{}", help="JSON object with adapter options")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = parse_args(argv)
    args = parser.parse_args(argv)

    adapter_name = args.adapter
    adapter_options: dict[str, Any]
    if args.config:
        cfg = load_run_config(Path(args.config).resolve())
        adapter_name = cfg.engine
        adapter_options = dict(cfg.adapter_options)
    else:
        try:
            parsed = json.loads(args.adapter_options_json)
        except json.JSONDecodeError as exc:
            raise ValueError("--adapter-options-json must be valid JSON object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("--adapter-options-json must decode to a JSON object")
        adapter_options = dict(expand_env_placeholders(parsed))

    if not isinstance(adapter_name, str) or not adapter_name.strip():
        raise ValueError("adapter name must be non-empty")

    summary = wait_for_adapter(
        adapter_name=adapter_name.strip(),
        adapter_options=adapter_options,
        timeout_s=float(args.timeout_s),
        poll_interval_s=float(args.poll_interval_s),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
