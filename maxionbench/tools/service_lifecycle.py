"""Manage benchmark service containers via docker compose."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

# Services that run as Docker containers in the portable profile
_PORTABLE_SERVICES: list[str] = ["qdrant", "pgvector"]
# Full reference profile adds the remaining engines
_REFERENCE_SERVICES: list[str] = ["qdrant", "pgvector", "opensearch", "weaviate", "milvus"]
_ALL_SERVICES: list[str] = ["qdrant", "pgvector", "opensearch", "weaviate", "milvus"]

_PROFILE_MAP: dict[str, list[str]] = {
    "portable": _PORTABLE_SERVICES,
    "reference": _REFERENCE_SERVICES,
    "all": _ALL_SERVICES,
}

# Host-side ports exposed by docker-compose.yml for each service
_SERVICE_ADAPTER_MAP: dict[str, tuple[str, dict[str, Any]]] = {
    "qdrant": ("qdrant", {"host": "localhost", "port": 6333}),
    "pgvector": ("pgvector", {"dsn": "postgresql://postgres:postgres@localhost:5432/postgres"}),
    "opensearch": ("opensearch", {"host": "localhost", "port": 9200}),
    "weaviate": ("weaviate", {"host": "localhost", "port": 8080}),
    "milvus": ("milvus", {"host": "localhost", "port": 19530}),
}

_LANCEDB_ENV_NOTE = (
    "Set MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI before running benchmarks:\n"
    "  export MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI=\"$PWD/artifacts/lancedb/service\""
)


def _find_compose_file(compose_file: str | None) -> Path:
    if compose_file:
        path = Path(compose_file).resolve()
        if not path.exists():
            raise FileNotFoundError(f"compose file not found: {path}")
        return path
    for directory in [Path.cwd()] + list(Path.cwd().parents):
        for name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
            candidate = directory / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "docker-compose.yml not found in the current directory or any parent. "
        "Run from the repository root or pass --compose-file."
    )


def _resolve_services(*, profile: str | None, services_csv: str | None) -> list[str]:
    if services_csv:
        return [s.strip() for s in services_csv.split(",") if s.strip()]
    return _PROFILE_MAP.get(profile or "portable", _PORTABLE_SERVICES)


def _compose_cmd(compose_file: Path) -> list[str]:
    if not shutil.which("docker"):
        raise RuntimeError(
            "docker not found on PATH. Install Docker Desktop and ensure 'docker compose' is available."
        )
    return ["docker", "compose", "--file", str(compose_file)]


def services_up(
    *,
    compose_file: Path,
    services: list[str],
    wait: bool,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    cmd = _compose_cmd(compose_file) + ["up", "--detach", "--quiet-pull"] + services
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker compose up failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    summary: dict[str, Any] = {
        "action": "up",
        "services": services,
        "compose_file": str(compose_file),
        "healthy": None,
        "note": _LANCEDB_ENV_NOTE,
    }
    if wait:
        _poll_service_health(
            services=services,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            summary=summary,
        )
    return summary


def services_down(*, compose_file: Path, volumes: bool) -> dict[str, Any]:
    cmd = _compose_cmd(compose_file) + ["down"]
    if volumes:
        cmd.append("--volumes")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker compose down failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    return {
        "action": "down",
        "compose_file": str(compose_file),
        "volumes_removed": volumes,
        "returncode": result.returncode,
    }


def services_status(*, compose_file: Path) -> dict[str, Any]:
    cmd = _compose_cmd(compose_file) + ["ps", "--format", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker compose ps failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    entries: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"raw": line})
    return {
        "action": "status",
        "compose_file": str(compose_file),
        "services": entries,
    }


def services_wait(
    *,
    services: list[str],
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"action": "wait", "services": services}
    _poll_service_health(
        services=services,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        summary=summary,
    )
    return summary


def _poll_service_health(
    *,
    services: list[str],
    timeout_s: float,
    poll_interval_s: float,
    summary: dict[str, Any],
) -> None:
    from maxionbench.tools.wait_adapter import wait_for_adapter

    results: list[dict[str, Any]] = []
    for svc in services:
        mapping = _SERVICE_ADAPTER_MAP.get(svc)
        if mapping is None:
            results.append({"service": svc, "skipped": True, "reason": "no adapter mapping"})
            continue
        adapter_name, adapter_options = mapping
        try:
            wait_for_adapter(
                adapter_name=adapter_name,
                adapter_options=adapter_options,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )
            results.append({"service": svc, "adapter": adapter_name, "ready": True})
        except Exception as exc:
            results.append({"service": svc, "adapter": adapter_name, "ready": False, "error": str(exc)})

    summary["wait_results"] = results
    summary["healthy"] = all(r.get("ready") or r.get("skipped") for r in results)


def _print_summary(summary: dict[str, Any]) -> None:
    action = summary.get("action", "")
    if action == "up":
        svcs = ", ".join(summary.get("services", []))
        print(f"Started: {svcs}")
        healthy = summary.get("healthy")
        if healthy is True:
            print("All services healthy.")
        elif healthy is False:
            print("WARNING: one or more services did not become healthy.")
        else:
            print("Health not checked. Run `maxionbench services wait` to verify.")
        note = summary.get("note")
        if note:
            print(f"\n{note}")
    elif action == "down":
        print("Services stopped.")
        if summary.get("volumes_removed"):
            print("Volumes removed.")
    elif action == "status":
        entries = summary.get("services", [])
        if not entries:
            print("No services running.")
        for e in entries:
            name = e.get("Name") or e.get("Service") or e.get("raw", "?")
            state = e.get("State") or e.get("Status", "?")
            print(f"  {name}: {state}")
    elif action == "wait":
        for r in summary.get("wait_results", []):
            svc = r.get("service", "?")
            if r.get("skipped"):
                print(f"  {svc}: skipped ({r.get('reason')})")
            elif r.get("ready"):
                print(f"  {svc}: ready")
            else:
                print(f"  {svc}: NOT ready — {r.get('error', 'timeout')}")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Manage benchmark service containers")
    parser.add_argument("action", choices=["up", "down", "status", "wait"])
    parser.add_argument(
        "--profile",
        choices=["portable", "reference", "all"],
        default="portable",
        help="Service group to target (default: portable = qdrant, pgvector)",
    )
    parser.add_argument(
        "--services",
        default=None,
        help="Comma-separated explicit service list; overrides --profile",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll service health after 'up' until all services are ready",
    )
    parser.add_argument("--volumes", action="store_true", help="Remove volumes on 'down'")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--compose-file", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        compose_file = _find_compose_file(args.compose_file)
        services = _resolve_services(profile=args.profile, services_csv=args.services)

        if args.action == "up":
            summary = services_up(
                compose_file=compose_file,
                services=services,
                wait=args.wait,
                timeout_s=args.timeout_s,
                poll_interval_s=args.poll_interval_s,
            )
        elif args.action == "down":
            summary = services_down(compose_file=compose_file, volumes=args.volumes)
        elif args.action == "status":
            summary = services_status(compose_file=compose_file)
        elif args.action == "wait":
            summary = services_wait(
                services=services,
                timeout_s=args.timeout_s,
                poll_interval_s=args.poll_interval_s,
            )
        else:
            raise ValueError(f"unknown action: {args.action}")

    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
