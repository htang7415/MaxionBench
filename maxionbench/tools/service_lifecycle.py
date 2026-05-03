"""Manage benchmark service containers via docker compose."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

# Services that run as Docker containers in the MaxionBench profile
_MAXIONBENCH_SERVICES: list[str] = ["qdrant", "pgvector"]
_PROFILE_MAP: dict[str, list[str]] = {
    "maxionbench": _MAXIONBENCH_SERVICES,
    "portable": _MAXIONBENCH_SERVICES,
}

# Host-side ports exposed by docker-compose.yml for each service
_SERVICE_ADAPTER_MAP: dict[str, tuple[str, dict[str, Any]]] = {
    "qdrant": ("qdrant", {"host": "localhost", "port": 6333}),
    "pgvector": ("pgvector", {"dsn": "postgresql://postgres:postgres@localhost:5432/postgres"}),
}

_LANCEDB_ENV_NOTE = (
    "Set MAXIONBENCH_LANCEDB_INPROC_URI to a local filesystem path before running LanceDB in-process benchmarks:\n"
    "  export MAXIONBENCH_LANCEDB_INPROC_URI=\"/tmp/maxionbench/lancedb/inproc\""
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
    return _PROFILE_MAP.get(profile or "maxionbench", _MAXIONBENCH_SERVICES)


def _normalize_argv(argv: list[str] | None) -> list[str]:
    normalized = list(sys.argv[1:] if argv is None else argv)
    for idx, token in enumerate(normalized[:-1]):
        if token == "--profile" and normalized[idx + 1] == "portable":
            normalized[idx + 1] = "maxionbench"
    return normalized


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
    skip_arch_check: bool = False,
) -> dict[str, Any]:
    image_platforms: list[dict[str, Any]] = []
    if not skip_arch_check:
        image_platforms = verify_service_image_platforms(compose_file=compose_file, services=services)
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
        "image_platforms": image_platforms,
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


def verify_service_image_platforms(*, compose_file: Path, services: list[str]) -> list[dict[str, Any]]:
    images = _compose_service_images(compose_file=compose_file, services=services)
    target_arch = _docker_arch(platform.machine())
    target = f"linux/{target_arch}"
    results: list[dict[str, Any]] = []
    for service, image in images.items():
        platforms = _inspect_image_platforms(image)
        supported = target in platforms
        result = {
            "service": service,
            "image": image,
            "target_platform": target,
            "platforms": sorted(platforms),
            "supported": supported,
        }
        results.append(result)
        if not supported:
            raise RuntimeError(
                f"image {image!r} for service {service!r} does not advertise {target}; "
                "set the corresponding MAXIONBENCH_*_IMAGE override to a native image or pass --skip-arch-check."
            )
    return results


def _compose_service_images(*, compose_file: Path, services: list[str]) -> dict[str, str]:
    result = subprocess.run(
        _compose_cmd(compose_file) + ["config", "--format", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker compose config failed (exit {result.returncode}):\n{result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"docker compose config did not emit valid JSON: {exc.msg}") from exc
    service_payload = payload.get("services", {})
    if not isinstance(service_payload, dict):
        raise RuntimeError("docker compose config JSON missing services mapping")
    images: dict[str, str] = {}
    for service in services:
        raw = service_payload.get(service, {})
        if not isinstance(raw, dict):
            continue
        image = str(raw.get("image") or "").strip()
        if image:
            images[service] = image
    return images


def _inspect_image_platforms(image: str) -> set[str]:
    result = subprocess.run(["docker", "manifest", "inspect", image], capture_output=True, text=True)
    if result.returncode != 0:
        local_platforms = _inspect_local_image_platforms(image)
        if local_platforms:
            return local_platforms
        raise RuntimeError(f"docker manifest inspect failed for {image!r} (exit {result.returncode}):\n{result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"docker manifest inspect did not emit valid JSON for {image!r}: {exc.msg}") from exc
    platforms: set[str] = set()
    manifests = payload.get("manifests")
    if isinstance(manifests, list):
        for item in manifests:
            if not isinstance(item, dict):
                continue
            platform_payload = item.get("platform")
            if not isinstance(platform_payload, dict):
                continue
            os_name = str(platform_payload.get("os") or "").strip()
            arch = str(platform_payload.get("architecture") or "").strip()
            if os_name and arch:
                platforms.add(f"{os_name}/{_docker_arch(arch)}")
    else:
        os_name = str(payload.get("os") or "").strip()
        arch = str(payload.get("architecture") or "").strip()
        if os_name and arch:
            platforms.add(f"{os_name}/{_docker_arch(arch)}")
    return platforms


def _inspect_local_image_platforms(image: str) -> set[str]:
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    if result.returncode != 0:
        return set()
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return set()
    if not isinstance(payload, list):
        return set()
    platforms: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        os_name = str(item.get("Os") or "").strip()
        arch = str(item.get("Architecture") or "").strip()
        if os_name and arch:
            platforms.add(f"{os_name}/{_docker_arch(arch)}")
    return platforms


def _docker_arch(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"aarch64", "arm64"}:
        return "arm64"
    if normalized in {"x86_64", "amd64"}:
        return "amd64"
    return normalized


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
    argv = _normalize_argv(argv)
    parser = ArgumentParser(description="Manage benchmark service containers")
    parser.add_argument("action", choices=["up", "down", "status", "wait"])
    parser.add_argument(
        "--profile",
        choices=["maxionbench"],
        default="maxionbench",
        help="Service group to target (maxionbench = qdrant, pgvector)",
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
    parser.add_argument(
        "--skip-arch-check",
        action="store_true",
        help="Skip Docker manifest architecture validation before 'up'",
    )
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
                skip_arch_check=bool(args.skip_arch_check),
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
