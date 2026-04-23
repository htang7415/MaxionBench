from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from maxionbench.tools import service_lifecycle as service_mod


def test_docker_arch_normalizes_apple_silicon_names() -> None:
    assert service_mod._docker_arch("arm64") == "arm64"
    assert service_mod._docker_arch("aarch64") == "arm64"
    assert service_mod._docker_arch("x86_64") == "amd64"


def test_verify_service_image_platforms_rejects_missing_host_arch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")

    def _fake_run(cmd: list[str], capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        del capture_output, text
        if cmd[-3:] == ["config", "--format", "json"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"services": {"qdrant": {"image": "qdrant/qdrant:test"}}}),
                stderr="",
            )
        if cmd[:3] == ["docker", "manifest", "inspect"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"manifests": [{"platform": {"os": "linux", "architecture": "amd64"}}]}),
                stderr="",
            )
        raise AssertionError(cmd)

    monkeypatch.setattr(service_mod.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(service_mod.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(service_mod.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="does not advertise linux/arm64"):
        service_mod.verify_service_image_platforms(compose_file=compose, services=["qdrant"])


def test_verify_service_image_platforms_accepts_host_arch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")

    def _fake_run(cmd: list[str], capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        del capture_output, text
        if cmd[-3:] == ["config", "--format", "json"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"services": {"pgvector": {"image": "pgvector/pgvector:test"}}}),
                stderr="",
            )
        if cmd[:3] == ["docker", "manifest", "inspect"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "manifests": [
                            {"platform": {"os": "linux", "architecture": "amd64"}},
                            {"platform": {"os": "linux", "architecture": "arm64"}},
                        ]
                    }
                ),
                stderr="",
            )
        raise AssertionError(cmd)

    monkeypatch.setattr(service_mod.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(service_mod.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(service_mod.subprocess, "run", _fake_run)

    summary = service_mod.verify_service_image_platforms(compose_file=compose, services=["pgvector"])
    assert summary == [
        {
            "service": "pgvector",
            "image": "pgvector/pgvector:test",
            "target_platform": "linux/arm64",
            "platforms": ["linux/amd64", "linux/arm64"],
            "supported": True,
        }
    ]


def test_verify_service_image_platforms_falls_back_to_local_image_inspect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")

    def _fake_run(cmd: list[str], capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        del capture_output, text
        if cmd[-3:] == ["config", "--format", "json"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"services": {"pgvector": {"image": "pgvector/pgvector:test"}}}),
                stderr="",
            )
        if cmd[:3] == ["docker", "manifest", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="dns failed")
        if cmd[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"Os": "linux", "Architecture": "arm64"}]),
                stderr="",
            )
        raise AssertionError(cmd)

    monkeypatch.setattr(service_mod.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(service_mod.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(service_mod.subprocess, "run", _fake_run)

    summary = service_mod.verify_service_image_platforms(compose_file=compose, services=["pgvector"])
    assert summary == [
        {
            "service": "pgvector",
            "image": "pgvector/pgvector:test",
            "target_platform": "linux/arm64",
            "platforms": ["linux/arm64"],
            "supported": True,
        }
    ]
