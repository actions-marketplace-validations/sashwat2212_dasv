"""Adversarial security tests for DockerSandboxProvider and ContainerJanitor (SDD-01)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from dasv.core.errors import SandboxSecurityError
from dasv.sandbox.docker_provider import (
    DockerSandboxProvider,
    build_docker_run_args,
    validate_bind_mount,
    validate_env_vars,
)
from dasv.sandbox.janitor import ContainerJanitor
from dasv.sandbox.provider import ContainerConfig
from tests.conftest import make_fake_process


@pytest.mark.parametrize(
    "forbidden_host",
    [
        Path("/"),
        Path("/etc"),
        Path("/etc/passwd"),
        Path("/etc/ssl/certs"),
        Path.home(),
        Path("~"),
        Path.home() / ".ssh",
        Path.home() / ".ssh" / "id_rsa",
        Path("/var/run/docker.sock"),
        Path("/run/docker.sock"),
        Path("/var/run"),
    ],
)
def test_forbidden_direct_and_nested_bind_mounts_rejected(forbidden_host: Path) -> None:
    with pytest.raises(SandboxSecurityError):
        validate_bind_mount(forbidden_host, "/mnt/data", "ro")

    cfg = ContainerConfig(
        image="python:3.12-slim",
        bind_mounts=[(forbidden_host, "/mnt/data", "ro")],
    )
    with pytest.raises(SandboxSecurityError):
        build_docker_run_args(cfg, run_id="run-sec-1")


def test_forbidden_symlinked_bind_mounts_rejected(tmp_path: Path) -> None:
    etc_link = tmp_path / "etc_link"
    etc_link.symlink_to("/etc")

    home_link = tmp_path / "home_link"
    home_link.symlink_to(Path.home())

    ssh_link = tmp_path / "ssh_link"
    ssh_link.symlink_to(Path.home() / ".ssh")

    root_link = tmp_path / "root_link"
    root_link.symlink_to("/")

    sock_link = tmp_path / "sock_link"
    sock_link.symlink_to("/var/run/docker.sock")

    custom_sock = tmp_path / "docker.sock"
    custom_sock.write_text("", encoding="utf-8")

    for bad_path in (
        etc_link,
        etc_link / "passwd",
        home_link,
        ssh_link,
        ssh_link / "authorized_keys",
        root_link,
        sock_link,
        custom_sock,
    ):
        with pytest.raises(SandboxSecurityError):
            validate_bind_mount(bad_path, "/mnt/escape", "ro")


def test_bind_mount_container_path_injection_and_safe_dir(tmp_path: Path) -> None:
    safe_dir = tmp_path / "verifier_ro"
    safe_dir.mkdir()

    # Relative or colon-injected container path must be rejected
    with pytest.raises(SandboxSecurityError):
        validate_bind_mount(safe_dir, "relative/path", "ro")
    with pytest.raises(SandboxSecurityError):
        validate_bind_mount(safe_dir, "/dasv_verifier:rw", "ro")

    # Safe temp directory mount succeeds and resolves cleanly
    resolved = validate_bind_mount(safe_dir, "/dasv_verifier", "ro")
    assert resolved == safe_dir.resolve()


@pytest.mark.parametrize(
    "bad_env",
    [
        {"OPENAI_API_KEY": "anything"},
        {"openai_api_key": "lowercase-key"},
        {"ANTHROPIC_API_KEY": "anything"},
        {"HF_TOKEN": "anything"},
        {"AWS_SECRET_ACCESS_KEY": "anything"},
        {"GITHUB_TOKEN": "anything"},
        {"CUSTOM_VAR": "sk-proj-secret1234567890abcd"},
        {"CUSTOM_GH": "ghp_1234567890abcdef"},
        {"CUSTOM_HF": "hf_1234567890abcdef"},
        {"CUSTOM_AWS": "AKIA1234567890ABCDEF"},
    ],
)
def test_secret_env_vars_rejected(bad_env: dict[str, str]) -> None:
    with pytest.raises(SandboxSecurityError):
        validate_env_vars(bad_env)

    cfg = ContainerConfig(image="python:3.12-slim", env_vars=bad_env)
    with pytest.raises(SandboxSecurityError):
        build_docker_run_args(cfg, run_id="run-env-1")


def test_safe_env_vars_and_resource_guardrails_enforced() -> None:
    cfg = ContainerConfig(
        image="python:3.12-slim",
        cpus=1.0,
        memory="512m",
        pids_limit=64,
        network_mode="none",
        env_vars={"PYTHONUNBUFFERED": "1", "TERM": "dumb"},
        labels={
            "dasv.managed": "false",
            "dasv.run_id": "spoofed-run",
            "dasv.plane": "verification",
        },
        plane="execution",
    )
    args = build_docker_run_args(cfg, run_id="real-run-42")

    assert "--network=none" in args
    assert "--cap-drop=ALL" in args
    assert "--security-opt=no-new-privileges:true" in args
    assert "--memory=512m" in args
    assert "--memory-swap=512m" in args
    assert "--pids-limit=64" in args
    assert "dasv.managed=true" in args
    assert "dasv.run_id=real-run-42" in args
    assert "dasv.plane=execution" in args
    assert "dasv.run_id=spoofed-run" not in args

    # Empty run_id is forbidden so containers can never escape janitor labeling
    with pytest.raises(SandboxSecurityError):
        build_docker_run_args(cfg, run_id="   ")


async def test_janitor_label_isolation_and_stubborn_orphan_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ps_filters: list[str] = []
    rm_targets: list[list[str]] = []

    async def fake_exec(*args: str, **kwargs: Any) -> Any:
        _ = kwargs
        if args[1] == "ps":
            filter_val = args[4]
            ps_filters.append(filter_val)
            if filter_val == "label=dasv.run_id=run-a":
                # Simulate ctr-stubborn surviving rm -f
                if ps_filters.count(filter_val) == 1:
                    return make_fake_process(returncode=0, stdout=b"ctr-a1\nctr-stubborn\n")
                return make_fake_process(returncode=0, stdout=b"ctr-stubborn\n")
            return make_fake_process(returncode=0, stdout=b"")
        if args[1] == "rm":
            rm_targets.append(list(args[3:]))
            return make_fake_process(returncode=0, stdout=b"")
        return make_fake_process(returncode=1, stderr=b"unexpected")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    janitor = ContainerJanitor()
    janitor.register("ctr-a1", "run-a")
    janitor.register("ctr-b1", "run-b")

    report_a = await janitor.reap_all("run-a")
    assert report_a.removed_ids == ["ctr-a1"]
    assert report_a.stubborn_ids == ["ctr-stubborn"]
    # Isolation: run-b was never queried or removed
    assert all("run-b" not in f for f in ps_filters)
    assert all("ctr-b1" not in batch for batch in rm_targets)
    assert janitor.tracked_ids("run-b") == {"ctr-b1"}


async def test_timeout_triggers_docker_kill_and_returns_124(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed_cids: list[str] = []

    async def fake_exec(*args: str, **kwargs: Any) -> Any:
        _ = kwargs
        subcmd = args[1]
        if subcmd == "run":
            return make_fake_process(returncode=0, stdout=b"ctr-hang-99\n")
        if subcmd == "inspect":
            return make_fake_process(returncode=0, stdout=b"sha256:abc\n")
        if subcmd == "image":
            return make_fake_process(returncode=0, stdout=b"arm64|sha256:abc\n")
        if subcmd == "exec":
            return make_fake_process(returncode=0, stdout=b"", delay_sec=1.0)
        if subcmd == "kill":
            killed_cids.append(args[2])
            return make_fake_process(returncode=0, stdout=b"")
        if subcmd == "rm":
            return make_fake_process(returncode=0, stdout=b"")
        return make_fake_process(returncode=0, stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    janitor = ContainerJanitor()
    provider = DockerSandboxProvider(janitor=janitor)
    handle = await provider.start(ContainerConfig(image="alpine:3.20"), run_id="run-timeout")

    res = await provider.exec(handle, ["sh", "-c", "while true; do :; done"], timeout_sec=0.05)
    assert res.timed_out is True
    assert res.exit_code == 124
    assert killed_cids == ["ctr-hang-99"]

    await provider.stop_and_remove(handle)
    assert janitor.tracked_ids("run-timeout") == set()
