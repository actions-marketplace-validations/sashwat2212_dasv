"""Unit & live-Docker tests for SandboxProvider, Docker provider, Janitor, and runtime probes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from dasv.core.errors import RuntimeSandboxError
from dasv.core.schemas import EnvSpec, EnvStandard, Severity
from dasv.sandbox.docker_provider import DockerSandboxProvider, build_docker_run_args
from dasv.sandbox.janitor import ContainerJanitor
from dasv.sandbox.probes import (
    probe_egress_and_env_scrub,
    probe_emulated_arch,
    probe_reset_residue,
    probe_writable_verifier_paths,
    run_runtime_boundary_probes,
)
from dasv.sandbox.provider import (
    ContainerConfig,
    ExecResult,
    SandboxHandle,
    SandboxProvider,
    current_host_arch,
    normalize_arch,
    truncate_output,
)
from tests.conftest import FakeSandboxProvider, make_fake_process


def test_provider_protocol_and_helpers() -> None:
    assert isinstance(DockerSandboxProvider(), SandboxProvider)
    assert isinstance(FakeSandboxProvider(), SandboxProvider)

    assert normalize_arch("x86_64") == "amd64"
    assert normalize_arch("linux/arm64/v8") == "arm64"
    assert normalize_arch("aarch64") == "arm64"
    assert normalize_arch(None) == "unknown"

    short_text, short_trunc = truncate_output(b"hello world", max_bytes=64)
    assert short_text == "hello world"
    assert short_trunc is False

    long_raw = b"A" * 200
    capped_text, capped_trunc = truncate_output(long_raw, max_bytes=64)
    assert capped_trunc is True
    assert capped_text.endswith("...[truncated by dasv]")
    assert len(capped_text.encode("utf-8")) <= 64


def test_build_docker_run_args_enforces_hardened_flags(tmp_path: Path) -> None:
    safe_dir = tmp_path / "tests_ro"
    safe_dir.mkdir()
    cfg = ContainerConfig(
        image="python:3.12-slim@sha256:" + ("a" * 64),
        cpus=1.5,
        memory="1g",
        pids_limit=128,
        read_only_root=True,
        bind_mounts=[(safe_dir, "/dasv_verifier", "ro")],
        labels={"dasv.managed": "false", "custom.tag": "v1"},
        plane="execution",
    )
    args = build_docker_run_args(cfg, run_id="run-123")

    assert args[:3] == ["run", "--detach", "--init"]
    assert "--network=none" in args
    assert "--cap-drop=ALL" in args
    assert "--security-opt=no-new-privileges:true" in args
    assert "--cpus=1.5" in args
    assert "--memory=1g" in args
    assert "--memory-swap=1g" in args
    assert "--pids-limit=128" in args
    assert "--read-only" in args
    assert "--volume" in args
    # Mandatory labels cannot be overridden by caller labels
    assert "dasv.managed=true" in args
    assert "dasv.run_id=run-123" in args
    assert "dasv.plane=execution" in args
    assert "custom.tag=v1" in args

    # Verification plane MUST force --network=none even if bridge was requested
    verify_cfg = ContainerConfig(
        image="python:3.12-slim",
        network_mode="bridge",
        plane="verification",
    )
    verify_args = build_docker_run_args(verify_cfg, run_id="run-verify")
    assert "--network=none" in verify_args
    assert "dasv.plane=verification" in verify_args


async def test_docker_provider_subprocess_lifecycle_truncation_and_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    invocations: list[tuple[str, ...]] = []
    other_arch = "amd64" if current_host_arch() == "arm64" else "arm64"

    async def fake_exec(*args: str, **kwargs: Any) -> Any:
        _ = kwargs
        invocations.append(tuple(args))
        subcmd = args[1]
        if subcmd == "info":
            return make_fake_process(returncode=0, stdout=f"27.3.1|{other_arch}\n".encode())
        if subcmd == "run":
            return make_fake_process(returncode=0, stdout=b"deadbeef1234\n")
        if subcmd == "inspect":
            return make_fake_process(returncode=0, stdout=b"sha256:imgdigest999\n")
        if subcmd == "image":
            return make_fake_process(
                returncode=0, stdout=f"{other_arch}|sha256:imgdigest999\n".encode()
            )
        if subcmd == "exec":
            if "sleep" in args:
                return make_fake_process(returncode=0, stdout=b"", delay_sec=0.5)
            return make_fake_process(returncode=0, stdout=b"X" * 300, stderr=b"err")
        if subcmd == "stats":
            return make_fake_process(returncode=0, stdout=b"64.5MiB / 2GiB\n")
        if subcmd == "diff":
            return make_fake_process(returncode=0, stdout=b"C /app\nA /tmp/.dasv_sentinel\n")
        if subcmd in {"kill", "cp", "rm"}:
            return make_fake_process(returncode=0, stdout=b"")
        return make_fake_process(returncode=1, stderr=b"unexpected")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    janitor = ContainerJanitor()
    provider = DockerSandboxProvider(max_output_bytes=80, janitor=janitor)

    health = await provider.health_check()
    assert health.healthy is True
    assert health.server_version == "27.3.1"
    assert health.server_arch == other_arch
    assert health.emulated_arch_warning is True

    handle = await provider.start(ContainerConfig(image="alpine:3.20"), run_id="run-abc")
    assert handle.container_id == "deadbeef1234"
    assert handle.image_digest == "sha256:imgdigest999"
    assert handle.resolved_arch == other_arch
    assert handle.emulated_arch is True
    assert "deadbeef1234" in janitor.tracked_ids("run-abc")

    exec_res = await provider.exec(handle, ["echo", "hi"], timeout_sec=5.0)
    assert exec_res.exit_code == 0
    assert exec_res.truncated is True
    assert len(exec_res.stdout.encode("utf-8")) <= 80
    assert exec_res.peak_memory_mb == pytest.approx(64.5)

    timeout_res = await provider.exec(handle, ["sleep", "10"], timeout_sec=0.05)
    assert timeout_res.timed_out is True
    assert timeout_res.exit_code == 124
    assert any(cmd[1] == "kill" and cmd[2] == "deadbeef1234" for cmd in invocations)

    diffs = await provider.diff_fs(handle)
    assert diffs == ["A /tmp/.dasv_sentinel", "C /app"]

    await provider.copy_from(handle, "/app/out.txt", tmp_path / "copied" / "out.txt")
    assert any(cmd[1] == "cp" for cmd in invocations)

    await provider.stop_and_remove(handle)
    assert janitor.tracked_ids("run-abc") == set()


async def test_docker_provider_missing_binary_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def missing_exec(*args: str, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_exec)

    provider = DockerSandboxProvider()
    health = await provider.health_check()
    assert health.healthy is False
    assert "not found" in health.detail.lower()

    with pytest.raises(RuntimeSandboxError, match="not found"):
        await provider.start(ContainerConfig(image="alpine:3.20"), run_id="run-missing")


async def test_container_janitor_reaps_tracked_and_docker_labeled_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ps_calls = 0
    rm_batches: list[list[str]] = []

    async def fake_exec(*args: str, **kwargs: Any) -> Any:
        nonlocal ps_calls
        _ = kwargs
        if args[1] == "ps":
            ps_calls += 1
            if ps_calls == 1:
                return make_fake_process(returncode=0, stdout=b"ctr-orphan-2\nctr-tracked-1\n")
            return make_fake_process(returncode=0, stdout=b"")
        if args[1] == "rm":
            rm_batches.append(list(args[3:]))
            return make_fake_process(returncode=0, stdout=b"")
        return make_fake_process(returncode=1, stderr=b"unexpected")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    janitor = ContainerJanitor()
    janitor.register("ctr-tracked-1", "run-xyz")
    report = await janitor.reap_all("run-xyz")

    assert sorted(report.discovered_ids) == ["ctr-orphan-2", "ctr-tracked-1"]
    assert sorted(report.removed_ids) == ["ctr-orphan-2", "ctr-tracked-1"]
    assert report.stubborn_ids == []
    assert rm_batches == [["ctr-orphan-2", "ctr-tracked-1"]]
    assert janitor.tracked_ids("run-xyz") == set()


async def test_container_janitor_with_fake_provider_and_stubborn_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_provider = FakeSandboxProvider()
    janitor = ContainerJanitor(provider=fake_provider)
    handle = await fake_provider.start(ContainerConfig(image="alpine:3.20"), run_id="run-fake")
    janitor.register_handle(handle, "run-fake")

    async def no_docker(*args: str, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        raise FileNotFoundError("no docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_docker)

    report = await janitor.reap_all("run-fake")
    assert report.removed_ids == [handle.container_id]
    assert report.stubborn_ids == []
    assert fake_provider.removed_ids == [handle.container_id]


async def test_container_janitor_dedupes_short_and_full_ids_and_handles_no_such_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_id = "0f94585965c413ed57e6b4c1a7cecd69cb01c11efefc8a1a36228d0d49a96f27"
    short_id = full_id[:12]
    other_id = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"

    # Case 1: tracked has 64-char ID while docker ps returns 12-char prefix -> deduped cleanly
    ps_args_seen: list[list[str]] = []
    rm_batches: list[list[str]] = []
    ps_calls = 0

    async def fake_exec_dedupe(*args: str, **kwargs: Any) -> Any:
        nonlocal ps_calls
        _ = kwargs
        if args[1] == "ps":
            ps_args_seen.append(list(args))
            ps_calls += 1
            if ps_calls == 1:
                return make_fake_process(returncode=0, stdout=f"{short_id}\n".encode())
            return make_fake_process(returncode=0, stdout=b"")
        if args[1] == "rm":
            rm_batches.append(list(args[3:]))
            return make_fake_process(returncode=0, stdout=f"{full_id}\n".encode())
        return make_fake_process(returncode=1, stderr=b"unexpected")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec_dedupe)
    janitor = ContainerJanitor()
    janitor.register(full_id, "run-dedupe")
    rep = await janitor.reap_all("run-dedupe")

    assert all("--no-trunc" in cmd for cmd in ps_args_seen)
    assert rep.discovered_ids == [full_id]
    assert rm_batches == [[full_id]]
    assert rep.removed_ids == [full_id]
    assert rep.stubborn_ids == []
    assert rep.errors == []
    assert janitor.tracked_ids("run-dedupe") == set()

    # Case 2: concurrent removal race (one removed on stdout, one "No such container" rc=1)
    ps_calls = 0

    async def fake_exec_race(*args: str, **kwargs: Any) -> Any:
        nonlocal ps_calls
        _ = kwargs
        if args[1] == "ps":
            ps_calls += 1
            if ps_calls == 1:
                return make_fake_process(returncode=0, stdout=f"{full_id}\n".encode())
            return make_fake_process(returncode=0, stdout=b"")
        if args[1] == "rm":
            return make_fake_process(
                returncode=1,
                stdout=f"{full_id}\n".encode(),
                stderr=f"Error response from daemon: No such container: {other_id}\n".encode(),
            )
        return make_fake_process(returncode=1, stderr=b"unexpected")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec_race)
    janitor_race = ContainerJanitor()
    janitor_race.register(full_id, "run-race")
    janitor_race.register(other_id, "run-race")
    rep_race = await janitor_race.reap_all("run-race")
    assert sorted(rep_race.removed_ids) == sorted([full_id, other_id])
    assert rep_race.stubborn_ids == []
    assert rep_race.errors == []
    assert janitor_race.tracked_ids("run-race") == set()

    # Case 3: real daemon failure (permission denied) still flags stubborn_ids & errors
    async def fake_exec_denied(*args: str, **kwargs: Any) -> Any:
        _ = kwargs
        if args[1] == "ps":
            return make_fake_process(returncode=0, stdout=f"{full_id}\n".encode())
        if args[1] == "rm":
            return make_fake_process(returncode=1, stderr=b"permission denied\n")
        return make_fake_process(returncode=1, stderr=b"unexpected")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec_denied)
    janitor_err = ContainerJanitor()
    janitor_err.register(full_id, "run-err")
    rep_err = await janitor_err.reap_all("run-err")
    assert rep_err.stubborn_ids == [full_id]
    assert any("permission denied" in err for err in rep_err.errors)


async def test_runtime_boundary_probes_clean_vs_vulnerable(tmp_path: Path) -> None:
    spec = EnvSpec(
        task_id="probe-test",
        standard=EnvStandard.HARBOR,
        root_path=tmp_path,
        instruction_text="Solve the task.",
        verifier_paths=["tests/", "tests/test.sh"],
        metadata={"supports_interactive_reset": True},
    )

    # 1. Hardened container -> zero runtime findings
    def clean_exec(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (handle, timeout_sec, workdir)
        joined = " ".join(cmd)
        if "os.access" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"exists": [], "writable": []}),
                stderr="",
                duration_ms=2.0,
            )
        if "egress_reachable" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"egress_reachable": [], "leaked_env_keys": []}),
                stderr="",
                duration_ms=2.0,
            )
        if "surviving" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"surviving": []}),
                stderr="",
                duration_ms=2.0,
            )
        return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

    clean_provider = FakeSandboxProvider(exec_handler=clean_exec, diff_entries=[])
    clean_handle = await clean_provider.start(ContainerConfig(image="hardened"), run_id="r1")
    reset_called = False

    async def fake_reset() -> None:
        nonlocal reset_called
        reset_called = True

    clean_findings = await run_runtime_boundary_probes(
        clean_provider, clean_handle, spec, reset_fn=fake_reset
    )
    assert reset_called is True
    assert clean_findings == []

    # 2. Vulnerable container -> triggers BNDY-005, BNDY-006 (x2), HERM-003, HERM-004
    def vuln_exec(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (handle, timeout_sec, workdir)
        joined = " ".join(cmd)
        if "os.access" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"exists": ["/app/tests"], "writable": ["/app/tests"]}),
                stderr="",
                duration_ms=2.0,
            )
        if "egress_reachable" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps(
                    {
                        "egress_reachable": ["1.1.1.1:53"],
                        "leaked_env_keys": ["OPENAI_API_KEY"],
                    }
                ),
                stderr="",
                duration_ms=2.0,
            )
        if "surviving" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"surviving": ["/tmp/.dasv_sentinel"]}),
                stderr="",
                duration_ms=2.0,
            )
        return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

    vuln_provider = FakeSandboxProvider(
        exec_handler=vuln_exec,
        diff_entries=["A /tmp/.dasv_sentinel"],
        emulated_arch=True,
    )
    vuln_handle = await vuln_provider.start(ContainerConfig(image="vuln"), run_id="r2")

    f5 = await probe_writable_verifier_paths(vuln_provider, vuln_handle, spec)
    f6 = await probe_egress_and_env_scrub(vuln_provider, vuln_handle, spec)
    f3 = await probe_reset_residue(vuln_provider, vuln_handle, spec)
    f4 = probe_emulated_arch(vuln_handle, spec)

    assert len(f5) == 1
    assert f5[0].rule_id == "RULE-BNDY-005"
    assert f5[0].severity == Severity.CRITICAL
    assert len(f6) == 2 and all(
        f.rule_id == "RULE-BNDY-006" and f.severity == Severity.CRITICAL for f in f6
    )
    assert len(f3) == 1 and f3[0].rule_id == "RULE-HERM-003" and f3[0].severity == Severity.HIGH
    assert len(f4) == 1 and f4[0].rule_id == "RULE-HERM-004" and f4[0].severity == Severity.MEDIUM

    all_vuln = await run_runtime_boundary_probes(vuln_provider, vuln_handle, spec)
    assert {f.rule_id for f in all_vuln} == {
        "RULE-BNDY-005",
        "RULE-BNDY-006",
        "RULE-HERM-003",
        "RULE-HERM-004",
    }


@pytest.mark.docker
async def test_live_docker_network_none_and_janitor_cleanup() -> None:
    """Runs automatically when a real Docker daemon is reachable on the host."""
    janitor = ContainerJanitor()
    provider = DockerSandboxProvider(janitor=janitor)
    run_id = "dasv-live-smoke"
    handle = await provider.start(
        ContainerConfig(image="python:3.12-slim", network_mode="none"),
        run_id=run_id,
    )
    try:
        spec = EnvSpec(
            task_id="live-smoke",
            standard=EnvStandard.HARBOR,
            root_path=Path("."),
            instruction_text="Live Docker smoke check.",
        )
        findings = await probe_egress_and_env_scrub(provider, handle, spec)
        assert findings == []
    finally:
        reap = await janitor.reap_all(run_id)
        assert reap.stubborn_ids == []
