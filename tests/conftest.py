"""Shared pytest hooks (auto-skip @pytest.mark.docker without daemon) and FakeSandboxProvider."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from dasv.sandbox.provider import (
    ContainerConfig,
    ExecResult,
    RuntimeHealth,
    SandboxHandle,
    current_host_arch,
    normalize_arch,
)


def _docker_daemon_available() -> bool:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        return False
    try:
        proc = subprocess.run(
            [docker_bin, "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    _ = config
    has_docker = _docker_daemon_available()
    if has_docker:
        return
    skip_docker = pytest.mark.skip(reason="Docker daemon is not available on this host")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_docker)


class FakeSandboxProvider:
    """Deterministic in-memory SandboxProvider for unit & adversarial tests."""

    def __init__(
        self,
        *,
        exec_handler: Callable[[SandboxHandle, list[str], float, str | None], ExecResult]
        | None = None,
        diff_entries: list[str] | None = None,
        resolved_arch: str | None = None,
        emulated_arch: bool = False,
        healthy: bool = True,
        copy_files: dict[str, bytes] | None = None,
    ) -> None:
        self._exec_handler = exec_handler
        self._diff_entries = list(diff_entries or [])
        self._resolved_arch = resolved_arch or current_host_arch()
        self._emulated_arch = emulated_arch
        self._healthy = healthy
        self._copy_files = dict(copy_files or {})

        self.started_configs: list[tuple[ContainerConfig, str]] = []
        self.active_handles: dict[str, SandboxHandle] = {}
        self.removed_ids: list[str] = []
        self.exec_calls: list[tuple[str, list[str], float, str | None]] = []
        self._counter = 0

    async def health_check(self) -> RuntimeHealth:
        host_arch = current_host_arch()
        return RuntimeHealth(
            healthy=self._healthy,
            runtime="fake-docker",
            server_version="27.0.0-fake",
            host_arch=host_arch,
            server_arch=self._resolved_arch,
            emulated_arch_warning=self._emulated_arch,
            detail="reachable" if self._healthy else "daemon unavailable",
        )

    async def build_image(
        self,
        context_dir: Path,
        dockerfile: Path,
        tag: str,
        platform: str | None = None,
    ) -> str:
        _ = (context_dir, dockerfile, platform)
        return f"sha256:{tag.replace(':', '_')}_{'0' * 32}"

    async def start(self, cfg: ContainerConfig, run_id: str) -> SandboxHandle:
        self._counter += 1
        cid = f"fake-ctr-{cfg.plane}-{self._counter:03d}"
        arch = normalize_arch(cfg.platform) if cfg.platform else self._resolved_arch
        emulated = self._emulated_arch or (arch not in {"unknown", current_host_arch()})
        handle = SandboxHandle(
            container_id=cid,
            image_digest=cfg.image,
            resolved_arch=arch,
            emulated_arch=emulated,
            plane=cfg.plane,
            run_id=run_id,
        )
        self.started_configs.append((cfg, run_id))
        self.active_handles[cid] = handle
        return handle

    async def exec(
        self,
        handle: SandboxHandle,
        cmd: list[str],
        timeout_sec: float,
        workdir: str | None = None,
    ) -> ExecResult:
        self.exec_calls.append((handle.container_id, list(cmd), timeout_sec, workdir))
        if self._exec_handler is not None:
            return self._exec_handler(handle, cmd, timeout_sec, workdir)
        return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

    async def copy_from(
        self,
        handle: SandboxHandle,
        container_path: str,
        host_dest: Path,
    ) -> None:
        _ = handle
        host_dest.parent.mkdir(parents=True, exist_ok=True)
        content = self._copy_files.get(container_path, b"")
        host_dest.write_bytes(content)

    async def diff_fs(self, handle: SandboxHandle) -> list[str]:
        _ = handle
        return list(self._diff_entries)

    async def stop_and_remove(self, handle: SandboxHandle) -> None:
        self.active_handles.pop(handle.container_id, None)
        self.removed_ids.append(handle.container_id)


def make_fake_process(
    *,
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    delay_sec: float = 0.0,
) -> Any:
    """Create a lightweight fake asyncio.subprocess.Process for subprocess unit tests."""
    import asyncio

    class _FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = returncode
            self.killed = False

        async def communicate(self) -> tuple[bytes, bytes]:
            if delay_sec > 0.0:
                await asyncio.sleep(delay_sec)
            return stdout, stderr

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    return _FakeProcess()


from tests.unit.conftest import (  # noqa: E402
    hardened_harbor_env as hardened_harbor_env,
)
from tests.unit.conftest import (  # noqa: E402
    legacy_verifiers_v0 as legacy_verifiers_v0,
)
from tests.unit.conftest import (  # noqa: E402
    minimal_openenv as minimal_openenv,
)
from tests.unit.conftest import (  # noqa: E402
    minimal_verifiers_v1 as minimal_verifiers_v1,
)
from tests.unit.conftest import (  # noqa: E402
    vulnerable_harbor_env as vulnerable_harbor_env,
)
