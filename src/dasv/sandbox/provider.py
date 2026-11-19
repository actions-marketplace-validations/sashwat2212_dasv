"""Pluggable OCI sandbox runtime protocol and domain models (SDD-01, ARCH-COMP-04)."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

DEFAULT_MAX_OUTPUT_BYTES = 65_536
_TRUNCATION_MARKER = "\n...[truncated by dasv]"

_ARCH_ALIASES: dict[str, str] = {
    "x86_64": "amd64",
    "x64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "arm64v8": "arm64",
}


def normalize_arch(raw_arch: str | None) -> str:
    """Normalize CPU/OCI architecture names (e.g. x86_64 -> amd64, aarch64 -> arm64)."""
    if not raw_arch:
        return "unknown"
    cleaned = raw_arch.strip().lower()
    if "/" in cleaned:
        parts = [p for p in cleaned.split("/") if p]
        if len(parts) >= 2:
            cleaned = parts[1]
    return _ARCH_ALIASES.get(cleaned, cleaned)


def current_host_arch() -> str:
    """Return normalized architecture of the current host."""
    return normalize_arch(platform.machine())


def truncate_output(
    raw: bytes,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> tuple[str, bool]:
    """Deterministically decode and cap output bytes to `max_bytes`."""
    if max_bytes <= 0:
        return "", bool(raw)
    if len(raw) <= max_bytes:
        return raw.decode("utf-8", errors="replace"), False
    marker_bytes = _TRUNCATION_MARKER.encode("utf-8")
    keep = max(0, max_bytes - len(marker_bytes))
    truncated_text = raw[:keep].decode("utf-8", errors="replace") + _TRUNCATION_MARKER
    return truncated_text, True


def _empty_bind_mounts() -> list[tuple[Path, str, Literal["ro", "rw"]]]:
    return []


class ContainerConfig(BaseModel):
    """Hardened container launch specification for Execution or Verification Plane."""

    image: str
    platform: str | None = None
    network_mode: Literal["none", "bridge"] = "none"
    cpus: float = Field(default=2.0, gt=0.0)
    memory: str = "2g"
    pids_limit: int = Field(default=256, ge=16)
    read_only_root: bool = False
    tmpfs_mounts: dict[str, str] = Field(
        default_factory=lambda: {"/tmp": "rw,noexec,nosuid,size=512m"}
    )
    bind_mounts: list[tuple[Path, str, Literal["ro", "rw"]]] = Field(
        default_factory=_empty_bind_mounts
    )
    env_vars: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    plane: Literal["execution", "verification"] = "execution"
    workdir: str | None = None
    command: list[str] = Field(
        default_factory=lambda: ["sh", "-c", "trap 'exit 0' TERM INT; while :; do sleep 60; done"]
    )


class ExecResult(BaseModel):
    """Deterministic result from executing a command inside a sandbox container."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    truncated: bool = False
    peak_memory_mb: float | None = None


class SandboxHandle(BaseModel):
    """Opaque handle referencing a running sandbox container."""

    container_id: str
    image_digest: str
    resolved_arch: str
    emulated_arch: bool
    plane: Literal["execution", "verification"] = "execution"
    run_id: str = ""


class RuntimeHealth(BaseModel):
    """Health and architecture parity report from the underlying container runtime."""

    healthy: bool
    runtime: str = "docker"
    server_version: str | None = None
    host_arch: str
    server_arch: str | None = None
    emulated_arch_warning: bool = False
    detail: str = ""


@runtime_checkable
class SandboxProvider(Protocol):
    """Pluggable OCI sandbox runtime protocol (Docker default, Modal/Podman extensible)."""

    async def health_check(self) -> RuntimeHealth: ...

    async def build_image(
        self,
        context_dir: Path,
        dockerfile: Path,
        tag: str,
        platform: str | None = None,
    ) -> str: ...

    async def start(self, cfg: ContainerConfig, run_id: str) -> SandboxHandle: ...

    async def exec(
        self,
        handle: SandboxHandle,
        cmd: list[str],
        timeout_sec: float,
        workdir: str | None = None,
    ) -> ExecResult: ...

    async def copy_from(
        self,
        handle: SandboxHandle,
        container_path: str,
        host_dest: Path,
    ) -> None: ...

    async def diff_fs(self, handle: SandboxHandle) -> list[str]: ...

    async def stop_and_remove(self, handle: SandboxHandle) -> None: ...
