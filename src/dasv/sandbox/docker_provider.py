"""Hardened async Docker CLI `SandboxProvider` implementation (SDD-01, ARCH-COMP-04)."""

from __future__ import annotations

import asyncio
import os
import re
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal

from dasv.core.errors import RuntimeSandboxError, SandboxSecurityError
from dasv.sandbox.janitor import ContainerJanitor
from dasv.sandbox.provider import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ContainerConfig,
    ExecResult,
    RuntimeHealth,
    SandboxHandle,
    current_host_arch,
    normalize_arch,
    truncate_output,
)

_FORBIDDEN_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "MISTRAL_API_KEY",
        "XAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    }
)

_SECRET_VALUE_RE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{12,}\b"
    r"|\bghp_[A-Za-z0-9]{10,}\b"
    r"|\bgithub_pat_[A-Za-z0-9_]{16,}\b"
    r"|\bhf_[A-Za-z0-9]{10,}\b"
    r"|\bAKIA[0-9A-Z]{16}\b)"
)
_DOCKER_MEM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP]?i?B?)\s*$", re.IGNORECASE)


def _is_same_or_child(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def validate_bind_mount(
    host_path: Path,
    container_path: str,
    mode: Literal["ro", "rw"],
) -> Path:
    """Enforce host mount security guardrails (SDD-01 §3.2); return normalized host path."""
    if mode not in {"ro", "rw"}:
        raise SandboxSecurityError(f"Unsupported bind mount mode '{mode}'; expected 'ro' or 'rw'.")
    if (
        not container_path
        or not container_path.startswith("/")
        or any(ch in container_path for ch in (":", "\n", "\r", "\x00"))
    ):
        raise SandboxSecurityError(
            f"Container bind mount target '{container_path}' must be a safe absolute POSIX path."
        )

    raw_str = str(host_path)
    if any(ch in raw_str for ch in ("\n", "\r", "\x00")):
        raise SandboxSecurityError(f"Forbidden control character in host bind mount '{raw_str}'.")

    expanded = host_path.expanduser()
    resolved = expanded.resolve(strict=False)
    norm_no_symlink = Path(os.path.normpath(str(expanded)))
    if ":" in str(resolved):
        raise SandboxSecurityError(
            f"Forbidden ':' character in resolved host bind mount '{resolved}'."
        )

    home_raw = Path.home()
    home_resolved = home_raw.resolve(strict=False)
    ssh_raw = home_raw / ".ssh"
    ssh_resolved = ssh_raw.resolve(strict=False)

    etc_roots = (Path("/etc"), Path("/private/etc"))
    docker_sock_paths = (
        Path("/var/run/docker.sock"),
        Path("/private/var/run/docker.sock"),
        Path("/run/docker.sock"),
    )
    docker_sock_dirs = (Path("/var/run"), Path("/private/var/run"), Path("/run"))

    candidates = (expanded, norm_no_symlink, resolved)

    for candidate in candidates:
        if candidate == Path("/"):
            raise SandboxSecurityError(
                f"Forbidden host bind mount '{raw_str}': mounting root '/' is prohibited."
            )
        if candidate in {home_raw, home_resolved}:
            raise SandboxSecurityError(
                f"Forbidden host bind mount '{raw_str}': mounting $HOME is prohibited."
            )
        if _is_same_or_child(candidate, ssh_raw) or _is_same_or_child(candidate, ssh_resolved):
            raise SandboxSecurityError(
                f"Forbidden host bind mount '{raw_str}': mounting ~/.ssh is prohibited."
            )
        if any(_is_same_or_child(candidate, etc_root) for etc_root in etc_roots):
            raise SandboxSecurityError(
                f"Forbidden host bind mount '{raw_str}': mounting /etc is prohibited."
            )
        if (
            candidate in docker_sock_paths
            or candidate in docker_sock_dirs
            or candidate.name == "docker.sock"
        ):
            raise SandboxSecurityError(
                f"Forbidden host bind mount '{raw_str}': mounting Docker socket is prohibited."
            )

    return resolved


def validate_env_vars(env_vars: dict[str, str]) -> None:
    """Reject passing host API keys or raw secret tokens into untrusted containers."""
    for key, value in env_vars.items():
        if key.upper() in _FORBIDDEN_ENV_KEYS:
            raise SandboxSecurityError(
                f"Forbidden sandbox env var '{key}': host API keys must never enter containers."
            )
        if _SECRET_VALUE_RE.search(value):
            raise SandboxSecurityError(
                f"Forbidden secret token pattern detected in sandbox env var '{key}'."
            )


def _parse_docker_mem_mb(raw_usage: str) -> float | None:
    """Parse Docker stats MemUsage string like '42.5MiB / 2GiB' into MB."""
    first = raw_usage.split("/", 1)[0].strip()
    match = _DOCKER_MEM_RE.match(first)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"b", ""}:
        return value / (1024.0 * 1024.0)
    if unit in {"k", "kb", "kib"}:
        return value / 1024.0
    if unit in {"m", "mb", "mib"}:
        return value
    if unit in {"g", "gb", "gib"}:
        return value * 1024.0
    if unit in {"t", "tb", "tib"}:
        return value * 1024.0 * 1024.0
    return None


def build_docker_run_args(cfg: ContainerConfig, run_id: str) -> list[str]:
    """Build deterministic, hardened `docker run` argument vector (never shell=True)."""
    if not run_id.strip():
        raise SandboxSecurityError("run_id must be non-empty for container labeling and reaping.")

    validate_env_vars(cfg.env_vars)

    network_mode = "none" if cfg.plane == "verification" else cfg.network_mode
    args: list[str] = [
        "run",
        "--detach",
        "--init",
        f"--network={network_mode}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        f"--cpus={cfg.cpus}",
        f"--memory={cfg.memory}",
        f"--memory-swap={cfg.memory}",
        f"--pids-limit={cfg.pids_limit}",
    ]

    if cfg.read_only_root:
        args.append("--read-only")
    if cfg.platform:
        args.extend(["--platform", cfg.platform])
    if cfg.workdir:
        args.extend(["--workdir", cfg.workdir])

    for dest, opts in sorted(cfg.tmpfs_mounts.items()):
        args.extend(["--tmpfs", f"{dest}:{opts}"])

    for host_path, container_path, mode in cfg.bind_mounts:
        safe_host = validate_bind_mount(host_path, container_path, mode)
        args.extend(["--volume", f"{safe_host}:{container_path}:{mode}"])

    merged_labels = dict(cfg.labels)
    merged_labels["dasv.managed"] = "true"
    merged_labels["dasv.run_id"] = run_id
    merged_labels["dasv.plane"] = cfg.plane
    for label_k, label_v in sorted(merged_labels.items()):
        args.extend(["--label", f"{label_k}={label_v}"])

    for env_k, env_v in sorted(cfg.env_vars.items()):
        args.extend(["-e", f"{env_k}={env_v}"])

    args.append(cfg.image)
    args.extend(cfg.command)
    return args


class DockerSandboxProvider:
    """Async Docker CLI runtime implementing `SandboxProvider` with strict isolation."""

    def __init__(
        self,
        *,
        docker_bin: str = "docker",
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        janitor: ContainerJanitor | None = None,
    ) -> None:
        self._docker_bin = docker_bin
        self._max_output_bytes = max_output_bytes
        self._janitor = janitor

    async def _run_docker(
        self,
        args: list[str],
        *,
        timeout_sec: float | None = None,
    ) -> tuple[int, bytes, bytes, bool]:
        """Execute `docker <args>` strictly via create_subprocess_exec (no shell)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._docker_bin,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeSandboxError(
                f"Docker CLI '{self._docker_bin}' not found on PATH. "
                "Install Docker Desktop or OrbStack and ensure the daemon is running."
            ) from exc
        except OSError as exc:
            raise RuntimeSandboxError(
                f"Failed to invoke Docker CLI '{self._docker_bin}': {exc}"
            ) from exc

        try:
            if timeout_sec is not None:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_sec,
                )
            else:
                stdout_b, stderr_b = await proc.communicate()
        except TimeoutError:
            with suppress(ProcessLookupError):
                proc.kill()
            partial_out = b""
            partial_err = b""
            with suppress(Exception):
                partial_out, partial_err = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            return 124, partial_out, partial_err or b"command timed out", True

        rc = proc.returncode if proc.returncode is not None else 1
        return rc, stdout_b, stderr_b, False

    async def health_check(self) -> RuntimeHealth:
        """Check Docker daemon reachability, server version, and host/server arch parity."""
        host_arch = current_host_arch()
        try:
            rc, stdout_b, stderr_b, timed_out = await self._run_docker(
                ["info", "--format", "{{.ServerVersion}}|{{.Architecture}}"],
                timeout_sec=5.0,
            )
        except RuntimeSandboxError as exc:
            return RuntimeHealth(
                healthy=False,
                runtime="docker",
                host_arch=host_arch,
                detail=str(exc),
            )

        if timed_out or rc != 0:
            detail = (
                stderr_b.decode("utf-8", errors="replace").strip()
                or stdout_b.decode("utf-8", errors="replace").strip()
                or "Docker daemon unreachable. Start OrbStack or Docker Desktop."
            )
            return RuntimeHealth(
                healthy=False,
                runtime="docker",
                host_arch=host_arch,
                detail=detail,
            )

        raw = stdout_b.decode("utf-8", errors="replace").strip()
        parts = raw.split("|", 1)
        server_version = parts[0].strip() or None
        server_arch = normalize_arch(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        emulated = server_arch is not None and server_arch != "unknown" and server_arch != host_arch
        return RuntimeHealth(
            healthy=True,
            runtime="docker",
            server_version=server_version,
            host_arch=host_arch,
            server_arch=server_arch,
            emulated_arch_warning=emulated,
            detail="reachable",
        )

    async def build_image(
        self,
        context_dir: Path,
        dockerfile: Path,
        tag: str,
        platform: str | None = None,
    ) -> str:
        """Build an OCI image and return its immutable image ID / digest."""
        args = ["build", "-f", str(dockerfile), "-t", tag]
        if platform:
            args.extend(["--platform", platform])
        args.append(str(context_dir))

        rc, _stdout_b, stderr_b, timed_out = await self._run_docker(args, timeout_sec=600.0)
        if timed_out or rc != 0:
            err_msg = stderr_b.decode("utf-8", errors="replace").strip()
            raise RuntimeSandboxError(f"docker build failed (exit {rc}): {err_msg}")

        inspect_rc, inspect_out, inspect_err, _ = await self._run_docker(
            ["image", "inspect", "--format", "{{.Id}}", tag],
            timeout_sec=10.0,
        )
        if inspect_rc != 0:
            err_msg = inspect_err.decode("utf-8", errors="replace").strip()
            raise RuntimeSandboxError(f"docker image inspect failed for '{tag}': {err_msg}")
        return inspect_out.decode("utf-8", errors="replace").strip()

    async def _inspect_image_metadata(
        self, container_id: str, fallback_image: str, requested_platform: str | None
    ) -> tuple[str, str]:
        """Resolve image digest and architecture for a running container."""
        digest = fallback_image
        resolved_arch = (
            normalize_arch(requested_platform) if requested_platform else current_host_arch()
        )

        rc, out_b, _err_b, timed_out = await self._run_docker(
            ["inspect", "--format", "{{.Image}}", container_id],
            timeout_sec=5.0,
        )
        if not timed_out and rc == 0:
            img_id = out_b.decode("utf-8", errors="replace").strip()
            if img_id:
                digest = img_id
                img_rc, img_out, _img_err, img_timeout = await self._run_docker(
                    ["image", "inspect", "--format", "{{.Architecture}}|{{.Id}}", img_id],
                    timeout_sec=5.0,
                )
                if not img_timeout and img_rc == 0:
                    parts = img_out.decode("utf-8", errors="replace").strip().split("|", 1)
                    if parts and parts[0].strip():
                        resolved_arch = normalize_arch(parts[0])
                    if len(parts) > 1 and parts[1].strip():
                        digest = parts[1].strip()
        return digest, resolved_arch

    async def start(self, cfg: ContainerConfig, run_id: str) -> SandboxHandle:
        """Start a hardened detached container and register it with ContainerJanitor."""
        run_args = build_docker_run_args(cfg, run_id)
        rc, stdout_b, stderr_b, timed_out = await self._run_docker(run_args, timeout_sec=60.0)
        if timed_out or rc != 0:
            err_msg = stderr_b.decode("utf-8", errors="replace").strip()
            raise RuntimeSandboxError(
                f"Failed to start sandbox container for image '{cfg.image}' (exit {rc}): {err_msg}"
            )

        container_id = stdout_b.decode("utf-8", errors="replace").strip().splitlines()[-1].strip()
        if not container_id:
            raise RuntimeSandboxError("docker run succeeded but returned an empty container ID.")

        if self._janitor is not None:
            self._janitor.register(container_id, run_id)

        image_digest, resolved_arch = await self._inspect_image_metadata(
            container_id, cfg.image, cfg.platform
        )
        host_arch = current_host_arch()
        emulated_arch = resolved_arch not in {"unknown", host_arch}

        handle = SandboxHandle(
            container_id=container_id,
            image_digest=image_digest,
            resolved_arch=resolved_arch,
            emulated_arch=emulated_arch,
            plane=cfg.plane,
            run_id=run_id,
        )
        if self._janitor is not None:
            self._janitor.register_handle(handle, run_id)
        return handle

    async def _sample_peak_memory_mb(self, container_id: str) -> float | None:
        rc, out_b, _err_b, timed_out = await self._run_docker(
            ["stats", "--no-stream", "--format", "{{.MemUsage}}", container_id],
            timeout_sec=3.0,
        )
        if timed_out or rc != 0:
            return None
        return _parse_docker_mem_mb(out_b.decode("utf-8", errors="replace").strip())

    async def exec(
        self,
        handle: SandboxHandle,
        cmd: list[str],
        timeout_sec: float,
        workdir: str | None = None,
    ) -> ExecResult:
        """Execute `cmd` inside `handle` with hard wall-clock timeout and output truncation."""
        args = ["exec"]
        if workdir:
            args.extend(["-w", workdir])
        args.append(handle.container_id)
        args.extend(cmd)

        start_ts = time.perf_counter()
        rc, stdout_b, stderr_b, timed_out = await self._run_docker(args, timeout_sec=timeout_sec)
        duration_ms = (time.perf_counter() - start_ts) * 1000.0

        if timed_out:
            # Force-terminate runaway container processes on timeout per SDD-01 §5.
            with suppress(RuntimeSandboxError):
                await self._run_docker(["kill", handle.container_id], timeout_sec=5.0)
            stdout_str, out_trunc = truncate_output(stdout_b, self._max_output_bytes)
            stderr_str, err_trunc = truncate_output(
                stderr_b or f"Execution timed out after {timeout_sec:.1f}s".encode(),
                self._max_output_bytes,
            )
            return ExecResult(
                exit_code=124,
                stdout=stdout_str,
                stderr=stderr_str,
                duration_ms=round(duration_ms, 3),
                timed_out=True,
                truncated=out_trunc or err_trunc,
                peak_memory_mb=None,
            )

        stdout_str, out_trunc = truncate_output(stdout_b, self._max_output_bytes)
        stderr_str, err_trunc = truncate_output(stderr_b, self._max_output_bytes)
        peak_mem = await self._sample_peak_memory_mb(handle.container_id)
        return ExecResult(
            exit_code=rc,
            stdout=stdout_str,
            stderr=stderr_str,
            duration_ms=round(duration_ms, 3),
            timed_out=False,
            truncated=out_trunc or err_trunc,
            peak_memory_mb=peak_mem,
        )

    async def copy_from(
        self,
        handle: SandboxHandle,
        container_path: str,
        host_dest: Path,
    ) -> None:
        """Copy scoped artifact files from container into host staging directory."""
        host_dest.parent.mkdir(parents=True, exist_ok=True)
        rc, _stdout_b, stderr_b, timed_out = await self._run_docker(
            ["cp", f"{handle.container_id}:{container_path}", str(host_dest)],
            timeout_sec=30.0,
        )
        if timed_out or rc != 0:
            err_msg = stderr_b.decode("utf-8", errors="replace").strip()
            raise RuntimeSandboxError(
                f"docker cp failed for '{handle.container_id}:{container_path}': {err_msg}"
            )

    async def diff_fs(self, handle: SandboxHandle) -> list[str]:
        """Return sorted container filesystem delta lines from `docker diff`."""
        rc, stdout_b, stderr_b, timed_out = await self._run_docker(
            ["diff", handle.container_id],
            timeout_sec=15.0,
        )
        if timed_out or rc != 0:
            err_msg = stderr_b.decode("utf-8", errors="replace").strip()
            raise RuntimeSandboxError(f"docker diff failed for '{handle.container_id}': {err_msg}")
        lines = [
            line.strip()
            for line in stdout_b.decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        return sorted(lines)

    async def stop_and_remove(self, handle: SandboxHandle) -> None:
        """Force-stop and remove a sandbox container and untrack it from ContainerJanitor."""
        rc, _stdout_b, stderr_b, timed_out = await self._run_docker(
            ["rm", "-f", handle.container_id],
            timeout_sec=15.0,
        )
        if timed_out or rc != 0:
            err_msg = stderr_b.decode("utf-8", errors="replace").strip()
            if "no such container" not in err_msg.lower():
                raise RuntimeSandboxError(
                    f"docker rm -f failed for '{handle.container_id}': {err_msg}"
                )
        if self._janitor is not None:
            self._janitor.unregister(handle.container_id)
