"""Zero-orphan container lifecycle tracker and reaper (`ContainerJanitor`, SDD-01)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from pydantic import BaseModel, Field

from dasv.sandbox.provider import SandboxHandle, SandboxProvider

logger = logging.getLogger(__name__)


class ReapReport(BaseModel):
    """Summary of containers reaped during teardown for a given run_id."""

    run_id: str
    discovered_ids: list[str] = Field(default_factory=list)
    removed_ids: list[str] = Field(default_factory=list)
    stubborn_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ContainerJanitor:
    """Tracks started containers in memory AND queries `docker ps -aq` by `dasv.run_id`."""

    def __init__(
        self,
        *,
        docker_bin: str = "docker",
        provider: SandboxProvider | None = None,
        batch_size: int = 50,
    ) -> None:
        self._docker_bin = docker_bin
        self._provider = provider
        self._batch_size = max(1, batch_size)
        self._tracked_by_run: dict[str, set[str]] = {}
        self._handles_by_id: dict[str, SandboxHandle] = {}

    def register(self, container_id: str, run_id: str, handle: SandboxHandle | None = None) -> None:
        """Track a started container ID for guaranteed cleanup on exit or signal."""
        cleaned = container_id.strip()
        if not cleaned:
            return
        self._tracked_by_run.setdefault(run_id, set()).add(cleaned)
        if handle is not None:
            self._handles_by_id[cleaned] = handle

    def register_handle(self, handle: SandboxHandle, run_id: str | None = None) -> None:
        """Convenience helper to track a `SandboxHandle`."""
        effective_run_id = run_id if run_id is not None else handle.run_id
        self.register(handle.container_id, effective_run_id, handle=handle)

    def unregister(self, container_id: str) -> None:
        """Remove a container ID once it has been confirmed stopped and removed."""
        cleaned = container_id.strip()
        for ids in self._tracked_by_run.values():
            ids.discard(cleaned)
        self._handles_by_id.pop(cleaned, None)

    def tracked_ids(self, run_id: str | None = None) -> set[str]:
        """Return currently tracked container IDs (optionally scoped to `run_id`)."""
        if run_id is not None:
            return set(self._tracked_by_run.get(run_id, set()))
        merged: set[str] = set()
        for ids in self._tracked_by_run.values():
            merged.update(ids)
        return merged

    async def _run_docker_cmd(
        self, args: list[str], timeout_sec: float
    ) -> tuple[int, str, str] | None:
        """Run a docker CLI command safely without shell=True; return None if docker is absent."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._docker_bin,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            return 1, "", str(exc)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except TimeoutError:
            with suppress(ProcessLookupError):
                proc.kill()
            with suppress(Exception):
                await proc.communicate()
            return 124, "", f"docker {' '.join(args[:2])} timed out after {timeout_sec:.1f}s"

        return (
            proc.returncode if proc.returncode is not None else 1,
            stdout_b.decode("utf-8", errors="replace"),
            stderr_b.decode("utf-8", errors="replace"),
        )

    async def list_labeled_containers(self, run_id: str, *, timeout_sec: float = 5.0) -> list[str]:
        """Query `docker ps -aq --filter label=dasv.run_id=<run_id>`."""
        if not run_id.strip():
            return []
        res = await self._run_docker_cmd(
            ["ps", "-aq", "--filter", f"label=dasv.run_id={run_id}"],
            timeout_sec=timeout_sec,
        )
        if res is None:
            return []
        rc, stdout, _stderr = res
        if rc != 0:
            return []
        return [line.strip() for line in stdout.splitlines() if line.strip()]

    async def reap_all(self, run_id: str, *, grace_timeout_sec: float = 10.0) -> ReapReport:
        """Force-remove every tracked or labeled container for `run_id` and verify zero orphans."""
        if not run_id.strip():
            return ReapReport(run_id=run_id)
        tracked = set(self._tracked_by_run.get(run_id, set()))
        docker_labeled = set(
            await self.list_labeled_containers(run_id, timeout_sec=min(5.0, grace_timeout_sec))
        )
        all_ids = sorted(tracked | docker_labeled)

        report = ReapReport(run_id=run_id, discovered_ids=all_ids)
        if not all_ids:
            self._tracked_by_run.pop(run_id, None)
            return report

        removed: set[str] = set()
        remaining_for_docker: list[str] = []

        # If a custom SandboxProvider is bound, reap tracked handles through it first.
        if self._provider is not None:
            for cid in all_ids:
                handle = self._handles_by_id.get(cid)
                if handle is None and cid in tracked:
                    handle = SandboxHandle(
                        container_id=cid,
                        image_digest="unknown",
                        resolved_arch="unknown",
                        emulated_arch=False,
                        run_id=run_id,
                    )
                if handle is not None:
                    try:
                        await asyncio.wait_for(
                            self._provider.stop_and_remove(handle),
                            timeout=grace_timeout_sec,
                        )
                        removed.add(cid)
                    except Exception as exc:
                        report.errors.append(f"provider stop_and_remove({cid}) failed: {exc}")
                        remaining_for_docker.append(cid)
                else:
                    remaining_for_docker.append(cid)
        else:
            remaining_for_docker = list(all_ids)

        # Force-remove any Docker-backed or discovered containers in batches.
        if remaining_for_docker:
            for idx in range(0, len(remaining_for_docker), self._batch_size):
                batch = remaining_for_docker[idx : idx + self._batch_size]
                res = await self._run_docker_cmd(
                    ["rm", "-f", *batch], timeout_sec=grace_timeout_sec
                )
                if res is None:
                    # Docker binary not present on host (e.g., unit test environment).
                    report.errors.append(
                        f"docker CLI unavailable while reaping container(s): {', '.join(batch)}"
                    )
                    continue
                rc, _stdout, stderr = res
                if rc == 0:
                    removed.update(batch)
                else:
                    report.errors.append(
                        f"docker rm -f exited {rc}: {stderr.strip() or 'unknown error'}"
                    )

        # Post-reap verification against live Docker labels.
        survivors_docker = set(
            await self.list_labeled_containers(run_id, timeout_sec=min(5.0, grace_timeout_sec))
        )
        unremoved = sorted((set(all_ids) - removed) | survivors_docker)

        for cid in removed - survivors_docker:
            self.unregister(cid)

        if not unremoved:
            self._tracked_by_run.pop(run_id, None)
        else:
            self._tracked_by_run[run_id] = set(unremoved)
            for cid in unremoved:
                logger.error("RUNTIME_ERROR: stubborn container survived reap_all: %s", cid)

        report.removed_ids = sorted(removed - survivors_docker)
        report.stubborn_ids = unremoved
        return report
