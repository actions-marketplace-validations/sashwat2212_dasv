"""Zero-orphan container lifecycle tracker and reaper (`ContainerJanitor`, SDD-01)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from contextlib import suppress

from pydantic import BaseModel, Field

from dasv.sandbox.provider import SandboxHandle, SandboxProvider

logger = logging.getLogger(__name__)

_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def _is_hex_container_id(cid: str) -> bool:
    return len(cid) >= 12 and all(ch in _HEX_CHARS for ch in cid)


def _same_container_id(a: str, b: str) -> bool:
    """Return True if two container IDs are equal or 12+ char hex prefix aliases of one another."""
    if a == b:
        return True
    if _is_hex_container_id(a) and _is_hex_container_id(b):
        return a.startswith(b) or b.startswith(a)
    return False


def _dedupe_container_ids(ids: Iterable[str]) -> list[str]:
    """Deduplicate IDs, collapsing shorter hex prefixes into matching longer hex IDs."""
    cleaned = sorted({cid.strip() for cid in ids if cid.strip()})
    hex_ids = [cid for cid in cleaned if _is_hex_container_id(cid)]
    result: list[str] = []
    for cid in cleaned:
        if _is_hex_container_id(cid) and any(
            other != cid and other.startswith(cid) and len(other) > len(cid) for other in hex_ids
        ):
            continue
        result.append(cid)
    return result


def _stderr_is_only_not_found(stderr: str) -> bool:
    """Return True if every non-empty stderr line reports an already-absent container."""
    lines = [line.strip().lower() for line in stderr.splitlines() if line.strip()]
    return bool(lines) and all("no such container" in line for line in lines)


def _matches_run_id(candidate: str, run_id: str) -> bool:
    """Return True if `candidate` equals `run_id` or is a scoped child (`<run_id>-...`)."""
    return candidate == run_id or candidate.startswith(f"{run_id}-")


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
        """Remove a container ID (and any hex prefix/full alias) once confirmed removed."""
        cleaned = container_id.strip()
        if not cleaned:
            return
        for ids in self._tracked_by_run.values():
            to_drop = {existing for existing in ids if _same_container_id(existing, cleaned)}
            ids.difference_update(to_drop)
        for existing in list(self._handles_by_id):
            if _same_container_id(existing, cleaned):
                self._handles_by_id.pop(existing, None)

    def tracked_ids(self, run_id: str | None = None) -> set[str]:
        """Return currently tracked container IDs (matching `run_id` and its child scopes)."""
        if run_id is not None:
            matched: set[str] = set()
            for key, ids in self._tracked_by_run.items():
                if _matches_run_id(key, run_id):
                    matched.update(ids)
            return matched
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
        """Query containers matching exact `dasv.run_id=<run_id>` and scoped `<run_id>-*` kids."""
        if not run_id.strip():
            return []
        matched: list[str] = []

        # 1. Exact run_id label query (preserves strict SDD-01 filter contract & unit mocks).
        exact_res = await self._run_docker_cmd(
            [
                "ps",
                "-aq",
                "--filter",
                f"label=dasv.run_id={run_id}",
                "--no-trunc",
            ],
            timeout_sec=timeout_sec,
        )
        if exact_res is None:
            return []
        rc, stdout, _stderr = exact_res
        if rc == 0:
            for raw_line in stdout.splitlines():
                cid = raw_line.strip()
                if cid:
                    matched.append(cid)

        # 2. Scoped child run_id query (catches `<run_id>-h1`, `<run_id>-gsg-*`, `<run_id>-t2`).
        managed_res = await self._run_docker_cmd(
            [
                "ps",
                "-a",
                "--filter",
                "label=dasv.managed=true",
                "--format",
                '{{.ID}}|{{.Label "dasv.run_id"}}',
                "--no-trunc",
            ],
            timeout_sec=timeout_sec,
        )
        if managed_res is not None:
            m_rc, m_stdout, _m_stderr = managed_res
            if m_rc == 0:
                for raw_line in m_stdout.splitlines():
                    line = raw_line.strip()
                    if "|" not in line:
                        continue
                    cid_part, label_part = line.split("|", 1)
                    if _matches_run_id(label_part.strip(), run_id) and cid_part.strip():
                        matched.append(cid_part.strip())

        return _dedupe_container_ids(matched)

    async def reap_all(self, run_id: str, *, grace_timeout_sec: float = 10.0) -> ReapReport:
        """Force-remove every tracked or labeled container for `run_id` (including child scopes)."""
        if not run_id.strip():
            return ReapReport(run_id=run_id)
        matched_keys = [k for k in self._tracked_by_run if _matches_run_id(k, run_id)]
        tracked: set[str] = set()
        for key in matched_keys:
            tracked.update(self._tracked_by_run.get(key, set()))
        docker_labeled = set(
            await self.list_labeled_containers(run_id, timeout_sec=min(5.0, grace_timeout_sec))
        )
        all_ids = _dedupe_container_ids(tracked | docker_labeled)

        report = ReapReport(run_id=run_id, discovered_ids=all_ids)
        if not all_ids:
            for key in matched_keys:
                self._tracked_by_run.pop(key, None)
            return report

        removed: set[str] = set()
        possibly_gone: set[str] = set()
        remaining_for_docker: list[str] = []

        # If a custom SandboxProvider is bound, reap tracked handles through it first.
        if self._provider is not None:
            for cid in all_ids:
                handle = self._handles_by_id.get(cid)
                if handle is None and any(_same_container_id(cid, t) for t in tracked):
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
                rc, stdout, stderr = res
                if rc == 0:
                    removed.update(batch)
                    continue

                # Parse any partial removals Docker printed to stdout before exiting non-zero.
                printed_removed = [line.strip() for line in stdout.splitlines() if line.strip()]
                for cid in batch:
                    if any(_same_container_id(cid, out_id) for out_id in printed_removed):
                        removed.add(cid)

                if _stderr_is_only_not_found(stderr):
                    for cid in batch:
                        if cid not in removed:
                            possibly_gone.add(cid)
                else:
                    report.errors.append(
                        f"docker rm -f exited {rc}: {stderr.strip() or 'unknown error'}"
                    )

        # Post-reap verification against live Docker labels.
        survivors_docker = set(
            await self.list_labeled_containers(run_id, timeout_sec=min(5.0, grace_timeout_sec))
        )

        # Any container that returned "No such container" and is absent post-reap is gone.
        for cid in possibly_gone:
            if not any(_same_container_id(cid, surv) for surv in survivors_docker):
                removed.add(cid)

        still_alive = {
            cid
            for cid in removed
            if any(_same_container_id(cid, surv) for surv in survivors_docker)
        }
        confirmed_removed = removed - still_alive
        unremoved = _dedupe_container_ids((set(all_ids) - confirmed_removed) | survivors_docker)

        for cid in confirmed_removed:
            self.unregister(cid)

        if not unremoved:
            for key in matched_keys:
                self._tracked_by_run.pop(key, None)
        else:
            for key in matched_keys:
                self._tracked_by_run.pop(key, None)
            self._tracked_by_run[run_id] = set(unremoved)
            for cid in unremoved:
                logger.error("RUNTIME_ERROR: stubborn container survived reap_all: %s", cid)

        report.removed_ids = sorted(confirmed_removed)
        report.stubborn_ids = unremoved
        return report
