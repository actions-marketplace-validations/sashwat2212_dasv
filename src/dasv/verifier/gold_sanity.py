"""3-Invariant GoldSanityGate mechanically verifying candidate Fixer patches (SDD-02, FEAT-14)."""

from __future__ import annotations

import asyncio
import posixpath
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from dasv.adapters.base import EnvironmentAdapter
from dasv.core.config import SandboxConfig
from dasv.core.schemas import EnvSpec
from dasv.sandbox.provider import SandboxProvider
from dasv.verifier.artifact_extractor import ExtractedArtifact
from dasv.verifier.clean_grader import CleanImageGrader, GradingContext
from dasv.verifier.determinism import DeterminismRunner

_DIFF_HEADER_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:[ab]/)?(.+?)(?:\t.*)?$")
_DIFF_GIT_RE = re.compile(r'^diff --git (?:"?a/([^"]+)"?)\s+(?:"?b/([^"]+)"?)$')


class GoldSanityResult(BaseModel):
    """Mechanical 3-invariant verification verdict for a Fixer hardening patch."""

    status: Literal["verified_fix", "rejected_fix"]
    exploit_blocked: bool
    oracle_preserved: bool
    noop_preserved: bool
    exploit_reward_after_fix: float | None = None
    oracle_pass_rate: float | None = None
    noop_pass_rate: float | None = None
    failure_reason: str | None = None


def _is_unsafe_diff_path(raw_path: str) -> bool:
    cleaned = raw_path.strip().strip('"')
    if not cleaned or cleaned == "/dev/null":
        return False
    if cleaned.startswith("/") or cleaned.startswith("\\"):
        return True
    if re.match(r"^[a-zA-Z]:[\\/]", cleaned):
        return True
    normalized = posixpath.normpath(cleaned.replace("\\", "/"))
    return normalized == ".." or normalized.startswith("../")


def validate_safe_unified_diff(diff_text: str) -> list[str]:
    """Validate unified diff syntax and reject absolute paths or `../` directory traversal.

    Returns a list of targeted relative file paths; raises `ValueError` if empty or unsafe.
    """
    stripped = diff_text.strip()
    if not stripped:
        raise ValueError("Candidate diff is empty.")

    touched: list[str] = []
    has_hunk = False
    for line in diff_text.splitlines():
        if line.startswith("@@ "):
            has_hunk = True
            continue
        m_git = _DIFF_GIT_RE.match(line)
        if m_git:
            for grp in (m_git.group(1), m_git.group(2)):
                if _is_unsafe_diff_path(grp):
                    raise ValueError(f"Unsafe path traversal or absolute path in diff: '{grp}'")
                norm = posixpath.normpath(grp.replace("\\", "/"))
                if norm not in touched:
                    touched.append(norm)
            continue
        m_hdr = _DIFF_HEADER_RE.match(line)
        if m_hdr:
            raw_target = m_hdr.group(1).strip().strip('"')
            if raw_target == "/dev/null":
                continue
            if _is_unsafe_diff_path(raw_target):
                raise ValueError(f"Unsafe path traversal or absolute path in diff: '{raw_target}'")
            norm = posixpath.normpath(raw_target.replace("\\", "/"))
            if norm not in touched:
                touched.append(norm)

    if not touched or not has_hunk:
        raise ValueError("Candidate diff must include valid file headers and at least one @@ hunk.")
    return touched


def apply_unified_diff_to_dir(target_dir: Path, diff_text: str) -> tuple[bool, str]:
    """Safely validate and apply a unified diff to `target_dir` using `git apply`."""
    try:
        validate_safe_unified_diff(diff_text)
    except ValueError as exc:
        return False, str(exc)

    resolved_root = target_dir.resolve()
    payload = diff_text if diff_text.endswith("\n") else diff_text + "\n"
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        input=payload.encode("utf-8"),
        cwd=str(resolved_root),
        capture_output=True,
        timeout=15,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        return False, err or f"git apply exited {proc.returncode}"
    return True, ""


class GoldSanityGate:
    """Mechanically validates Fixer patches against 3 mandatory RLVR verifier invariants."""

    def __init__(
        self,
        provider: SandboxProvider,
        *,
        sandbox_config: SandboxConfig | None = None,
    ) -> None:
        self._provider = provider
        self._sandbox_config = sandbox_config or SandboxConfig()

    async def verify_patch(
        self,
        spec: EnvSpec,
        adapter: EnvironmentAdapter,
        exploit_artifact: ExtractedArtifact,
        candidate_diff: str,
        *,
        run_id: str = "dasv-gsg",
        oracle_replays: int = 5,
        noop_replays: int = 2,
        semaphore: asyncio.Semaphore | None = None,
    ) -> GoldSanityResult:
        """Apply `candidate_diff` to a clean temp copy of `spec` and verify all 3 invariants."""
        try:
            validate_safe_unified_diff(candidate_diff)
        except ValueError as exc:
            return GoldSanityResult(
                status="rejected_fix",
                exploit_blocked=False,
                oracle_preserved=False,
                noop_preserved=False,
                failure_reason=f"Invalid or unsafe diff: {exc}",
            )

        with tempfile.TemporaryDirectory(prefix="dasv_gsg_") as tmp_dir:
            patched_root = Path(tmp_dir) / spec.root_path.name
            shutil.copytree(spec.root_path, patched_root)

            applied, apply_err = apply_unified_diff_to_dir(patched_root, candidate_diff)
            if not applied:
                return GoldSanityResult(
                    status="rejected_fix",
                    exploit_blocked=False,
                    oracle_preserved=False,
                    noop_preserved=False,
                    failure_reason=f"Candidate diff failed to apply cleanly: {apply_err}",
                )

            patched_spec = adapter.load(patched_root)
            # Preserve container image digest/ref if original spec had resolved digest metadata.
            updates: dict[str, object] = {}
            if not patched_spec.image_ref and spec.image_ref:
                updates["image_ref"] = spec.image_ref
            if not patched_spec.image_digest and spec.image_digest:
                updates["image_digest"] = spec.image_digest
            if updates:
                patched_spec = patched_spec.model_copy(update=updates)

            grader = CleanImageGrader(self._provider, sandbox_config=self._sandbox_config)

            # Invariant 1: Confirmed exploit artifact must now score reward == 0.0.
            exploit_outcome = await grader.grade(
                patched_spec,
                exploit_artifact,
                f"{run_id}-exploit",
            )
            exploit_blocked = exploit_outcome.reward <= 0.0

            # Invariant 2 & 3: Canonical Oracle == 100% and No-op == 0% with 0 variance.
            grading_ctx = GradingContext(
                grader,
                run_id=f"{run_id}-det",
                timeout_sec=float(self._sandbox_config.episode_timeout_sec),
            )
            det_runner = DeterminismRunner(max_concurrent=self._sandbox_config.max_concurrent)
            det_result = await det_runner.run(
                adapter,
                patched_spec,
                grading_ctx,
                oracle_trials=oracle_replays if patched_spec.has_oracle else 0,
                noop_trials=noop_replays,
                semaphore=semaphore,
            )
            metrics = det_result.metrics

            oracle_preserved = (
                True
                if not patched_spec.has_oracle
                else (metrics.oracle_pass_rate >= 1.0 - 1e-9 and metrics.replay_variance <= 0.0)
            )
            noop_preserved = metrics.noop_pass_rate <= 0.0 and metrics.replay_variance <= 0.0

            reasons: list[str] = []
            if not exploit_blocked:
                reasons.append(
                    f"Exploit still passed after fix (reward={exploit_outcome.reward:.2f})"
                )
            if not oracle_preserved:
                reasons.append(
                    f"Oracle regressed (pass_rate={metrics.oracle_pass_rate:.3f}, "
                    f"variance={metrics.replay_variance:.6f})"
                )
            if not noop_preserved:
                reasons.append(f"No-op leaked after fix (pass_rate={metrics.noop_pass_rate:.3f})")

            all_ok = exploit_blocked and oracle_preserved and noop_preserved
            return GoldSanityResult(
                status="verified_fix" if all_ok else "rejected_fix",
                exploit_blocked=exploit_blocked,
                oracle_preserved=oracle_preserved,
                noop_preserved=noop_preserved,
                exploit_reward_after_fix=exploit_outcome.reward,
                oracle_pass_rate=metrics.oracle_pass_rate if patched_spec.has_oracle else None,
                noop_pass_rate=metrics.noop_pass_rate,
                failure_reason="; ".join(reasons) if reasons else None,
            )
