"""Native validator subprocess delegator with scrubbed env & graceful degradation (SDD-03)."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

from pydantic import BaseModel, Field

from dasv.core.errors import NativeDelegateError
from dasv.core.schemas import AuditFinding, Severity

_SAFE_ENV_KEYS = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "UV_CACHE_DIR")
_MAX_OUTPUT_CHARS = 4096


def _scrubbed_env() -> dict[str, str]:
    return {key: os.environ[key] for key in _SAFE_ENV_KEYS if key in os.environ}


class NativeDelegateResult(BaseModel):
    binary: str
    found: bool
    exit_code: int | None = None
    duration_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""
    findings: list[AuditFinding] = Field(default_factory=list[AuditFinding])


class NativeDelegateRunner:
    """Safely executes optional native CLIs (`harbor`, `openenv`, `vf-validate`, `hud`)."""

    async def run(
        self,
        binary: str,
        args: list[str],
        cwd: Path,
        *,
        timeout_sec: float = 60.0,
        require_native: bool = False,
    ) -> NativeDelegateResult:
        resolved_bin = shutil.which(binary)
        if resolved_bin is None:
            if require_native:
                raise NativeDelegateError(
                    f"Required native validator binary '{binary}' was not found on PATH."
                )
            advisory = AuditFinding(
                rule_id="RULE-NATIVE-001",
                title=f"Native CLI '{binary}' not installed",
                severity=Severity.INFO,
                tier=0,
                source="native_delegate",
                file_path=None,
                line_number=None,
                evidence=f"Executable '{binary}' not found on PATH; ran dasv fallback checks only.",
                remediation=f"Install '{binary}' to enable delegated upstream schema validation.",
            )
            return NativeDelegateResult(binary=binary, found=False, findings=[advisory])

        start = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            resolved_bin,
            *args,
            cwd=str(cwd),
            env=_scrubbed_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            finding = AuditFinding(
                rule_id="RULE-NATIVE-002",
                title=f"Native validator '{binary}' timed out",
                severity=Severity.HIGH,
                tier=0,
                source="native_delegate",
                file_path=None,
                line_number=None,
                evidence=(
                    f"Command '{binary} {' '.join(args)}' exceeded {timeout_sec:.1f}s timeout."
                ),
                remediation=f"Run '{binary} {' '.join(args)}' manually to diagnose hangs.",
            )
            if require_native:
                raise NativeDelegateError(finding.evidence) from None
            return NativeDelegateResult(
                binary=binary,
                found=True,
                exit_code=None,
                duration_ms=elapsed_ms,
                findings=[finding],
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stdout = raw_out.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
        stderr = raw_err.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
        exit_code = proc.returncode if proc.returncode is not None else -1

        findings: list[AuditFinding] = []
        if exit_code != 0:
            snippet = (stderr or stdout or f"Exited with status {exit_code}").strip()
            finding = AuditFinding(
                rule_id="RULE-NATIVE-003",
                title=f"Native validator '{binary}' reported failure (exit {exit_code})",
                severity=Severity.HIGH,
                tier=0,
                source="native_delegate",
                file_path=None,
                line_number=None,
                evidence=snippet[:512],
                remediation=f"Fix issues reported by '{binary} {' '.join(args)}'.",
            )
            if require_native:
                raise NativeDelegateError(
                    f"Native validator '{binary}' failed (exit {exit_code}): {snippet[:256]}"
                )
            findings.append(finding)

        return NativeDelegateResult(
            binary=binary,
            found=True,
            exit_code=exit_code,
            duration_ms=elapsed_ms,
            stdout=stdout,
            stderr=stderr,
            findings=findings,
        )
