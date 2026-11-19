"""Adapter capability metadata and `EnvironmentAdapter` protocol (SDD-03)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from dasv.core.schemas import AuditFinding, EnvSpec, EnvStandard


class AdapterCapabilities(BaseModel):
    supports_clean_image_grading: bool = True
    supports_interactive_reset: bool = False
    supports_rubric_introspection: bool = False
    native_cli_binary: str | None = None


class EnvironmentAdapter(Protocol):
    standard: EnvStandard
    capabilities: AdapterCapabilities

    def detect(self, path: Path) -> bool:
        """Fast (< 10 ms) deterministic detection; never executes untrusted code."""
        ...

    def load(self, path: Path) -> EnvSpec:
        """Parse manifest/config into canonical EnvSpec with zero side effects."""
        ...

    def static_findings(self, spec: EnvSpec) -> list[AuditFinding]:
        """Return adapter-specific Tier 0 static findings (e.g., legacy v0 deprecation)."""
        ...

    async def run_native_check(
        self,
        spec: EnvSpec,
        timeout_sec: float = 60.0,
        require_native: bool = False,
    ) -> list[AuditFinding]:
        """Delegate baseline checks to the native CLI if installed."""
        ...

    async def run_oracle(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        """Execute canonical Oracle solution (Stage 6 Tier 1 runtime)."""
        ...

    async def run_noop(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        """Execute canonical No-op baseline (Stage 6 Tier 1 runtime)."""
        ...
