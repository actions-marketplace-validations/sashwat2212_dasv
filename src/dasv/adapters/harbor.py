"""Harbor (`task.toml` / `instruction.md`) environment adapter (FEAT-01, SDD-03)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from dasv.adapters.base import AdapterCapabilities
from dasv.adapters.native import NativeDelegateRunner
from dasv.core.errors import ConfigError, RuntimeSandboxError
from dasv.core.schemas import AuditFinding, EnvSpec, EnvStandard

_FROM_RE = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)", re.IGNORECASE)


def _extract_first_from_image(dockerfile: Path) -> str | None:
    if not dockerfile.is_file():
        return None
    for line in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _FROM_RE.match(line)
        if match:
            return match.group(1)
    return None


class HarborAdapter:
    standard: EnvStandard = EnvStandard.HARBOR
    capabilities: AdapterCapabilities = AdapterCapabilities(
        supports_clean_image_grading=True,
        supports_interactive_reset=False,
        supports_rubric_introspection=False,
        native_cli_binary="harbor",
    )

    def __init__(self, native_runner: NativeDelegateRunner | None = None) -> None:
        self._native = native_runner or NativeDelegateRunner()

    def detect(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        return (path / "task.toml").is_file() and (path / "instruction.md").is_file()

    def load(self, path: Path) -> EnvSpec:
        if not self.detect(path):
            raise ConfigError(
                f"Path '{path}' is not a valid Harbor task directory "
                "(requires task.toml and instruction.md)."
            )

        task_toml_path = path / "task.toml"
        try:
            raw: dict[str, Any] = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"Malformed Harbor task.toml at '{task_toml_path}': {exc}") from exc

        instruction_text = (path / "instruction.md").read_text(encoding="utf-8")

        metadata_tbl: dict[str, Any] = (
            raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}
        )
        env_tbl: dict[str, Any] = (
            raw.get("environment", {}) if isinstance(raw.get("environment"), dict) else {}
        )
        verifier_tbl: dict[str, Any] = (
            raw.get("verifier", {}) if isinstance(raw.get("verifier"), dict) else {}
        )
        solution_tbl: dict[str, Any] = (
            raw.get("solution", {}) if isinstance(raw.get("solution"), dict) else {}
        )

        task_id = str(metadata_tbl.get("name") or path.resolve().name)

        oracle_entrypoint: str | None = None
        configured_oracle = solution_tbl.get("entrypoint")
        if isinstance(configured_oracle, str) and configured_oracle.strip():
            oracle_entrypoint = configured_oracle.strip()
        elif (path / "solution" / "solve.sh").is_file():
            oracle_entrypoint = "solution/solve.sh"
        has_oracle = oracle_entrypoint is not None and (path / oracle_entrypoint).exists()

        verifier_paths: list[str] = ["tests/", "tests/test.sh"]
        extra_verifier = verifier_tbl.get("test_path") or verifier_tbl.get("entrypoint")
        if isinstance(extra_verifier, str) and extra_verifier not in verifier_paths:
            verifier_paths.append(extra_verifier)

        dockerfile = path / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            dockerfile = path / "Dockerfile"

        image_ref: str | None = (
            str(env_tbl["image"]) if isinstance(env_tbl.get("image"), str) else None
        )
        if image_ref is None:
            image_ref = _extract_first_from_image(dockerfile)

        image_digest: str | None = None
        if image_ref and "@sha256:" in image_ref:
            image_digest = "sha256:" + image_ref.split("@sha256:", 1)[1]

        spec_metadata: dict[str, Any] = {
            "harbor_config": raw,
            "network_mode": env_tbl.get("network_mode"),
            "base_commit": metadata_tbl.get("base_commit") or env_tbl.get("base_commit"),
        }

        return EnvSpec(
            task_id=task_id,
            standard=EnvStandard.HARBOR,
            root_path=path.resolve(),
            instruction_text=instruction_text,
            image_ref=image_ref,
            image_digest=image_digest,
            has_oracle=has_oracle,
            oracle_entrypoint=oracle_entrypoint,
            verifier_paths=verifier_paths,
            rubric_type="deterministic",
            metadata=spec_metadata,
        )

    def static_findings(self, spec: EnvSpec) -> list[AuditFinding]:
        return []

    async def run_native_check(
        self,
        spec: EnvSpec,
        timeout_sec: float = 60.0,
        require_native: bool = False,
    ) -> list[AuditFinding]:
        # Explicitly use modern `harbor check <task-dir>` (never deprecated `harbor tasks check`)
        result = await self._native.run(
            "harbor",
            ["check", str(spec.root_path)],
            cwd=spec.root_path,
            timeout_sec=timeout_sec,
            require_native=require_native,
        )
        return result.findings

    async def run_oracle(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        raise RuntimeSandboxError(
            "Harbor Tier 1 container oracle execution is scheduled for Stage 6 M1."
        )

    async def run_noop(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        raise RuntimeSandboxError(
            "Harbor Tier 1 container no-op execution is scheduled for Stage 6 M1."
        )
