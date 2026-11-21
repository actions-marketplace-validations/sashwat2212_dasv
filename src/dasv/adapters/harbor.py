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
from dasv.verifier.clean_grader import coerce_grading_context, outcome_to_details

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

        explicit_image = (
            str(env_tbl["image"]).strip()
            if isinstance(env_tbl.get("image"), str) and env_tbl["image"].strip()
            else None
        )
        docker_image = (
            str(env_tbl["docker_image"]).strip()
            if isinstance(env_tbl.get("docker_image"), str) and env_tbl["docker_image"].strip()
            else None
        )
        image_ref: str | None = explicit_image
        if image_ref is None:
            image_ref = _extract_first_from_image(dockerfile)
        if image_ref is None:
            image_ref = docker_image

        image_digest: str | None = None
        if image_ref and "@sha256:" in image_ref:
            image_digest = "sha256:" + image_ref.split("@sha256:", 1)[1]

        dockerfile_rel: str | None = None
        if dockerfile.is_file():
            dockerfile_rel = dockerfile.relative_to(path).as_posix()

        raw_allow_internet = env_tbl.get("allow_internet")
        allow_internet: bool | None = (
            raw_allow_internet if isinstance(raw_allow_internet, bool) else None
        )
        raw_network_mode = env_tbl.get("network_mode")
        if raw_network_mode is None and allow_internet is not None:
            resolved_network_mode: Any = "no-network" if not allow_internet else "allow-internet"
        else:
            resolved_network_mode = raw_network_mode

        spec_metadata: dict[str, Any] = {
            "harbor_config": raw,
            "network_mode": resolved_network_mode,
            "allow_internet": allow_internet,
            "docker_image": docker_image,
            "base_commit": metadata_tbl.get("base_commit") or env_tbl.get("base_commit"),
            "image_explicit": explicit_image is not None,
            "dockerfile_path": dockerfile_rel,
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
        if not spec.has_oracle or not spec.oracle_entrypoint:
            raise RuntimeSandboxError(
                f"Harbor task '{spec.task_id}' does not declare an existing oracle_entrypoint."
            )
        oracle_path = (spec.root_path / spec.oracle_entrypoint).resolve()
        try:
            oracle_path.relative_to(spec.root_path.resolve())
        except ValueError as exc:
            raise RuntimeSandboxError(
                f"Oracle entrypoint '{spec.oracle_entrypoint}' escapes task root."
            ) from exc
        if not oracle_path.is_file():
            raise RuntimeSandboxError(f"Oracle script not found: {oracle_path}")

        ctx = coerce_grading_context(sandbox)
        outcome, artifact = await ctx.grader.execute_and_grade(
            spec,
            ctx.run_id,
            script_bytes=oracle_path.read_bytes(),
            timeout_sec=ctx.timeout_sec,
            workdir=ctx.workdir,
        )
        return outcome.reward, outcome_to_details(outcome, artifact)

    async def run_noop(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        ctx = coerce_grading_context(sandbox)
        outcome, artifact = await ctx.grader.execute_and_grade(
            spec,
            ctx.run_id,
            agent_cmd=["true"],
            timeout_sec=ctx.timeout_sec,
            workdir=ctx.workdir,
        )
        return outcome.reward, outcome_to_details(outcome, artifact)
