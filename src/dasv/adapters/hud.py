"""HUD (`hud-python` MCP tasks & `tasks.grade`) plugin adapter (FEAT-04, SDD-03)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

from dasv.adapters.base import AdapterCapabilities
from dasv.adapters.native import NativeDelegateRunner
from dasv.core.errors import ConfigError, RuntimeSandboxError
from dasv.core.schemas import AuditFinding, EnvSpec, EnvStandard
from dasv.verifier.clean_grader import coerce_grading_context, outcome_to_details


def _pyproject_mentions_hud(path: Path) -> tuple[bool, dict[str, Any]]:
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False, {}
    try:
        data: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False, {}
    tool_tbl = data.get("tool")
    if isinstance(tool_tbl, dict) and "hud" in tool_tbl:
        return True, data
    project_tbl = data.get("project")
    if isinstance(project_tbl, dict):
        project_map = cast(dict[str, Any], project_tbl)
        deps_raw = project_map.get("dependencies", [])
        if isinstance(deps_raw, list):
            deps = cast(list[Any], deps_raw)
            if any(
                isinstance(d, str) and d.lower().startswith(("hud-python", "hud_python"))
                for d in deps
            ):
                return True, data
    return False, data


class HudAdapter:
    standard: EnvStandard = EnvStandard.HUD
    capabilities: AdapterCapabilities = AdapterCapabilities(
        supports_clean_image_grading=True,
        supports_interactive_reset=True,
        supports_rubric_introspection=False,
        native_cli_binary="hud",
    )

    def __init__(self, native_runner: NativeDelegateRunner | None = None) -> None:
        self._native = native_runner or NativeDelegateRunner()

    def detect(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        if (path / "hud.toml").is_file():
            return True
        mentions_hud, _ = _pyproject_mentions_hud(path)
        return mentions_hud

    def load(self, path: Path) -> EnvSpec:
        if not self.detect(path):
            raise ConfigError(
                f"Path '{path}' is not a valid HUD environment (expected hud.toml or hud-python)."
            )

        raw: dict[str, Any] = {}
        hud_toml = path / "hud.toml"
        if hud_toml.is_file():
            try:
                raw = tomllib.loads(hud_toml.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise ConfigError(f"Failed to parse HUD config '{hud_toml}': {exc}") from exc

        _, pyproject = _pyproject_mentions_hud(path)
        project_tbl: dict[str, Any] = (
            pyproject.get("project", {}) if isinstance(pyproject.get("project"), dict) else {}
        )
        task_id = str(raw.get("name") or project_tbl.get("name") or path.resolve().name)
        instruction_text = (
            (path / "README.md").read_text(encoding="utf-8")
            if (path / "README.md").is_file()
            else str(raw.get("description") or "")
        )
        verifier_paths = ["tasks/grade.py"] if (path / "tasks" / "grade.py").is_file() else []
        if (path / "tests").is_dir() and "tests/" not in verifier_paths:
            verifier_paths.append("tests/")
        explicit_image = (
            str(raw["image"]).strip()
            if isinstance(raw.get("image"), str) and raw["image"].strip()
            else None
        )
        image_digest: str | None = None
        if explicit_image and "@sha256:" in explicit_image:
            image_digest = "sha256:" + explicit_image.split("@sha256:", 1)[1]

        dockerfile = path / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            dockerfile = path / "Dockerfile"
        dockerfile_rel = dockerfile.relative_to(path).as_posix() if dockerfile.is_file() else None

        oracle_entrypoint: str | None = None
        configured_oracle = raw.get("oracle_entrypoint")
        if isinstance(configured_oracle, str) and configured_oracle.strip():
            oracle_entrypoint = configured_oracle.strip()
        elif (path / "solution" / "solve.sh").is_file():
            oracle_entrypoint = "solution/solve.sh"
        has_oracle = oracle_entrypoint is not None and (path / oracle_entrypoint).exists()

        return EnvSpec(
            task_id=task_id,
            standard=EnvStandard.HUD,
            root_path=path.resolve(),
            instruction_text=instruction_text,
            image_ref=explicit_image,
            image_digest=image_digest,
            has_oracle=has_oracle,
            oracle_entrypoint=oracle_entrypoint,
            verifier_paths=verifier_paths,
            rubric_type="deterministic",
            metadata={
                "hud_config": raw,
                "network_mode": raw.get("network_mode"),
                "image_explicit": explicit_image is not None,
                "dockerfile_path": dockerfile_rel,
            },
        )

    def static_findings(self, spec: EnvSpec) -> list[AuditFinding]:
        return []

    async def run_native_check(
        self,
        spec: EnvSpec,
        timeout_sec: float = 60.0,
        require_native: bool = False,
    ) -> list[AuditFinding]:
        result = await self._native.run(
            "hud",
            ["validate", str(spec.root_path)],
            cwd=spec.root_path,
            timeout_sec=timeout_sec,
            require_native=require_native,
        )
        return result.findings

    async def run_oracle(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        if not spec.has_oracle or not spec.oracle_entrypoint:
            raise RuntimeSandboxError(
                f"HUD task '{spec.task_id}' does not declare an existing oracle_entrypoint "
                "(set oracle_entrypoint in hud.toml or provide solution/solve.sh)."
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
