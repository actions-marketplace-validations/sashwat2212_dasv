"""OpenEnv (HTTP/WS Gymnasium container contract) environment adapter (FEAT-02, SDD-03)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, cast

import yaml

from dasv.adapters.base import AdapterCapabilities
from dasv.adapters.native import NativeDelegateRunner
from dasv.core.errors import ConfigError, RuntimeSandboxError
from dasv.core.schemas import AuditFinding, EnvSpec, EnvStandard
from dasv.verifier.clean_grader import coerce_grading_context, outcome_to_details

_FROM_RE = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)", re.IGNORECASE)


def _find_openenv_dockerfile(path: Path) -> Path | None:
    for candidate in (
        path / "server" / "Dockerfile",
        path / "environment" / "Dockerfile",
        path / "Dockerfile",
    ):
        if candidate.is_file():
            return candidate
    return None


def _extract_first_from(dockerfile: Path | None) -> str | None:
    if dockerfile is None or not dockerfile.is_file():
        return None
    for line in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _FROM_RE.match(line)
        if match:
            return match.group(1)
    return None


def _find_manifest(path: Path) -> Path | None:
    for name in ("openenv.yaml", "openenv.yml", "openenv.toml"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def _pyproject_mentions_openenv(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        data: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    tool_tbl = data.get("tool")
    if isinstance(tool_tbl, dict) and "openenv" in tool_tbl:
        return True
    project_tbl = data.get("project")
    if isinstance(project_tbl, dict):
        project_map = cast(dict[str, Any], project_tbl)
        deps_raw = project_map.get("dependencies", [])
        if isinstance(deps_raw, list):
            deps = cast(list[Any], deps_raw)
            if any(isinstance(d, str) and d.lower().startswith("openenv") for d in deps):
                return True
    return False


class OpenEnvAdapter:
    standard: EnvStandard = EnvStandard.OPENENV
    capabilities: AdapterCapabilities = AdapterCapabilities(
        supports_clean_image_grading=True,
        supports_interactive_reset=True,
        supports_rubric_introspection=False,
        native_cli_binary="openenv",
    )

    def __init__(self, native_runner: NativeDelegateRunner | None = None) -> None:
        self._native = native_runner or NativeDelegateRunner()

    def detect(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        if _find_manifest(path) is not None:
            return True
        return _pyproject_mentions_openenv(path)

    def load(self, path: Path) -> EnvSpec:
        if not self.detect(path):
            raise ConfigError(
                f"Path '{path}' is not a valid OpenEnv environment "
                "(expected openenv.yaml/toml or openenv pyproject.toml)."
            )

        raw: dict[str, Any] = {}
        manifest = _find_manifest(path)
        if manifest is not None:
            try:
                text = manifest.read_text(encoding="utf-8")
                if manifest.suffix == ".toml":
                    loaded: Any = tomllib.loads(text)
                else:
                    loaded = yaml.safe_load(text) or {}
                if not isinstance(loaded, dict):
                    raise ConfigError(f"OpenEnv manifest '{manifest}' must be a mapping.")
                raw = dict(cast(dict[str, Any], loaded))
            except (OSError, tomllib.TOMLDecodeError, yaml.YAMLError) as exc:
                raise ConfigError(f"Failed to parse OpenEnv manifest '{manifest}': {exc}") from exc

        task_id = str(raw.get("name") or raw.get("env_id") or path.resolve().name)

        instruction_text = ""
        if isinstance(raw.get("description"), str):
            instruction_text = raw["description"]
        elif (path / "instruction.md").is_file():
            instruction_text = (path / "instruction.md").read_text(encoding="utf-8")
        elif (path / "README.md").is_file():
            instruction_text = (path / "README.md").read_text(encoding="utf-8")

        dockerfile = _find_openenv_dockerfile(path)
        explicit_image = (
            str(raw["image"]).strip()
            if isinstance(raw.get("image"), str) and raw["image"].strip()
            else None
        )
        image_ref: str | None = explicit_image or _extract_first_from(dockerfile)
        image_digest: str | None = None
        if image_ref and "@sha256:" in image_ref:
            image_digest = "sha256:" + image_ref.split("@sha256:", 1)[1]
        dockerfile_rel = dockerfile.relative_to(path).as_posix() if dockerfile is not None else None

        verifier_paths: list[str] = []
        if (path / "tests").exists():
            verifier_paths.append("tests/")
        if isinstance(raw.get("verifier_paths"), list):
            verifier_paths.extend(str(p) for p in raw["verifier_paths"])

        env_section = raw.get("environment")
        network_mode: Any = raw.get("network_mode")
        if network_mode is None and isinstance(env_section, dict):
            network_mode = cast(dict[str, Any], env_section).get("network_mode")

        oracle_entrypoint = (
            str(raw["oracle_entrypoint"]) if isinstance(raw.get("oracle_entrypoint"), str) else None
        )
        has_oracle = oracle_entrypoint is not None and (path / oracle_entrypoint).exists()

        return EnvSpec(
            task_id=task_id,
            standard=EnvStandard.OPENENV,
            root_path=path.resolve(),
            instruction_text=instruction_text,
            image_ref=image_ref,
            image_digest=image_digest,
            has_oracle=has_oracle,
            oracle_entrypoint=oracle_entrypoint,
            verifier_paths=verifier_paths,
            rubric_type="deterministic",
            metadata={
                "openenv_manifest": raw,
                "network_mode": network_mode,
                "supports_interactive_reset": True,
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
            "openenv",
            ["validate", "--level", "static", "--skip-build", str(spec.root_path)],
            cwd=spec.root_path,
            timeout_sec=timeout_sec,
            require_native=require_native,
        )
        return result.findings

    async def run_oracle(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        if not spec.has_oracle or not spec.oracle_entrypoint:
            raise RuntimeSandboxError(
                f"OpenEnv task '{spec.task_id}' does not declare an existing oracle_entrypoint."
            )
        oracle_path = (spec.root_path / spec.oracle_entrypoint).resolve()
        try:
            oracle_path.relative_to(spec.root_path.resolve())
        except ValueError as exc:
            raise RuntimeSandboxError(
                f"Oracle entrypoint '{spec.oracle_entrypoint}' escapes environment root."
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
