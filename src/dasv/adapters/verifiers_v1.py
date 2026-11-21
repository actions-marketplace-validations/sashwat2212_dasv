"""Prime Intellect `verifiers` v1 adapter with AST introspection & v0 guard (FEAT-03)."""

from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from dasv.adapters.base import AdapterCapabilities
from dasv.adapters.native import NativeDelegateRunner
from dasv.core.errors import ConfigError, RuntimeSandboxError
from dasv.core.schemas import AuditFinding, EnvSpec, EnvStandard, Severity
from dasv.verifier.clean_grader import coerce_grading_context, outcome_to_details

_LEGACY_V0_SYMBOLS = {"SingleTurnEnv", "MultiTurnEnv", "ToolEnv", "CodeEnv"}
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "build", "dist"}


@dataclass
class _VerifiersAstSummary:
    imports_verifiers: bool = False
    has_load_environment: bool = False
    has_v1_taskset: bool = False
    has_legacy_v0: bool = False
    legacy_v0_files: list[str] = field(default_factory=list[str])
    rubric_classes: set[str] = field(default_factory=set[str])
    reward_methods: list[str] = field(default_factory=list[str])
    verifier_files: list[str] = field(default_factory=list[str])


def _attr_or_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript):
        return _attr_or_name(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_reward_decorator(dec: ast.expr) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    return _attr_or_name(target) == "reward"


def _introspect_ast(root: Path) -> _VerifiersAstSummary:
    summary = _VerifiersAstSummary()
    for py_file in sorted(root.rglob("*.py")):
        rel = py_file.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue

        file_has_legacy = False
        file_has_rubric = False

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "load_environment":
                    summary.has_load_environment = True
                if any(_is_reward_decorator(dec) for dec in node.decorator_list):
                    summary.reward_methods.append(node.name)
                    summary.rubric_classes.add("Rubric")
                    file_has_rubric = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "verifiers" or alias.name.startswith("verifiers."):
                        summary.imports_verifiers = True
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "verifiers" or mod.startswith("verifiers."):
                    summary.imports_verifiers = True
                if mod.startswith("verifiers.envs"):
                    file_has_legacy = True
                for alias in node.names:
                    if alias.name in {"Taskset", "Task", "TasksetConfig"}:
                        summary.has_v1_taskset = True
                    if alias.name in _LEGACY_V0_SYMBOLS:
                        file_has_legacy = True
                    if alias.name in {"Rubric", "JudgeRubric", "RubricGroup"}:
                        summary.rubric_classes.add(alias.name)
                        file_has_rubric = True
            elif isinstance(node, ast.Call):
                callee = _attr_or_name(node.func)
                if callee == "Taskset":
                    summary.has_v1_taskset = True
                elif callee in _LEGACY_V0_SYMBOLS:
                    file_has_legacy = True
                elif callee in {"Rubric", "JudgeRubric", "RubricGroup"}:
                    summary.rubric_classes.add(callee)
                    file_has_rubric = True
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = _attr_or_name(base)
                    if base_name in {"Taskset", "Task", "TasksetConfig"}:
                        summary.has_v1_taskset = True
                    if base_name in _LEGACY_V0_SYMBOLS:
                        file_has_legacy = True

        if file_has_legacy:
            summary.has_legacy_v0 = True
            summary.legacy_v0_files.append(rel.as_posix())
        if file_has_rubric:
            summary.verifier_files.append(rel.as_posix())

    return summary


def _pyproject_mentions_verifiers(path: Path) -> tuple[bool, dict[str, Any]]:
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False, {}
    try:
        data: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False, {}

    tool_tbl = data.get("tool")
    if isinstance(tool_tbl, dict) and "verifiers" in tool_tbl:
        return True, data

    project_tbl = data.get("project")
    if isinstance(project_tbl, dict):
        project_dict = cast(dict[str, Any], project_tbl)
        deps_raw: object = project_dict.get("dependencies", [])
        if isinstance(deps_raw, list):
            deps = cast(list[object], deps_raw)
            if any(
                isinstance(d, str)
                and (d.lower().startswith("verifiers") or "verifiers==" in d.lower())
                for d in deps
            ):
                return True, data
    return False, data


class VerifiersV1Adapter:
    standard: EnvStandard = EnvStandard.VERIFIERS_V1
    capabilities: AdapterCapabilities = AdapterCapabilities(
        supports_clean_image_grading=True,
        supports_interactive_reset=False,
        supports_rubric_introspection=True,
        native_cli_binary="vf-validate",
    )

    def __init__(self, native_runner: NativeDelegateRunner | None = None) -> None:
        self._native = native_runner or NativeDelegateRunner()

    def detect(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        if not (path / "pyproject.toml").is_file():
            return False
        has_dep, _ = _pyproject_mentions_verifiers(path)
        summary = _introspect_ast(path)
        if not (has_dep or summary.imports_verifiers):
            return False
        return summary.has_load_environment or summary.has_v1_taskset or summary.has_legacy_v0

    def load(self, path: Path) -> EnvSpec:
        if not self.detect(path):
            raise ConfigError(
                f"Path '{path}' is not a valid Prime Intellect verifiers package "
                "(expected pyproject.toml with verifiers dependency and load_environment/Taskset)."
            )

        _, pyproject = _pyproject_mentions_verifiers(path)
        project_tbl: dict[str, Any] = (
            pyproject.get("project", {}) if isinstance(pyproject.get("project"), dict) else {}
        )
        task_id = str(project_tbl.get("name") or path.resolve().name)

        instruction_text = ""
        if (path / "README.md").is_file():
            instruction_text = (path / "README.md").read_text(encoding="utf-8")
        elif isinstance(project_tbl.get("description"), str):
            instruction_text = project_tbl["description"]

        summary = _introspect_ast(path)
        has_judge = "JudgeRubric" in summary.rubric_classes
        has_prog = "Rubric" in summary.rubric_classes or "RubricGroup" in summary.rubric_classes
        rubric_type: Literal["deterministic", "rubric_llm", "hybrid"]
        if has_judge and has_prog:
            rubric_type = "hybrid"
        elif has_judge:
            rubric_type = "rubric_llm"
        else:
            rubric_type = "deterministic"

        legacy_v0_only = summary.has_legacy_v0 and not summary.has_v1_taskset

        tool_tbl = pyproject.get("tool")
        vf_raw = (
            cast(dict[str, Any], tool_tbl).get("verifiers") if isinstance(tool_tbl, dict) else None
        )
        vf_tbl: dict[str, Any] = cast(dict[str, Any], vf_raw) if isinstance(vf_raw, dict) else {}
        explicit_image = (
            str(vf_tbl["image"]).strip()
            if isinstance(vf_tbl.get("image"), str) and vf_tbl["image"].strip()
            else None
        )
        dockerfile = path / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            dockerfile = path / "Dockerfile"
        dockerfile_rel = dockerfile.relative_to(path).as_posix() if dockerfile.is_file() else None
        image_digest: str | None = None
        if explicit_image and "@sha256:" in explicit_image:
            image_digest = "sha256:" + explicit_image.split("@sha256:", 1)[1]

        oracle_entrypoint: str | None = None
        configured_oracle = vf_tbl.get("oracle_entrypoint")
        if isinstance(configured_oracle, str) and configured_oracle.strip():
            oracle_entrypoint = configured_oracle.strip()
        elif (path / "solution" / "solve.sh").is_file():
            oracle_entrypoint = "solution/solve.sh"
        has_oracle = oracle_entrypoint is not None and (path / oracle_entrypoint).exists()

        return EnvSpec(
            task_id=task_id,
            standard=EnvStandard.VERIFIERS_V1,
            root_path=path.resolve(),
            instruction_text=instruction_text,
            image_ref=explicit_image,
            image_digest=image_digest,
            has_oracle=has_oracle,
            oracle_entrypoint=oracle_entrypoint,
            verifier_paths=summary.verifier_files,
            rubric_type=rubric_type,
            metadata={
                "has_load_environment": summary.has_load_environment,
                "has_v1_taskset": summary.has_v1_taskset,
                "legacy_v0_detected": legacy_v0_only,
                "legacy_v0_files": summary.legacy_v0_files,
                "rubric_classes": sorted(summary.rubric_classes),
                "image_explicit": explicit_image is not None,
                "dockerfile_path": dockerfile_rel,
            },
        )

    def static_findings(self, spec: EnvSpec) -> list[AuditFinding]:
        if not spec.metadata.get("legacy_v0_detected"):
            return []
        raw_files: object = spec.metadata.get("legacy_v0_files")
        files: list[str] = (
            [str(f) for f in cast(list[object], raw_files)] if isinstance(raw_files, list) else []
        )
        first_file = files[0] if files else None
        return [
            AuditFinding(
                rule_id="RULE-ADPT-001",
                title="Deprecated legacy verifiers v0 environment detected",
                severity=Severity.HIGH,
                tier=0,
                source="dasv",
                file_path=first_file,
                line_number=1 if first_file else None,
                evidence=(
                    "Detected legacy verifiers v0 symbols "
                    f"without v1 Taskset in: {', '.join(files) or spec.task_id}."
                ),
                remediation=(
                    "Migrate the package to Prime Intellect verifiers v1 using vf.Taskset, "
                    "RubricGroup, and load_environment()."
                ),
            )
        ]

    async def run_native_check(
        self,
        spec: EnvSpec,
        timeout_sec: float = 60.0,
        require_native: bool = False,
    ) -> list[AuditFinding]:
        result = await self._native.run(
            "vf-validate",
            [str(spec.root_path), "-n", "1"],
            cwd=spec.root_path,
            timeout_sec=timeout_sec,
            require_native=require_native,
        )
        return result.findings

    async def run_oracle(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        if not spec.has_oracle or not spec.oracle_entrypoint:
            raise RuntimeSandboxError(
                f"verifiers v1 task '{spec.task_id}' does not declare an existing "
                "oracle_entrypoint (set [tool.verifiers].oracle_entrypoint or solution/solve.sh)."
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
