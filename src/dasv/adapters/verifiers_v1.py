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

_LEGACY_V0_SYMBOLS = {"SingleTurnEnv", "MultiTurnEnv", "ToolEnv", "CodeEnv"}
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "build", "dist"}


@dataclass
class _VerifiersAstSummary:
    has_load_environment: bool = False
    has_v1_taskset: bool = False
    has_legacy_v0: bool = False
    legacy_v0_files: list[str] = field(default_factory=list[str])
    rubric_classes: set[str] = field(default_factory=set[str])
    verifier_files: list[str] = field(default_factory=list[str])


def _attr_or_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


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
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name == "load_environment"
            ):
                summary.has_load_environment = True
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith("verifiers.envs"):
                    file_has_legacy = True
                for alias in node.names:
                    if alias.name == "Taskset":
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
        has_dep, _ = _pyproject_mentions_verifiers(path)
        if not has_dep:
            return False
        summary = _introspect_ast(path)
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

        return EnvSpec(
            task_id=task_id,
            standard=EnvStandard.VERIFIERS_V1,
            root_path=path.resolve(),
            instruction_text=instruction_text,
            image_ref=None,
            image_digest=None,
            has_oracle=False,
            oracle_entrypoint=None,
            verifier_paths=summary.verifier_files,
            rubric_type=rubric_type,
            metadata={
                "has_load_environment": summary.has_load_environment,
                "has_v1_taskset": summary.has_v1_taskset,
                "legacy_v0_detected": legacy_v0_only,
                "legacy_v0_files": summary.legacy_v0_files,
                "rubric_classes": sorted(summary.rubric_classes),
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
        raise RuntimeSandboxError(
            "verifiers v1 Tier 1 oracle execution is scheduled for Stage 6 M1."
        )

    async def run_noop(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        raise RuntimeSandboxError(
            "verifiers v1 Tier 1 no-op execution is scheduled for Stage 6 M1."
        )
