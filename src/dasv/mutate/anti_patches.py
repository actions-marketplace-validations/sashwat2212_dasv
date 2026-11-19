"""Deterministic $0 Anti-Patch Mutators (`MUT-AP-001` through `MUT-AP-005`, SDD-02)."""

from __future__ import annotations

import ast
import copy
import difflib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from dasv.core.schemas import AuditFinding, EnvSpec, Severity
from dasv.verifier.artifact_extractor import (
    ExtractedArtifact,
    compute_artifact_sha256,
    is_protected_verifier_path,
)

_IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "solution",
    "tests",
    "test",
}


class AntiPatchMutation(BaseModel):
    """A single deterministically synthesized anti-patch targeting source code."""

    mutator_id: Literal[
        "MUT-AP-001",
        "MUT-AP-002",
        "MUT-AP-003",
        "MUT-AP-004",
        "MUT-AP-005",
    ]
    name: str
    description: str
    target_file: str
    patch_diff: str
    mutated_source: str | None = None

    def to_extracted_artifact(self) -> ExtractedArtifact:
        """Convert this anti-patch into an immutable ExtractedArtifact for CleanImageGrader."""
        patch_bytes = self.patch_diff.encode("utf-8")
        return ExtractedArtifact(
            mode="git_patch",
            patch_bytes=patch_bytes,
            sha256=compute_artifact_sha256("git_patch", patch_bytes, {}),
        )


class AntiPatchOutcome(BaseModel):
    """Grading outcome of a single anti-patch evaluated inside CleanImageGrader."""

    mutator_id: str
    name: str
    target_file: str
    reward: float
    survived: bool
    detail: str = ""


def _build_unified_diff(rel_posix: str, original: str, mutated: str) -> str:
    lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        mutated.splitlines(keepends=True),
        fromfile=f"a/{rel_posix}",
        tofile=f"b/{rel_posix}",
    )
    body = "".join(lines)
    if not body:
        return ""
    header = f"diff --git a/{rel_posix} b/{rel_posix}\n"
    return header + body


def discover_mutable_source_files(spec: EnvSpec, *, max_files: int = 5) -> list[Path]:
    """Return candidate source files under `spec.root_path`, strictly excluding verifier paths."""
    root = spec.root_path.resolve()
    if not root.is_dir():
        return []

    oracle_rel = (spec.oracle_entrypoint or "").strip().lstrip("./").lstrip("/")
    candidates: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts)
        if parts & _IGNORED_DIRS:
            continue
        if oracle_rel and rel == oracle_rel:
            continue
        if is_protected_verifier_path(rel, spec.verifier_paths):
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        candidates.append(path)
        if len(candidates) >= max_files:
            break
    return candidates


def _preserve_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return [body[0]]
    return []


class _ReturnConstantStubTransformer(ast.NodeTransformer):
    """MUT-AP-002: Rewrite non-dunder function bodies to return a trivial constant."""

    def __init__(self, constant_value: str | bytes | bool | int | float | None = None) -> None:
        self._constant = constant_value
        self.mutated_count = 0

    def _rewrite_fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        if node.name.startswith("__") and node.name.endswith("__"):
            return node
        self.mutated_count += 1
        doc = _preserve_docstring(node.body)
        node.body = [*doc, ast.Return(value=ast.Constant(value=self._constant))]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._rewrite_fn(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._rewrite_fn(node)


class _ExceptionSwallowerTransformer(ast.NodeTransformer):
    """MUT-AP-003: Wrap function bodies in `try: ... except Exception: return None`."""

    def __init__(self) -> None:
        self.mutated_count = 0

    def _rewrite_fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        if node.name.startswith("__") and node.name.endswith("__"):
            return node
        doc = _preserve_docstring(node.body)
        inner = node.body[len(doc) :]
        if not inner:
            return node
        self.mutated_count += 1
        try_node = ast.Try(
            body=inner,
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id="Exception", ctx=ast.Load()),
                    name=None,
                    body=[ast.Return(value=ast.Constant(value=None))],
                )
            ],
            orelse=[],
            finalbody=[],
        )
        node.body = [*doc, try_node]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._rewrite_fn(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._rewrite_fn(node)


class _ExitCodeSpooferTransformer(ast.NodeTransformer):
    """MUT-AP-004: Inject `import sys; sys.exit(0)` at function entry to test runner isolation."""

    def __init__(self) -> None:
        self.mutated_count = 0

    def _rewrite_fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        if node.name.startswith("__") and node.name.endswith("__"):
            return node
        self.mutated_count += 1
        doc = _preserve_docstring(node.body)
        import_sys = ast.Import(names=[ast.alias(name="sys", asname=None)])
        exit_zero = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="sys", ctx=ast.Load()),
                    attr="exit",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=0)],
                keywords=[],
            )
        )
        node.body = [*doc, import_sys, exit_zero]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._rewrite_fn(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._rewrite_fn(node)


class _EdgeCaseNullifierTransformer(ast.NodeTransformer):
    """MUT-AP-005: Remove conditional guard branches (`if ...: raise ...` / early `return`)."""

    def __init__(self) -> None:
        self.mutated_count = 0

    @staticmethod
    def _is_guard_if(stmt: ast.stmt) -> bool:
        if not isinstance(stmt, ast.If) or stmt.orelse:
            return False
        return any(isinstance(sub, (ast.Raise, ast.Return)) for sub in stmt.body)

    def _rewrite_fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        new_body: list[ast.stmt] = []
        removed_here = False
        for stmt in node.body:
            if self._is_guard_if(stmt):
                self.mutated_count += 1
                removed_here = True
                continue
            new_body.append(stmt)
        if removed_here:
            node.body = new_body or [ast.Pass()]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._rewrite_fn(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._rewrite_fn(node)


def _apply_ast_mutator(
    source: str,
    transformer: ast.NodeTransformer,
) -> tuple[str, int] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    cloned = copy.deepcopy(tree)
    mutated_tree = transformer.visit(cloned)
    ast.fix_missing_locations(mutated_tree)
    mutated_count = int(getattr(transformer, "mutated_count", 0))
    if mutated_count <= 0:
        return None
    rendered = ast.unparse(mutated_tree) + "\n"
    # Guarantee generated Python parses cleanly before emitting.
    ast.parse(rendered)
    return rendered, mutated_count


def generate_anti_patches(spec: EnvSpec, *, max_files: int = 3) -> list[AntiPatchMutation]:
    """Generate deterministic $0 anti-patches (`MUT-AP-001`..`MUT-AP-005`) for `spec`."""
    root = spec.root_path.resolve()
    files = discover_mutable_source_files(spec, max_files=max_files)
    mutations: list[AntiPatchMutation] = []

    if not files:
        # Always emit a zero-change empty patch baseline (MUT-AP-001) even if no .py source exists.
        mutations.append(
            AntiPatchMutation(
                mutator_id="MUT-AP-001",
                name="Empty / No-op Patch",
                description="Empty patch verifying the grader rejects an untouched codebase.",
                target_file=".",
                patch_diff="",
                mutated_source=None,
            )
        )
        return mutations

    for file_path in files:
        rel_posix = file_path.relative_to(root).as_posix()
        try:
            original = file_path.read_text(encoding="utf-8")
        except OSError:
            continue

        # MUT-AP-001: Comment-only no-op patch
        comment_mutated = (
            original + ("\n" if original and not original.endswith("\n") else "") + "# dasv noop\n"
        )
        diff_noop = _build_unified_diff(rel_posix, original, comment_mutated)
        if diff_noop:
            mutations.append(
                AntiPatchMutation(
                    mutator_id="MUT-AP-001",
                    name="Comment-Only No-op Patch",
                    description="Adds a harmless comment without solving the task.",
                    target_file=rel_posix,
                    patch_diff=diff_noop,
                    mutated_source=comment_mutated,
                )
            )

        # MUT-AP-002: Return-constant stub (None and 0 variants)
        stub_res = _apply_ast_mutator(original, _ReturnConstantStubTransformer(None))
        if stub_res is not None:
            rendered, _ = stub_res
            diff_stub = _build_unified_diff(rel_posix, original, rendered)
            if diff_stub:
                mutations.append(
                    AntiPatchMutation(
                        mutator_id="MUT-AP-002",
                        name="Return-Constant Stub",
                        description="Rewrites function bodies to `return None`.",
                        target_file=rel_posix,
                        patch_diff=diff_stub,
                        mutated_source=rendered,
                    )
                )

        # MUT-AP-003: Exception swallower
        swallow_res = _apply_ast_mutator(original, _ExceptionSwallowerTransformer())
        if swallow_res is not None:
            rendered, _ = swallow_res
            diff_swallow = _build_unified_diff(rel_posix, original, rendered)
            if diff_swallow:
                mutations.append(
                    AntiPatchMutation(
                        mutator_id="MUT-AP-003",
                        name="Exception Swallower",
                        description="Wraps function bodies in `try/except Exception: return None`.",
                        target_file=rel_posix,
                        patch_diff=diff_swallow,
                        mutated_source=rendered,
                    )
                )

        # MUT-AP-004: Exit-code spoofer
        spoof_res = _apply_ast_mutator(original, _ExitCodeSpooferTransformer())
        if spoof_res is not None:
            rendered, _ = spoof_res
            diff_spoof = _build_unified_diff(rel_posix, original, rendered)
            if diff_spoof:
                mutations.append(
                    AntiPatchMutation(
                        mutator_id="MUT-AP-004",
                        name="Exit-Code Spoofer",
                        description="Injects `sys.exit(0)` into target functions to test runner "
                        "isolation against in-process exit spoofing.",
                        target_file=rel_posix,
                        patch_diff=diff_spoof,
                        mutated_source=rendered,
                    )
                )

        # MUT-AP-005: Edge-case guard nullifier
        guard_res = _apply_ast_mutator(original, _EdgeCaseNullifierTransformer())
        if guard_res is not None:
            rendered, _ = guard_res
            diff_guard = _build_unified_diff(rel_posix, original, rendered)
            if diff_guard:
                mutations.append(
                    AntiPatchMutation(
                        mutator_id="MUT-AP-005",
                        name="Edge-Case Guard Nullifier",
                        description="Deletes `if ...: raise/return` boundary checks.",
                        target_file=rel_posix,
                        patch_diff=diff_guard,
                        mutated_source=rendered,
                    )
                )

    return mutations


def summarize_anti_patch_outcomes(
    outcomes: list[AntiPatchOutcome],
) -> tuple[float, list[AuditFinding]]:
    """Compute Mutation Survival Rate and emit `RULE-MUT-002` when broken patches score > 0."""
    if not outcomes:
        return 0.0, []

    survivors = [o for o in outcomes if o.survived]
    survival_rate = round(len(survivors) / len(outcomes), 6)
    if not survivors:
        return 0.0, []

    severity = Severity.CRITICAL if survival_rate >= 0.25 else Severity.HIGH
    survivor_summary = ", ".join(
        f"{s.mutator_id} ({s.target_file}, r={s.reward:.2f})" for s in survivors
    )
    findings = [
        AuditFinding(
            rule_id="RULE-MUT-002",
            title="Weak verifier passed broken anti-patch mutation(s)",
            severity=severity,
            tier=1,
            source="dasv",
            file_path=survivors[0].target_file if survivors[0].target_file != "." else None,
            line_number=None,
            evidence=(
                f"Mutation Survival Rate = {survival_rate * 100.0:.1f}% "
                f"({len(survivors)}/{len(outcomes)} broken patches scored reward > 0): "
                f"{survivor_summary}"
            ),
            remediation=(
                "Strengthen verifier assertions to validate semantic outputs, boundary cases, "
                "and collected test counts (preventing stub, swallow, and exit-0 bypasses)."
            ),
        )
    ]
    return survival_rate, findings
