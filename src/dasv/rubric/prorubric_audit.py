"""ProRubric additive-compensation static AST/schema auditor (`RULE-RUBR-001`, SDD-02 §4.1)."""

from __future__ import annotations

import ast
import json
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field

from dasv.core.schemas import AuditFinding, EnvSpec, Severity
from dasv.rubric.divergence import (
    DEFAULT_DELTA_THRESHOLD,
    DEFAULT_TAU_THRESHOLD,
    JudgeDivergenceResult,
    JudgeSample,
    build_default_divergence_samples,
    detect_reference_free_judge,
    run_judge_divergence_audit,
)
from dasv.rubric.perturbations import (
    DEFAULT_INFLATION_THRESHOLD,
    RubricJudgeScorer,
    VerbosityAuditResult,
    run_verbosity_perturbation_audit,
)

_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
}

_RUBRIC_CALLEES = {"RubricGroup", "Rubric", "JudgeRubric"}

_CONJUNCTIVE_MARKERS = frozenset(
    {
        "conjunctive",
        "conjunctive_gate",
        "conjunctive_gates",
        "conjunctive_gating",
        "mandatory_gates",
        "mandatory_criteria",
        "required_gates",
        "required_criteria",
        "hard_gates",
        "hard_violations",
        "hard_constraints",
        "all_required_pass",
        "no_hard_violation",
        "gate_fn",
        "gating",
        "fail_fast",
        "short_circuit_gate",
        "prorubricgate",
        "prorubric_gate",
    }
)

_ADDITIVE_AGG_VALUES = frozenset(
    {
        "sum",
        "additive",
        "linear",
        "linear_sum",
        "weighted_sum",
        "weighted_average",
        "weighted_mean",
        "mean",
        "average",
    }
)

_CONJUNCTIVE_AGG_VALUES = frozenset(
    {
        "conjunctive",
        "conjunctive_gated",
        "prorubric",
        "gated",
        "gated_weighted_sum",
        "all_pass",
        "multiplicative_gate",
    }
)


class ProRubricCriterionSpec(BaseModel):
    """Single criterion inside a recommended ProRubric conjunctive gate specification."""

    name: str
    role: Literal["mandatory_gate", "hard_violation_check", "secondary_weighted"]
    weight: float = 0.0
    description: str = ""


class ProRubricGateSpec(BaseModel):
    """Recommended ProRubric conjunctive protocol wrapper closing additive compensation."""

    protocol: Literal["prorubric_conjunctive_v1"] = "prorubric_conjunctive_v1"
    formula: str = (
        "score = (1.0 if all_required_pass and no_hard_violation else 0.0) * sum(w_i * c_i)"
    )
    source_file: str | None = None
    mandatory_gates: list[ProRubricCriterionSpec] = Field(
        default_factory=list[ProRubricCriterionSpec]
    )
    hard_violation_checks: list[ProRubricCriterionSpec] = Field(
        default_factory=list[ProRubricCriterionSpec]
    )
    secondary_criteria: list[ProRubricCriterionSpec] = Field(
        default_factory=list[ProRubricCriterionSpec]
    )
    python_wrapper_snippet: str = ""


class RubricInspectionTarget(BaseModel):
    """Summary of one statically analyzed rubric definition (Python AST or YAML/JSON/TOML)."""

    file_path: str
    line_number: int | None = None
    kind: Literal["python_ast", "yaml_schema", "json_schema", "toml_schema"]
    rubric_name: str
    criterion_count: int = 0
    criterion_names: list[str] = Field(default_factory=list[str])
    uses_additive_aggregation: bool = False
    has_conjunctive_gate: bool = False
    evidence: str = ""


class RubricAuditReport(BaseModel):
    """Complete output of `dasv rubric` combining RULE-RUBR-001, 002, and CHERRL 003."""

    task_id: str
    targets_inspected: list[RubricInspectionTarget] = Field(
        default_factory=list[RubricInspectionTarget]
    )
    findings: list[AuditFinding] = Field(default_factory=list[AuditFinding])
    recommended_gate_specs: list[ProRubricGateSpec] = Field(default_factory=list[ProRubricGateSpec])
    verbosity_audit: VerbosityAuditResult | None = None
    judge_divergence: JudgeDivergenceResult | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    budget_exceeded: bool = False

    @property
    def passed(self) -> bool:
        return not self.budget_exceeded and not any(
            f.rule_id in {"RULE-RUBR-001", "RULE-RUBR-002", "RULE-RUBR-003"} for f in self.findings
        )


def _norm_key(raw: str) -> str:
    return raw.strip().lower().replace("-", "_")


def _attr_or_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_conjunctive_identifier(name: str) -> bool:
    low = _norm_key(name)
    return low in _CONJUNCTIVE_MARKERS or "conjunctive" in low or "mandatory_gate" in low


def build_prorubric_gate_spec(
    *,
    source_file: str | None,
    criterion_names: list[str],
) -> ProRubricGateSpec:
    """Synthesize a concrete ProRubric conjunctive gate spec + Python protocol wrapper."""
    cleaned = [c.strip() for c in criterion_names if c.strip()]
    primary_names = [
        c
        for c in cleaned
        if any(
            tok in c.lower()
            for tok in ("correct", "fact", "accura", "answer", "logic", "pass", "valid", "safety")
        )
    ]
    if not primary_names:
        primary_names = cleaned[:1] if cleaned else ["factual_correctness"]
    secondary_names = [c for c in cleaned if c not in primary_names]
    if not secondary_names:
        secondary_names = ["clarity_and_structure", "completeness"]

    sec_weight = round(1.0 / max(1, len(secondary_names)), 4)
    mandatory = [
        ProRubricCriterionSpec(
            name=name,
            role="mandatory_gate",
            weight=0.0,
            description="Must pass unconditionally before any secondary rubric credit is awarded.",
        )
        for name in primary_names
    ]
    hard_checks = [
        ProRubricCriterionSpec(
            name="no_factual_contradiction_or_fabrication",
            role="hard_violation_check",
            weight=0.0,
            description=(
                "Any fabricated citation, hallucinated fact, or unsafe shortcut forces 0.0."
            ),
        )
    ]
    secondary = [
        ProRubricCriterionSpec(
            name=name,
            role="secondary_weighted",
            weight=sec_weight,
            description="Weighted secondary quality signal awarded ONLY when all gates pass.",
        )
        for name in secondary_names
    ]

    gate_list_literal = repr([m.name for m in mandatory])
    violation_literal = repr([h.name for h in hard_checks])
    sec_weights_literal = repr({s.name: s.weight for s in secondary})
    snippet = (
        "def prorubric_conjunctive_score(\n"
        "    gate_checks: dict[str, bool],\n"
        "    violation_flags: dict[str, bool],\n"
        "    secondary_scores: dict[str, float],\n"
        ") -> float:\n"
        f"    required_gates = {gate_list_literal}\n"
        f"    hard_violations = {violation_literal}\n"
        f"    secondary_weights = {sec_weights_literal}\n"
        "    all_required_pass = all(bool(gate_checks.get(k, False)) for k in required_gates)\n"
        "    no_hard_violation = not any(\n"
        "        bool(violation_flags.get(k, False)) for k in hard_violations\n"
        "    )\n"
        "    if not (all_required_pass and no_hard_violation):\n"
        "        return 0.0\n"
        "    total_w = sum(secondary_weights.values()) or 1.0\n"
        "    weighted = sum(\n"
        "        secondary_scores.get(k, 0.0) * w for k, w in secondary_weights.items()\n"
        "    )\n"
        "    return max(0.0, min(1.0, weighted / total_w))\n"
    )
    return ProRubricGateSpec(
        source_file=source_file,
        mandatory_gates=mandatory,
        hard_violation_checks=hard_checks,
        secondary_criteria=secondary,
        python_wrapper_snippet=snippet,
    )


def _extract_list_item_names(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return []
    names: list[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            names.append(elt.value)
        elif isinstance(elt, ast.Name):
            names.append(elt.id)
        elif isinstance(elt, ast.Attribute):
            names.append(elt.attr)
        elif isinstance(elt, ast.Call):
            callee = _attr_or_name(elt.func)
            if callee:
                names.append(callee)
        elif isinstance(elt, ast.Dict):
            for k, v in zip(elt.keys, elt.values, strict=False):
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and _norm_key(k.value) in {"name", "id", "criterion"}
                    and isinstance(v, ast.Constant)
                    and isinstance(v.value, str)
                ):
                    names.append(v.value)
                    break
    return names


def _dict_ast_keys_and_values(node: ast.Dict) -> dict[str, ast.AST]:
    mapping: dict[str, ast.AST] = {}
    for k, v in zip(node.keys, node.values, strict=False):
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            mapping[_norm_key(k.value)] = v
    return mapping


def _tree_has_conjunctive_guard(tree: ast.AST) -> bool:
    """Detect module/function-level conjunctive gating logic in Python AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and _is_conjunctive_identifier(node.id):
            return True
        if isinstance(node, ast.Attribute) and _is_conjunctive_identifier(node.attr):
            return True
        if isinstance(node, ast.keyword) and node.arg and _is_conjunctive_identifier(node.arg):
            if isinstance(node.value, ast.Constant) and node.value.value is False:
                continue
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            low = _norm_key(node.value)
            if low in _CONJUNCTIVE_AGG_VALUES or low in {
                "all_required_pass",
                "no_hard_violation",
                "mandatory_gate",
            }:
                return True
        if isinstance(node, ast.FunctionDef) and _is_conjunctive_identifier(node.name):
            return True
    return False


def _function_has_zero_short_circuit_gate(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if a scoring function short-circuits to 0.0 on a failed guard condition."""
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            for sub in node.body:
                if (
                    isinstance(sub, ast.Return)
                    and isinstance(sub.value, ast.Constant)
                    and isinstance(sub.value.value, (int, float))
                    and float(sub.value.value) == 0.0
                ):
                    return True
        elif isinstance(node, ast.IfExp):
            for branch in (node.body, node.orelse):
                if (
                    isinstance(branch, ast.Constant)
                    and isinstance(branch.value, (int, float))
                    and float(branch.value) == 0.0
                ):
                    return True
    return False


def _function_computes_weighted_sum(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if a Python scoring function computes a linear/weighted additive sum."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and _attr_or_name(node.func) == "sum":
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            # Look for chained additions of weighted terms (w * s + ...)
            has_mul = any(
                isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult)
                for sub in ast.walk(node)
            )
            if has_mul:
                return True
    return False


def inspect_python_rubric_file(
    py_path: Path,
    *,
    rel_path: str,
) -> list[RubricInspectionTarget]:
    """Statically parse a Python file via `ast` for additive vs conjunctive rubric definitions."""
    try:
        source = py_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    file_has_conjunctive = _tree_has_conjunctive_guard(tree)
    targets: list[RubricInspectionTarget] = []

    for node in ast.walk(tree):
        # 1. Prime Intellect verifiers v1 calls: RubricGroup(...), Rubric(...), JudgeRubric(...)
        if isinstance(node, ast.Call):
            callee = _attr_or_name(node.func)
            if callee not in _RUBRIC_CALLEES:
                continue

            kw_map: dict[str, ast.AST] = {
                _norm_key(kw.arg): kw.value for kw in node.keywords if kw.arg is not None
            }
            item_names: list[str] = []
            if node.args:
                item_names.extend(_extract_list_item_names(node.args[0]))
            for list_key in ("rubrics", "funcs", "criteria", "items", "reward_funcs"):
                if list_key in kw_map:
                    item_names.extend(_extract_list_item_names(kw_map[list_key]))

            has_weights = "weights" in kw_map or "criterion_weights" in kw_map
            agg_val = ""
            if "aggregation" in kw_map and isinstance(kw_map["aggregation"], ast.Constant):
                agg_val = _norm_key(str(kw_map["aggregation"].value))

            call_conjunctive = agg_val in _CONJUNCTIVE_AGG_VALUES or any(
                _is_conjunctive_identifier(k)
                and not (isinstance(v, ast.Constant) and v.value is False)
                for k, v in kw_map.items()
            )
            is_additive = (
                callee == "RubricGroup"
                or has_weights
                or len(item_names) >= 2
                or agg_val in _ADDITIVE_AGG_VALUES
            )
            if is_additive:
                targets.append(
                    RubricInspectionTarget(
                        file_path=rel_path,
                        line_number=getattr(node, "lineno", None),
                        kind="python_ast",
                        rubric_name=callee,
                        criterion_count=max(len(item_names), 2 if has_weights else len(item_names)),
                        criterion_names=item_names,
                        uses_additive_aggregation=not call_conjunctive,
                        has_conjunctive_gate=call_conjunctive or file_has_conjunctive,
                        evidence=(
                            f"{callee} at {rel_path}:{getattr(node, 'lineno', 1)} "
                            f"aggregates {len(item_names) or 'multiple'} criteria "
                            f"(weights={has_weights}, aggregation={agg_val or 'default_linear'})."
                        ),
                    )
                )

        # 2. Dict-literal rubric schemas in Python (e.g. {"criteria": [...], "weights": ...})
        elif isinstance(node, ast.Dict):
            kv = _dict_ast_keys_and_values(node)
            if not ({"criteria", "items", "rubric_items"} & set(kv.keys())):
                continue
            crit_node = kv.get("criteria") or kv.get("items") or kv.get("rubric_items")
            names = _extract_list_item_names(crit_node) if crit_node is not None else []
            has_weights = "weights" in kv or (
                isinstance(crit_node, ast.List)
                and any(
                    isinstance(elt, ast.Dict) and "weight" in _dict_ast_keys_and_values(elt)
                    for elt in crit_node.elts
                )
            )
            agg_val = ""
            if "aggregation" in kv and isinstance(kv["aggregation"], ast.Constant):
                agg_val = _norm_key(str(kv["aggregation"].value))
            dict_conjunctive = agg_val in _CONJUNCTIVE_AGG_VALUES or any(
                _is_conjunctive_identifier(k)
                and not (isinstance(v, ast.Constant) and v.value is False)
                for k, v in kv.items()
            )
            if has_weights or len(names) >= 2 or agg_val in _ADDITIVE_AGG_VALUES:
                targets.append(
                    RubricInspectionTarget(
                        file_path=rel_path,
                        line_number=getattr(node, "lineno", None),
                        kind="python_ast",
                        rubric_name="python_rubric_dict",
                        criterion_count=len(names),
                        criterion_names=names,
                        uses_additive_aggregation=not dict_conjunctive,
                        has_conjunctive_gate=dict_conjunctive or file_has_conjunctive,
                        evidence=(
                            f"Python rubric dict at {rel_path}:{getattr(node, 'lineno', 1)} "
                            f"defines {len(names)} criteria (weights={has_weights}, "
                            f"aggregation={agg_val or 'linear'})."
                        ),
                    )
                )

        # 3. Explicit rubric scoring functions computing weighted linear sums without zero-gate
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_low = node.name.lower()
            if not any(tok in fn_low for tok in ("rubric", "score", "grade", "reward")):
                continue
            if _function_computes_weighted_sum(node):
                has_gate = (
                    _function_has_zero_short_circuit_gate(node)
                    or _tree_has_conjunctive_guard(node)
                    or file_has_conjunctive
                )
                targets.append(
                    RubricInspectionTarget(
                        file_path=rel_path,
                        line_number=node.lineno,
                        kind="python_ast",
                        rubric_name=node.name,
                        criterion_count=2,
                        criterion_names=[],
                        uses_additive_aggregation=True,
                        has_conjunctive_gate=has_gate,
                        evidence=(
                            f"Scoring function `{node.name}` at {rel_path}:{node.lineno} "
                            f"computes an additive weighted sum "
                            f"(conjunctive_gate={has_gate})."
                        ),
                    )
                )

    return targets


def inspect_rubric_dict(
    data: dict[str, Any],
    *,
    file_path: str,
    kind: Literal["yaml_schema", "json_schema", "toml_schema"] = "yaml_schema",
) -> list[RubricInspectionTarget]:
    """Statically inspect a parsed YAML/JSON/TOML rubric mapping for additive compensation."""
    targets: list[RubricInspectionTarget] = []

    def _walk_mapping(obj: dict[str, Any], label: str) -> None:
        norm_map: dict[str, Any] = {_norm_key(str(k)): v for k, v in obj.items()}

        # Recurse into nested `rubric` / `rubrics` sub-tables first if top-level is a task manifest.
        if "rubric" in norm_map and isinstance(norm_map["rubric"], dict):
            _walk_mapping(cast(dict[str, Any], norm_map["rubric"]), f"{label}.rubric")

        criteria_raw = (
            norm_map.get("criteria") or norm_map.get("items") or norm_map.get("rubric_items")
        )
        weights_raw = norm_map.get("weights")
        agg_raw = str(norm_map.get("aggregation") or norm_map.get("mode") or "").strip()
        agg_norm = _norm_key(agg_raw)

        has_rubric_shape = (
            isinstance(criteria_raw, list)
            or isinstance(weights_raw, (list, dict))
            or agg_norm in (_ADDITIVE_AGG_VALUES | _CONJUNCTIVE_AGG_VALUES)
        )
        if not has_rubric_shape:
            return

        criterion_names: list[str] = []
        item_has_weight = False
        item_has_gate = False
        if isinstance(criteria_raw, list):
            for idx, entry in enumerate(cast(list[Any], criteria_raw), start=1):
                if isinstance(entry, str):
                    criterion_names.append(entry)
                elif isinstance(entry, dict):
                    ed = {_norm_key(str(k)): v for k, v in cast(dict[Any, Any], entry).items()}
                    cname = str(
                        ed.get("name") or ed.get("id") or ed.get("criterion") or f"criterion_{idx}"
                    )
                    criterion_names.append(cname)
                    if "weight" in ed or "points" in ed:
                        item_has_weight = True
                    if any(
                        bool(ed.get(gk))
                        for gk in (
                            "mandatory",
                            "required_gate",
                            "conjunctive_gate",
                            "hard_gate",
                            "is_gate",
                            "gate",
                            "hard_violation",
                        )
                    ):
                        item_has_gate = True

        has_conjunctive = (
            agg_norm in _CONJUNCTIVE_AGG_VALUES
            or item_has_gate
            or any(_is_conjunctive_identifier(k) and bool(v) for k, v in norm_map.items())
        )
        uses_additive = (
            agg_norm in _ADDITIVE_AGG_VALUES
            or item_has_weight
            or isinstance(weights_raw, (list, dict))
            or (len(criterion_names) >= 2 and not has_conjunctive)
        )
        if uses_additive or has_conjunctive:
            targets.append(
                RubricInspectionTarget(
                    file_path=file_path,
                    line_number=1,
                    kind=kind,
                    rubric_name=str(norm_map.get("name") or label),
                    criterion_count=len(criterion_names),
                    criterion_names=criterion_names,
                    uses_additive_aggregation=uses_additive and not has_conjunctive,
                    has_conjunctive_gate=has_conjunctive,
                    evidence=(
                        f"Rubric schema in {file_path} defines {len(criterion_names)} criteria "
                        f"(aggregation={agg_norm or 'implicit_additive'}, "
                        f"conjunctive_gate={has_conjunctive})."
                    ),
                )
            )

    _walk_mapping(data, Path(file_path).stem)
    return targets


def _discover_rubric_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        suffix = path.suffix.lower()
        name_low = path.name.lower()
        if (
            suffix == ".py"
            or name_low == "task.toml"
            or (
                suffix in {".yaml", ".yml", ".json"}
                and ("rubric" in name_low or "judge" in name_low or "criteria" in name_low)
            )
        ):
            found.append(path)
    return found


def audit_rubric_static(
    target_path: Path,
    *,
    spec: EnvSpec | None = None,
) -> tuple[list[RubricInspectionTarget], list[AuditFinding], list[ProRubricGateSpec]]:
    """Run $0 static ProRubric AST/schema analysis for RULE-RUBR-001 on a directory or file."""
    root = target_path.resolve()
    base_root = (
        spec.root_path.resolve() if spec is not None else (root.parent if root.is_file() else root)
    )
    files = _discover_rubric_files(root)

    targets: list[RubricInspectionTarget] = []
    for path in files:
        try:
            rel_posix = path.relative_to(base_root).as_posix()
        except ValueError:
            rel_posix = path.name
        suffix = path.suffix.lower()
        if suffix == ".py":
            targets.extend(inspect_python_rubric_file(path, rel_path=rel_posix))
        elif suffix in {".yaml", ".yml"}:
            try:
                loaded_y: Any = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(loaded_y, dict):
                targets.extend(
                    inspect_rubric_dict(
                        cast(dict[str, Any], loaded_y),
                        file_path=rel_posix,
                        kind="yaml_schema",
                    )
                )
        elif suffix == ".json":
            try:
                loaded_j: Any = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(loaded_j, dict):
                targets.extend(
                    inspect_rubric_dict(
                        cast(dict[str, Any], loaded_j),
                        file_path=rel_posix,
                        kind="json_schema",
                    )
                )
        elif suffix == ".toml":
            try:
                loaded_t: dict[str, Any] = tomllib.loads(
                    path.read_text(encoding="utf-8", errors="replace")
                )
            except (OSError, tomllib.TOMLDecodeError):
                continue
            targets.extend(inspect_rubric_dict(loaded_t, file_path=rel_posix, kind="toml_schema"))

    findings: list[AuditFinding] = []
    gate_specs: list[ProRubricGateSpec] = []
    flagged_files: set[tuple[str, int | None]] = set()

    for target in targets:
        if target.uses_additive_aggregation and not target.has_conjunctive_gate:
            key = (target.file_path, target.line_number)
            if key in flagged_files:
                continue
            flagged_files.add(key)
            gate_spec = build_prorubric_gate_spec(
                source_file=target.file_path,
                criterion_names=target.criterion_names,
            )
            gate_specs.append(gate_spec)
            findings.append(
                AuditFinding(
                    rule_id="RULE-RUBR-001",
                    title="Additive rubric compensation risk: missing conjunctive pass/fail gate",
                    severity=Severity.HIGH,
                    tier=2,
                    source="dasv",
                    file_path=target.file_path,
                    line_number=target.line_number,
                    evidence=(
                        f"{target.evidence} Linear additive aggregation allows verbose or "
                        "stylistic fluff criteria to buy back points for factually wrong answers."
                    ),
                    remediation=(
                        "Adopt a ProRubric conjunctive gate (`all_required_pass AND "
                        "no_hard_violation`) that zeroes total reward whenever mandatory "
                        "correctness or safety criteria fail."
                    ),
                )
            )

    return targets, findings, gate_specs


class ProRubricAuditor:
    """Runs static ProRubric checks (RULE-RUBR-001) + verbosity harness (RULE-RUBR-002)."""

    async def audit(
        self,
        target_path: Path,
        *,
        spec: EnvSpec | None = None,
        task_id: str | None = None,
        task_prompt: str | None = None,
        scorer: RubricJudgeScorer | None = None,
        baseline_response: str | None = None,
        inflation_threshold: float = DEFAULT_INFLATION_THRESHOLD,
        divergence_judges: Sequence[tuple[str, RubricJudgeScorer]] | None = None,
        divergence_samples: Sequence[JudgeSample] | None = None,
        tau_threshold: float = DEFAULT_TAU_THRESHOLD,
        delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
        seed: int = 42,
        out_dir: Path | None = None,
    ) -> RubricAuditReport:
        """Execute static ProRubric audit, verbosity harness, and optional CHERRL divergence."""
        effective_task_id = task_id or (spec.task_id if spec is not None else target_path.name)
        effective_prompt = (
            task_prompt
            if task_prompt is not None
            else (spec.instruction_text if spec is not None else f"Evaluate {effective_task_id}.")
        )

        targets, static_findings, gate_specs = audit_rubric_static(target_path, spec=spec)
        findings: list[AuditFinding] = list(static_findings)
        verbosity_res: VerbosityAuditResult | None = None
        divergence_res: JudgeDivergenceResult | None = None
        first_file = targets[0].file_path if targets else None
        budget_tripped = False

        if scorer is not None:
            verbosity_res = await run_verbosity_perturbation_audit(
                effective_prompt,
                scorer,
                baseline_response=baseline_response,
                inflation_threshold=inflation_threshold,
                file_path=first_file,
            )
            findings.extend(verbosity_res.findings)
            if not verbosity_res.passed and not gate_specs:
                gate_specs.append(
                    build_prorubric_gate_spec(
                        source_file=first_file,
                        criterion_names=targets[0].criterion_names if targets else [],
                    )
                )

        if divergence_judges:
            is_ref_free, ref_evidence = detect_reference_free_judge(target_path, targets)
            samples = (
                list(divergence_samples)
                if divergence_samples
                else build_default_divergence_samples(
                    effective_prompt,
                    baseline_response=baseline_response,
                )
            )
            divergence_res = await run_judge_divergence_audit(
                samples,
                divergence_judges,
                tau_threshold=tau_threshold,
                delta_threshold=delta_threshold,
                is_reference_free=is_ref_free,
                reference_anchor_evidence=ref_evidence,
                file_path=first_file,
                seed=seed,
            )
            findings.extend(divergence_res.findings)
            budget_tripped = divergence_res.partial_budget_exceeded
            if not divergence_res.passed and not gate_specs:
                gate_specs.append(
                    build_prorubric_gate_spec(
                        source_file=first_file,
                        criterion_names=targets[0].criterion_names if targets else [],
                    )
                )

        report = RubricAuditReport(
            task_id=effective_task_id,
            targets_inspected=targets,
            findings=findings,
            recommended_gate_specs=gate_specs,
            verbosity_audit=verbosity_res,
            judge_divergence=divergence_res,
            budget_exceeded=budget_tripped,
        )
        if out_dir is not None:
            self.write_artifacts(out_dir, report)
        return report

    @staticmethod
    def write_artifacts(out_dir: Path, report: RubricAuditReport) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "rubric_audit.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if report.recommended_gate_specs:
            gate_payload = {
                "task_id": report.task_id,
                "prorubric_gates": [
                    g.model_dump(mode="json") for g in report.recommended_gate_specs
                ],
            }
            (out_dir / "prorubric_gate.yaml").write_text(
                yaml.safe_dump(gate_payload, sort_keys=False),
                encoding="utf-8",
            )
