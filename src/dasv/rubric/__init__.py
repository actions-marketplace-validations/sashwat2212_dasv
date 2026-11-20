"""ProRubric additive-compensation & verbosity/fluff rubric audit engine (SDD-02, FEAT-15)."""

from __future__ import annotations

from dasv.rubric.perturbations import (
    DEFAULT_INFLATION_THRESHOLD,
    PERTURBATION_KINDS,
    CallableRubricScorer,
    LLMRubricJudgeScorer,
    PerturbationKind,
    PerturbationOutcome,
    RubricJudgeScorer,
    RubricPerturbation,
    VerbosityAuditResult,
    build_default_wrong_baseline,
    generate_verbosity_perturbations,
    run_verbosity_perturbation_audit,
)
from dasv.rubric.prorubric_audit import (
    ProRubricAuditor,
    ProRubricCriterionSpec,
    ProRubricGateSpec,
    RubricAuditReport,
    RubricInspectionTarget,
    audit_rubric_static,
    build_prorubric_gate_spec,
    inspect_python_rubric_file,
    inspect_rubric_dict,
)

__all__ = [
    "CallableRubricScorer",
    "DEFAULT_INFLATION_THRESHOLD",
    "LLMRubricJudgeScorer",
    "PERTURBATION_KINDS",
    "PerturbationKind",
    "PerturbationOutcome",
    "ProRubricAuditor",
    "ProRubricCriterionSpec",
    "ProRubricGateSpec",
    "RubricAuditReport",
    "RubricInspectionTarget",
    "RubricJudgeScorer",
    "RubricPerturbation",
    "VerbosityAuditResult",
    "audit_rubric_static",
    "build_default_wrong_baseline",
    "build_prorubric_gate_spec",
    "generate_verbosity_perturbations",
    "inspect_python_rubric_file",
    "inspect_rubric_dict",
    "run_verbosity_perturbation_audit",
]
