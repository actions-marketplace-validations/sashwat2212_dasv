"""Deterministic $0 Anti-Patch and ImpossibleBench mutation suite (SDD-02, FEAT-11)."""

from __future__ import annotations

from dasv.mutate.anti_patches import (
    AntiPatchMutation,
    AntiPatchOutcome,
    discover_mutable_source_files,
    generate_anti_patches,
    summarize_anti_patch_outcomes,
)
from dasv.mutate.impossible_bench import (
    ConflictingSpecVariant,
    OneOffMutation,
    OneOffOutcome,
    discover_python_verifier_files,
    generate_conflicting_spec_variants,
    generate_one_off_mutations,
    summarize_one_off_outcomes,
)
from dasv.mutate.runner import MutationReport, MutationSuiteRunner, OneOffOracleEvaluator

__all__ = [
    "AntiPatchMutation",
    "AntiPatchOutcome",
    "ConflictingSpecVariant",
    "MutationReport",
    "MutationSuiteRunner",
    "OneOffMutation",
    "OneOffOracleEvaluator",
    "OneOffOutcome",
    "discover_mutable_source_files",
    "discover_python_verifier_files",
    "generate_anti_patches",
    "generate_conflicting_spec_variants",
    "generate_one_off_mutations",
    "summarize_anti_patch_outcomes",
    "summarize_one_off_outcomes",
]
