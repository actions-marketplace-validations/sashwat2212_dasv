"""Clean-image verifier engine, artifact extraction, and N-trial determinism runner (SDD-02)."""

from __future__ import annotations

from dasv.verifier.artifact_extractor import (
    ArtifactExtractor,
    ExtractedArtifact,
    build_tamper_findings,
    compute_artifact_sha256,
    filter_git_patch,
    is_protected_verifier_path,
)
from dasv.verifier.clean_grader import (
    VERIFIER_MOUNT_PATH,
    CleanImageGrader,
    GradeOutcome,
    GradingContext,
    coerce_grading_context,
    outcome_to_details,
    parse_verifier_reward,
    resolve_verifier_command,
    resolve_verifier_host_dir,
)
from dasv.verifier.determinism import (
    DeterminismResult,
    DeterminismRunner,
    TrialRecord,
)
from dasv.verifier.gold_sanity import (
    GoldSanityGate,
    GoldSanityResult,
    apply_unified_diff_to_dir,
    validate_safe_unified_diff,
)

__all__ = [
    "ArtifactExtractor",
    "CleanImageGrader",
    "DeterminismResult",
    "DeterminismRunner",
    "ExtractedArtifact",
    "GoldSanityGate",
    "GoldSanityResult",
    "GradeOutcome",
    "GradingContext",
    "TrialRecord",
    "VERIFIER_MOUNT_PATH",
    "apply_unified_diff_to_dir",
    "build_tamper_findings",
    "coerce_grading_context",
    "compute_artifact_sha256",
    "filter_git_patch",
    "is_protected_verifier_path",
    "outcome_to_details",
    "parse_verifier_reward",
    "resolve_verifier_command",
    "resolve_verifier_host_dir",
    "validate_safe_unified_diff",
]
