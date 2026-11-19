"""Exact RLVR statistical estimators, HackabilityScore, and Grade A-F logic (SDD-04)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from dasv.core.schemas import (
    AuditFinding,
    CurriculumBucket,
    DeterminismMetrics,
    ExploitTrace,
    Severity,
)

_CRIT_BOUNDARY_RULES = {
    "RULE-BNDY-001",
    "RULE-BNDY-002",
    "RULE-BNDY-003",
    "RULE-BNDY-004",
    "RULE-BNDY-005",
    "RULE-BNDY-006",
}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Chen et al. pass@k estimator without factorial overflow."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if c < 0 or c > n:
        raise ValueError("c must satisfy 0 <= c <= n")
    if k < 1 or k > n:
        raise ValueError("k must satisfy 1 <= k <= n")
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0

    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= 1.0 - (c / (n - i))
    return max(0.0, min(1.0, 1.0 - prob_all_fail))


def pass_pow_k(n: int, c: int, k: int) -> float:
    """Reliability estimator pass^k = (c / n)^k exposing flakiness across k trials."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if c < 0 or c > n:
        raise ValueError("c must satisfy 0 <= c <= n")
    if k < 1:
        raise ValueError("k must be >= 1")
    return (c / n) ** k


def binary_reward_variance(p: float) -> float:
    """Within-group binary reward variance p * (1 - p)."""
    if p < 0.0 or p > 1.0:
        raise ValueError("p must satisfy 0.0 <= p <= 1.0")
    return p * (1.0 - p)


def zero_grad_group_prob(p: float, group_size: int) -> float:
    """Probability that a GRPO/DAPO group of size G has zero advantage variance: p^G + (1-p)^G."""
    if p < 0.0 or p > 1.0:
        raise ValueError("p must satisfy 0.0 <= p <= 1.0")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    return (p**group_size) + ((1.0 - p) ** group_size)


def taskset_zero_grad_waste_pct(pass_rates: Sequence[float], group_size: int) -> float:
    """Mean zero-gradient group probability across a taskset, expressed as a percentage [0, 100]."""
    if not pass_rates:
        raise ValueError("pass_rates must be non-empty")
    mean_prob = sum(zero_grad_group_prob(p, group_size) for p in pass_rates) / len(pass_rates)
    return mean_prob * 100.0


def classify_curriculum_bucket(
    pass_at_1: float,
    pass_at_g: float,
    replay_variance: float = 0.0,
    *,
    goldilocks_low: float = 0.05,
    goldilocks_high: float = 0.40,
    trivial_threshold: float = 0.90,
) -> CurriculumBucket:
    """Classify a task into GOLDILOCKS, TRIVIAL_ZERO_GRAD, STARVED_ZERO_GRAD, or FLAKY."""
    if replay_variance > 0.0:
        return CurriculumBucket.FLAKY
    if pass_at_g <= 0.0:
        return CurriculumBucket.STARVED_ZERO_GRAD
    if pass_at_1 > trivial_threshold:
        return CurriculumBucket.TRIVIAL_ZERO_GRAD
    if goldilocks_low <= pass_at_1 <= goldilocks_high:
        return CurriculumBucket.GOLDILOCKS
    # Mid-band tasks outside strict Goldilocks but non-trivial/non-starved still yield signal
    return CurriculumBucket.GOLDILOCKS if pass_at_1 > 0.0 else CurriculumBucket.STARVED_ZERO_GRAD


def compute_hackability_score(
    findings: Sequence[AuditFinding],
    mutation_survival_rate: float | None = None,
    exploits: Sequence[ExploitTrace] = (),
    rubric_score: float | None = None,
) -> float:
    """Compute bounded HackabilityScore in [0.0, 100.0] per SDD-04 section 2.1."""
    crit_boundary_hits = {
        f.rule_id
        for f in findings
        if f.severity == Severity.CRITICAL and f.rule_id in _CRIT_BOUNDARY_RULES
    }
    i_crit_boundary = min(1.0, len(crit_boundary_hits) / 2.0)

    m_survival = (
        0.0 if mutation_survival_rate is None else max(0.0, min(1.0, mutation_survival_rate))
    )

    if exploits:
        unpatched = sum(1 for e in exploits if e.gold_sanity_status != "verified_fix")
        h_redteam = unpatched / len(exploits)
    else:
        h_redteam = 0.0

    derived_rubric = 0.0
    rule_ids = {f.rule_id for f in findings}
    if "RULE-RUBR-002" in rule_ids:
        derived_rubric = 1.0
    elif "RULE-RUBR-001" in rule_ids or "RULE-RUBR-003" in rule_ids:
        derived_rubric = 0.5
    r_rubric = max(derived_rubric, max(0.0, min(1.0, rubric_score or 0.0)))

    score = 40.0 * i_crit_boundary + 25.0 * m_survival + 25.0 * h_redteam + 10.0 * r_rubric
    return round(min(100.0, max(0.0, score)), 2)


def compute_grade(
    findings: Sequence[AuditFinding],
    hackability_score: float,
    determinism: DeterminismMetrics | None = None,
    mutation_survival_rate: float | None = None,
    exploits: Sequence[ExploitTrace] = (),
) -> Literal["A", "B", "C", "D", "F"]:
    """Deterministic Certification Grade A-F decision table per SDD-04 section 2.2."""
    has_critical = any(f.severity == Severity.CRITICAL for f in findings)
    has_high = any(f.severity == Severity.HIGH for f in findings)
    has_medium_or_low = any(f.severity in {Severity.MEDIUM, Severity.LOW} for f in findings)
    has_unpatched_exploit = any(e.gold_sanity_status != "verified_fix" for e in exploits)

    oracle_failed = determinism is not None and determinism.oracle_pass_rate < 1.0
    noop_failed = determinism is not None and determinism.noop_pass_rate > 0.0
    is_flaky = determinism is not None and determinism.replay_variance > 0.0

    # Grade F: Critical breach, unpatched exploit, broken Oracle/No-op, or HackabilityScore > 50
    if (
        has_critical
        or has_unpatched_exploit
        or oracle_failed
        or noop_failed
        or hackability_score > 50.0
    ):
        return "F"

    # Grade D: 1+ HIGH findings, flakiness, or 25 < HackabilityScore <= 50
    if has_high or is_flaky or hackability_score > 25.0:
        return "D"

    # Grade C: 10 < HackabilityScore <= 25 (or non-zero mutation survival)
    if hackability_score > 10.0:
        return "C"

    # Grade B: MEDIUM/LOW advisories, 0 < HackabilityScore <= 10, or mutation survival > 0
    has_mutation_survival = mutation_survival_rate is not None and mutation_survival_rate > 0.0
    if has_medium_or_low or hackability_score > 0.0 or has_mutation_survival:
        return "B"

    return "A"
