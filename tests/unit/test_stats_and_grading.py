"""Unit tests for unbiased pass@k, pass^k, RLVR signal stats, HackabilityScore & Grade A-F."""

from __future__ import annotations

import pytest

from dasv.core.schemas import (
    AuditFinding,
    CurriculumBucket,
    DeterminismMetrics,
    ExploitTrace,
    Severity,
)
from dasv.profile.stats import (
    binary_reward_variance,
    classify_curriculum_bucket,
    compute_grade,
    compute_hackability_score,
    pass_at_k,
    pass_pow_k,
    taskset_zero_grad_waste_pct,
    zero_grad_group_prob,
)


def _finding(rule_id: str, severity: Severity) -> AuditFinding:
    return AuditFinding(
        rule_id=rule_id,
        title="t",
        severity=severity,
        tier=0,
        source="dasv",
        evidence="e",
        remediation="r",
    )


def test_pass_at_k_exact_and_large_n() -> None:
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 8, 5) == 1.0  # n - c = 2 < k = 5
    assert pass_at_k(10, 3, 2) == pytest.approx(8.0 / 15.0)
    # Large n without integer factorial overflow
    val = pass_at_k(1000, 50, 10)
    assert 0.35 < val < 0.45

    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)
    with pytest.raises(ValueError):
        pass_at_k(5, 1, 6)


def test_pass_pow_k_variance_and_zero_grad() -> None:
    assert pass_pow_k(10, 10, 10) == pytest.approx(1.0)
    assert pass_pow_k(10, 9, 10) == pytest.approx(0.9**10)
    assert binary_reward_variance(0.5) == pytest.approx(0.25)
    assert binary_reward_variance(0.0) == 0.0
    assert zero_grad_group_prob(0.0, 8) == pytest.approx(1.0)
    assert zero_grad_group_prob(1.0, 8) == pytest.approx(1.0)
    assert zero_grad_group_prob(0.5, 8) == pytest.approx(2.0 * (0.5**8))
    assert taskset_zero_grad_waste_pct([0.0, 1.0], 8) == pytest.approx(100.0)


def test_curriculum_bucketing() -> None:
    assert classify_curriculum_bucket(0.25, 0.90, replay_variance=0.05) == CurriculumBucket.FLAKY
    assert classify_curriculum_bucket(0.0, 0.0) == CurriculumBucket.STARVED_ZERO_GRAD
    assert classify_curriculum_bucket(0.95, 1.0) == CurriculumBucket.TRIVIAL_ZERO_GRAD
    assert classify_curriculum_bucket(0.20, 0.85) == CurriculumBucket.GOLDILOCKS


def test_hackability_score_and_grade_boundaries() -> None:
    # Grade A: clean (INFO advisories permitted)
    info_only = [_finding("RULE-NATIVE-001", Severity.INFO)]
    assert compute_hackability_score(info_only) == 0.0
    assert compute_grade(info_only, 0.0) == "A"

    # Grade B: MEDIUM advisory only
    med = [_finding("RULE-HERM-004", Severity.MEDIUM)]
    assert compute_grade(med, 0.0) == "B"

    # Grade C: HackabilityScore in (10, 25] with no CRITICAL/HIGH
    score_c = compute_hackability_score([], mutation_survival_rate=0.6)
    assert score_c == pytest.approx(15.0)
    assert compute_grade([], score_c, mutation_survival_rate=0.6) == "C"

    # Grade D: HIGH finding or flakiness
    high = [_finding("RULE-HERM-001", Severity.HIGH)]
    assert compute_grade(high, 0.0) == "D"
    flaky_det = DeterminismMetrics(
        oracle_trials=10,
        oracle_pass_rate=1.0,
        noop_trials=3,
        noop_pass_rate=0.0,
        replay_variance=0.04,
        pass_at_1=1.0,
        pass_pow_k={10: 0.35},
        latency_p50_ms=50.0,
        latency_p95_ms=80.0,
        peak_memory_mb=32.0,
    )
    assert compute_grade([], 0.0, determinism=flaky_det) == "D"

    # Grade F: CRITICAL boundary breach, unpatched exploit, or broken oracle
    crit = [
        _finding("RULE-BNDY-001", Severity.CRITICAL),
        _finding("RULE-BNDY-002", Severity.CRITICAL),
    ]
    assert compute_hackability_score(crit) == pytest.approx(40.0)
    assert compute_grade(crit, 40.0) == "F"

    unpatched = [
        ExploitTrace(
            exploit_id="e1",
            task_id="t1",
            hack_class="test_tampering",
            reward_obtained=1.0,
            steps=[],
            gold_sanity_status="not_run",
        )
    ]
    assert compute_hackability_score([], exploits=unpatched) == pytest.approx(25.0)
    assert compute_grade([], 25.0, exploits=unpatched) == "F"

    # Verified fix zeroes red-team contribution
    verified = [unpatched[0].model_copy(update={"gold_sanity_status": "verified_fix"})]
    assert compute_hackability_score([], exploits=verified) == 0.0
    assert compute_grade([], 0.0, exploits=verified) == "A"
