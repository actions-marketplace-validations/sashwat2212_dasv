"""Unit tests for train-inference logprob drift hook & RULE-DRFT-001 (SDD-05 §3)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from dasv.core.errors import ConfigError
from dasv.core.schemas import Severity
from dasv.drift import LogprobDriftTracker, compute_logprob_drift


class _DuckTensor:
    """Minimal tensor stub mimicking `torch.Tensor.detach().cpu().tolist()` without torch."""

    def __init__(self, data: list[list[float]]) -> None:
        self._data = data

    def detach(self) -> _DuckTensor:
        return self

    def cpu(self) -> _DuckTensor:
        return self

    def tolist(self) -> list[list[float]]:
        return self._data


def test_compute_logprob_drift_zero_and_action_mask_exclusion() -> None:
    """Zero drift on policy action tokens returns none even when masked prompt tokens differ."""
    train_lp = [[-5.0, -4.0, -1.2, -0.8, -0.5]]
    # Prompt tokens (mask=0) have huge mismatch, policy tokens (mask=1) match identically.
    infer_lp = [[-0.1, -9.9, -1.2, -0.8, -0.5]]
    action_mask = [[0, 0, 1, 1, 1]]

    stats = compute_logprob_drift(_DuckTensor(train_lp), _DuckTensor(infer_lp), action_mask, step=7)
    assert stats.step == 7
    assert stats.action_tokens == 3
    assert stats.mean_abs_delta == 0.0
    assert stats.p95_abs_delta == 0.0
    assert stats.p99_abs_delta == 0.0
    assert stats.kl_estimate == 0.0
    assert stats.max_is_ratio == 1.0
    assert stats.min_is_ratio == 1.0
    assert stats.icepop_masked_frac == 0.0
    assert stats.recommended_mitigation == "none"
    assert stats.finding is None


def test_compute_logprob_drift_mitigation_ladder_and_overflow_safety() -> None:
    """Verify tis_clip, icepop_mask, check_quant_precision, RULE-DRFT-001, and exp clamping."""
    # 1. Mild uniform drift (~0.08) -> tis_clip (MEDIUM)
    tr_mild = [-1.0] * 10
    inf_mild = [-1.08] * 10
    mask_10 = [1] * 10
    s_mild = compute_logprob_drift(tr_mild, inf_mild, mask_10, step=10)
    assert s_mild.mean_abs_delta == pytest.approx(0.08, abs=1e-5)
    assert s_mild.recommended_mitigation == "tis_clip"
    assert s_mild.finding is not None
    assert s_mild.finding.rule_id == "RULE-DRFT-001"
    assert s_mild.finding.severity == Severity.MEDIUM

    # 2. Tail drift (1/10 tokens = 10% with delta = +0.9 -> exp(0.9)=2.46 > 2.0) -> icepop_mask
    tr_tail = [-1.0] * 9 + [-0.1]
    inf_tail = [-1.0] * 10
    s_tail = compute_logprob_drift(tr_tail, inf_tail, mask_10, step=20)
    assert s_tail.icepop_masked_frac == pytest.approx(0.10, abs=1e-5)
    assert s_tail.max_is_ratio == pytest.approx(math.exp(0.9), rel=1e-4)
    assert s_tail.recommended_mitigation == "icepop_mask"
    assert s_tail.finding is not None
    assert s_tail.finding.severity == Severity.MEDIUM

    # 3. Severe quantization mismatch (mean_abs_delta >= 0.25) -> check_quant_precision (HIGH)
    tr_sev = [-0.5] * 8
    inf_sev = [-1.2] * 8
    s_sev = compute_logprob_drift(tr_sev, inf_sev, [1] * 8, step=30)
    assert s_sev.recommended_mitigation == "check_quant_precision"
    assert s_sev.finding is not None
    assert s_sev.finding.severity == Severity.HIGH

    # 4. Extreme FP8 blowup (delta = 500) must clamp cleanly without OverflowError
    s_ext = compute_logprob_drift([0.0], [-500.0], [1], step=40)
    assert math.isfinite(s_ext.max_is_ratio)
    assert s_ext.recommended_mitigation == "check_quant_precision"

    # 5. All-masked sequence -> 0 action tokens, none
    s_empty = compute_logprob_drift([-1.0, -2.0], [-5.0, -6.0], [0, 0], step=50)
    assert s_empty.action_tokens == 0
    assert s_empty.recommended_mitigation == "none"

    # 6. Shape mismatch & invalid icepop_bounds raise ConfigError
    with pytest.raises(ConfigError):
        compute_logprob_drift([-1.0, -2.0], [-1.0], [1, 1])
    with pytest.raises(ConfigError):
        compute_logprob_drift([-1.0], [-1.0], [1], icepop_bounds=(2.0, 0.5))


def test_logprob_drift_tracker_flushes_jsonl(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "drift_metrics.jsonl"
    tracker = LogprobDriftTracker(log_path=log_file)
    tracker.record_step([-1.0, -1.0], [-1.0, -1.0], [1, 1], step=1)
    tracker.record_step([-0.2, -0.2], [-1.0, -1.0], [1, 1], step=2)

    assert len(tracker.history) == 2
    lines = [json.loads(ln) for ln in log_file.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 2
    assert lines[0]["recommended_mitigation"] == "none"
    assert lines[1]["recommended_mitigation"] == "check_quant_precision"
