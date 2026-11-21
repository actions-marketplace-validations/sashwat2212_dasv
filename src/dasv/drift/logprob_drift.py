"""Train-inference log-probability drift diagnostic hook (SDD-05 §3, RULE-DRFT-001)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

from dasv.core.errors import ConfigError
from dasv.core.schemas import AuditFinding, Severity

MitigationKind = Literal["none", "tis_clip", "icepop_mask", "check_quant_precision"]

_EXP_CLAMP = 50.0


class LogprobDriftStats(BaseModel):
    """Rollout-to-training log-probability mismatch metrics over policy action tokens."""

    step: int
    action_tokens: int
    mean_abs_delta: float
    p95_abs_delta: float
    p99_abs_delta: float
    max_abs_delta: float
    kl_estimate: float
    max_is_ratio: float
    min_is_ratio: float
    icepop_masked_frac: float
    recommended_mitigation: MitigationKind
    finding: AuditFinding | None = None


def _to_nested_floats(data: Any) -> list[float]:
    """Flatten 1D/2D Python lists or duck-typed torch/numpy tensors into a 1D float list."""
    raw = data
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ConfigError("Expected a 1D or 2D sequence/tensor of numeric values.")
    flat: list[float] = []
    for item in cast(Sequence[Any], raw):
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for sub in cast(Sequence[Any], item):
                flat.append(float(sub))
        else:
            flat.append(float(item))
    return flat


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def compute_logprob_drift(
    train_logprobs: Any,
    infer_logprobs: Any,
    action_mask: Any,
    *,
    step: int = 0,
    icepop_bounds: tuple[float, float] = (0.5, 2.0),
    mean_delta_threshold: float = 0.05,
    p99_delta_threshold: float = 0.35,
    icepop_frac_threshold: float = 0.05,
) -> LogprobDriftStats:
    """Compute Delta_t = log pi_train - log pi_infer stats strictly on `action_mask == 1` tokens.

    Accepts 1D `[T]` or 2D `[B, T]` Python lists or duck-typed PyTorch/NumPy tensors without
    requiring `torch` in core `dasv` dependencies.
    """
    low_bound, high_bound = icepop_bounds
    if low_bound <= 0.0 or high_bound <= low_bound:
        raise ConfigError(f"Invalid icepop_bounds {icepop_bounds}: require 0 < low < high.")

    tr = _to_nested_floats(train_logprobs)
    inf = _to_nested_floats(infer_logprobs)
    mask = _to_nested_floats(action_mask)
    if not (len(tr) == len(inf) == len(mask)):
        raise ConfigError(
            f"Shape mismatch: train_logprobs ({len(tr)}), infer_logprobs ({len(inf)}), "
            f"and action_mask ({len(mask)}) must contain the same number of elements."
        )

    deltas: list[float] = []
    abs_deltas: list[float] = []
    ratios: list[float] = []
    kl_terms: list[float] = []
    out_of_band = 0

    for t_lp, i_lp, m in zip(tr, inf, mask, strict=True):
        if m <= 0.5:
            continue
        delta = t_lp - i_lp
        deltas.append(delta)
        abs_deltas.append(abs(delta))
        clamped = max(-_EXP_CLAMP, min(_EXP_CLAMP, delta))
        ratio = math.exp(clamped)
        ratios.append(ratio)
        # Schulman k3 non-negative KL estimator for KL(pi_train || pi_infer)
        neg_clamped = max(-_EXP_CLAMP, min(_EXP_CLAMP, -delta))
        kl_terms.append(max(0.0, math.exp(neg_clamped) - 1.0 + delta))
        if ratio < low_bound or ratio > high_bound:
            out_of_band += 1

    if not deltas:
        return LogprobDriftStats(
            step=step,
            action_tokens=0,
            mean_abs_delta=0.0,
            p95_abs_delta=0.0,
            p99_abs_delta=0.0,
            max_abs_delta=0.0,
            kl_estimate=0.0,
            max_is_ratio=1.0,
            min_is_ratio=1.0,
            icepop_masked_frac=0.0,
            recommended_mitigation="none",
            finding=None,
        )

    n = len(deltas)
    sorted_abs = sorted(abs_deltas)
    mean_abs = sum(abs_deltas) / n
    p95_abs = _percentile(sorted_abs, 0.95)
    p99_abs = _percentile(sorted_abs, 0.99)
    max_abs = sorted_abs[-1]
    kl_est = sum(kl_terms) / n
    max_ratio = max(ratios)
    min_ratio = min(ratios)
    icepop_frac = out_of_band / n

    mitigation: MitigationKind = "none"
    severity: Severity | None = None
    if mean_abs >= 0.25 or icepop_frac >= 0.20:
        mitigation = "check_quant_precision"
        severity = Severity.HIGH
    elif icepop_frac >= icepop_frac_threshold or p99_abs >= p99_delta_threshold:
        mitigation = "icepop_mask"
        severity = Severity.MEDIUM
    elif mean_abs >= mean_delta_threshold:
        mitigation = "tis_clip"
        severity = Severity.MEDIUM

    finding: AuditFinding | None = None
    if severity is not None:
        finding = AuditFinding(
            rule_id="RULE-DRFT-001",
            title=f"Train-inference logprob drift detected at step {step} ({mitigation})",
            severity=severity,
            tier=3,
            source="dasv",
            evidence=(
                f"Step {step} ({n} action tokens): mean|delta|={mean_abs:.4f}, "
                f"p95={p95_abs:.4f}, p99={p99_abs:.4f}, KL_k3={kl_est:.4f}, "
                f"IS ratio in [{min_ratio:.3f}, {max_ratio:.3f}], "
                f"icepop_masked_frac={icepop_frac:.3f} outside [{low_bound}, {high_bound}]."
            ),
            remediation=(
                f"Recommended mitigation: `{mitigation}`. Align inference & training kernels "
                "(BF16/FP8 precision, RoPE/attention backends) or enable truncated importance "
                "sampling / IcePop masking before policy collapse."
            ),
        )

    return LogprobDriftStats(
        step=step,
        action_tokens=n,
        mean_abs_delta=round(mean_abs, 6),
        p95_abs_delta=round(p95_abs, 6),
        p99_abs_delta=round(p99_abs, 6),
        max_abs_delta=round(max_abs, 6),
        kl_estimate=round(kl_est, 6),
        max_is_ratio=round(max_ratio, 6),
        min_is_ratio=round(min_ratio, 6),
        icepop_masked_frac=round(icepop_frac, 6),
        recommended_mitigation=mitigation,
        finding=finding,
    )


class LogprobDriftTracker:
    """Lightweight trainer callback recording per-step `LogprobDriftStats` to JSONL."""

    def __init__(
        self,
        log_path: Path | None = None,
        *,
        icepop_bounds: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        self._log_path = log_path
        self._icepop_bounds = icepop_bounds
        self._history: list[LogprobDriftStats] = []

    @property
    def history(self) -> list[LogprobDriftStats]:
        return list(self._history)

    def record_step(
        self,
        train_logprobs: Any,
        infer_logprobs: Any,
        action_mask: Any,
        *,
        step: int,
    ) -> LogprobDriftStats:
        stats = compute_logprob_drift(
            train_logprobs,
            infer_logprobs,
            action_mask,
            step=step,
            icepop_bounds=self._icepop_bounds,
        )
        self._history.append(stats)
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(stats.model_dump(mode="json"), sort_keys=True) + "\n")
        return stats
