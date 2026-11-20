"""RLVR statistical estimators, grading engine, and Tier 3 signal profiler (SDD-04)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from dasv.profile.signal_profiler import (
        CurriculumEntry,
        RolloutEvaluator,
        SignalProfiler,
        TasksetProfileReport,
        build_signal_profile,
        build_tier3_profile_hook,
        discover_taskset_specs,
        export_curriculum,
        load_rewards_from_traces_jsonl,
    )

_LAZY_SIGNAL_EXPORTS = frozenset(
    {
        "CurriculumEntry",
        "RolloutEvaluator",
        "SignalProfiler",
        "TasksetProfileReport",
        "build_signal_profile",
        "build_tier3_profile_hook",
        "discover_taskset_specs",
        "export_curriculum",
        "load_rewards_from_traces_jsonl",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SIGNAL_EXPORTS:
        from dasv.profile import signal_profiler

        return getattr(signal_profiler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CurriculumEntry",
    "RolloutEvaluator",
    "SignalProfiler",
    "TasksetProfileReport",
    "binary_reward_variance",
    "build_signal_profile",
    "build_tier3_profile_hook",
    "classify_curriculum_bucket",
    "compute_grade",
    "compute_hackability_score",
    "discover_taskset_specs",
    "export_curriculum",
    "load_rewards_from_traces_jsonl",
    "pass_at_k",
    "pass_pow_k",
    "taskset_zero_grad_waste_pct",
    "zero_grad_group_prob",
]
