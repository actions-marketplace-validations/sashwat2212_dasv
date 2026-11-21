"""Train-inference logprob drift hook & LoRA-RLVR recipe validation (SDD-05, FEAT-18)."""

from __future__ import annotations

from dasv.drift.logprob_drift import (
    LogprobDriftStats,
    LogprobDriftTracker,
    MitigationKind,
    compute_logprob_drift,
)
from dasv.drift.recipe_config import (
    REQUIRED_ALL_LINEAR_MODULES,
    LoRAAblationConfig,
    LoRARecipeConfig,
    ModelRecipeConfig,
    OptimizerRecipeConfig,
    RLObjectiveRecipeConfig,
    load_ablation_recipe_config,
)

__all__ = [
    "LogprobDriftStats",
    "LogprobDriftTracker",
    "LoRAAblationConfig",
    "LoRARecipeConfig",
    "MitigationKind",
    "ModelRecipeConfig",
    "OptimizerRecipeConfig",
    "REQUIRED_ALL_LINEAR_MODULES",
    "RLObjectiveRecipeConfig",
    "compute_logprob_drift",
    "load_ablation_recipe_config",
]
