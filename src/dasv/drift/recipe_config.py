"""Pydantic validator for *LoRA Without Regret* Dr. GRPO/DAPO ablation recipes (SDD-05 §2)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dasv.core.errors import ConfigError

REQUIRED_ALL_LINEAR_MODULES: frozenset[str] = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


class ModelRecipeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    engine: str = "unsloth + trl"


class LoRARecipeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_modules: list[str]
    r: int = Field(ge=1, le=32)
    lora_alpha: int = Field(ge=1)
    lora_dropout: float = Field(ge=0.0, le=0.5, default=0.0)

    @model_validator(mode="after")
    def _enforce_all_linear_modules(self) -> LoRARecipeConfig:
        missing = sorted(REQUIRED_ALL_LINEAR_MODULES - set(self.target_modules))
        if missing:
            raise ValueError(
                "LoRA Without Regret Rule 1 violated: target_modules must include ALL linear "
                f"attention + MLP layers; missing: {', '.join(missing)}."
            )
        return self


class OptimizerRecipeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(gt=0.0)
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(ge=0.0, le=0.5, default=0.05)
    prompts_per_step: int = Field(ge=1, lt=32)
    group_size_g: int = Field(ge=2, default=8)
    gradient_accumulation_steps: int = Field(ge=1, default=1)

    @model_validator(mode="after")
    def _enforce_effective_prompt_batch(self) -> OptimizerRecipeConfig:
        effective = self.prompts_per_step * self.gradient_accumulation_steps
        if effective >= 32:
            raise ValueError(
                "LoRA Without Regret Rule 4 violated: effective prompts_per_step "
                f"({effective}) must stay strictly < 32."
            )
        return self


class RLObjectiveRecipeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: Literal["dr_grpo", "dapo", "grpo"] = "dr_grpo"
    clip_low: float = Field(gt=0.0, le=0.5, default=0.20)
    clip_high: float = Field(gt=0.0, le=0.8, default=0.28)
    dynamic_sampling: bool = True
    kl_beta: float = Field(ge=0.0, default=0.001)
    max_steps: int = Field(ge=1, default=200)
    seed: int = 42

    @model_validator(mode="after")
    def _enforce_clip_higher(self) -> RLObjectiveRecipeConfig:
        if self.clip_high < self.clip_low:
            raise ValueError(
                f"Invalid clip bounds: clip_high ({self.clip_high}) < clip_low ({self.clip_low})."
            )
        return self


class LoRAAblationConfig(BaseModel):
    """Top-level schema enforcing *LoRA Without Regret* + Dr. GRPO/DAPO guardrails."""

    model_config = ConfigDict(extra="forbid")

    model: ModelRecipeConfig
    lora: LoRARecipeConfig
    optimizer: OptimizerRecipeConfig
    rl_objective: RLObjectiveRecipeConfig


def load_ablation_recipe_config(path: Path) -> LoRAAblationConfig:
    """Load and strictly validate a `recipes/lora_dr_grpo_ablation/config.yaml` file."""
    if not path.is_file():
        raise ConfigError(f"Ablation recipe config not found: '{path}'.")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return LoRAAblationConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"Invalid LoRA ablation recipe config '{path}': {exc}") from exc
