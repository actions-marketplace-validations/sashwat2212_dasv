"""Strict `dasv.toml` configuration loader (ADR-002: extra='forbid')."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dasv.core.errors import ConfigError


class _StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_StrictConfigModel):
    default_tiers: list[Literal[0, 1, 2, 3]] = Field(default_factory=lambda: [0, 1])
    fail_on: Literal["critical", "high", "medium", "low"] = "high"
    strict_arch_benchmark: bool = True


class SandboxConfig(_StrictConfigModel):
    runtime: Literal["docker", "modal"] = "docker"
    network_mode: Literal["none", "bridge"] = "none"
    cpus: float = Field(default=2.0, gt=0.0)
    memory: str = "2g"
    pids_limit: int = Field(default=256, ge=16)
    episode_timeout_sec: int = Field(default=120, ge=1)
    max_concurrent: int = Field(default=4, ge=1)


class DeterminismConfig(_StrictConfigModel):
    oracle_replays: int = Field(default=10, ge=1)
    noop_replays: int = Field(default=3, ge=1)


class BudgetConfig(_StrictConfigModel):
    max_cost_usd: float = Field(default=2.00, ge=0.0)
    max_tokens: int = Field(default=250_000, ge=1)
    max_redteam_turns: int = Field(default=12, ge=1)


class LlmConfig(_StrictConfigModel):
    provider: Literal["openai_compat"] = "openai_compat"
    base_url: str = "http://localhost:11434/v1"
    attacker_model: str = "qwen2.5-coder:7b"
    fixer_model: str = "qwen2.5-coder:7b"


class DasvConfig(_StrictConfigModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    determinism: DeterminismConfig = Field(default_factory=DeterminismConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)


def load_config(path: Path | None = None) -> DasvConfig:
    """Load and strictly validate a `dasv.toml` configuration file."""
    target: Path | None = path
    if target is None:
        default_path = Path.cwd() / "dasv.toml"
        if not default_path.is_file():
            return DasvConfig()
        target = default_path
    elif not target.is_file():
        raise ConfigError(f"Configuration file not found: {target}")

    try:
        raw_text = target.read_text(encoding="utf-8")
        raw_data = tomllib.loads(raw_text)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Failed to parse TOML configuration '{target}': {exc}") from exc

    try:
        return DasvConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in '{target}': {exc}") from exc
