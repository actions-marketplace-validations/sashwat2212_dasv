"""Frozen Pydantic v2 domain schemas per 02-PRD-FEATURES.md."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class EnvStandard(StrEnum):
    HARBOR = "harbor"
    OPENENV = "openenv"
    VERIFIERS_V1 = "verifiers_v1"
    HUD = "hud"


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def meets_fail_threshold(findings: Sequence[AuditFinding], fail_on: Severity) -> bool:
    """Return True if any finding meets or exceeds the configured fail_on severity."""
    threshold = SEVERITY_RANK[fail_on]
    return any(SEVERITY_RANK[f.severity] >= threshold for f in findings)


class CurriculumBucket(StrEnum):
    GOLDILOCKS = "goldilocks"
    TRIVIAL_ZERO_GRAD = "trivial_zero_grad"
    STARVED_ZERO_GRAD = "starved_zero_grad"
    FLAKY = "flaky"


class EnvSpec(BaseModel):
    task_id: str
    standard: EnvStandard
    root_path: Path
    instruction_text: str
    image_ref: str | None = None
    image_digest: str | None = None
    has_oracle: bool = False
    oracle_entrypoint: str | None = None
    verifier_paths: list[str] = Field(default_factory=list)
    rubric_type: Literal["deterministic", "rubric_llm", "hybrid"] = "deterministic"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditFinding(BaseModel):
    rule_id: str
    title: str
    severity: Severity
    tier: Literal[0, 1, 2, 3]
    source: Literal["dasv", "native_delegate"]
    file_path: str | None = None
    line_number: int | None = None
    evidence: str
    remediation: str


class DeterminismMetrics(BaseModel):
    oracle_trials: int
    oracle_pass_rate: float
    noop_trials: int
    noop_pass_rate: float
    replay_variance: float
    pass_at_1: float
    pass_pow_k: dict[int, float]
    latency_p50_ms: float
    latency_p95_ms: float
    peak_memory_mb: float
    emulated_arch_warning: bool = False


class ExploitTrace(BaseModel):
    exploit_id: str
    task_id: str
    hack_class: str
    reward_obtained: float
    steps: list[dict[str, Any]]
    proposed_fix_diff: str | None = None
    gold_sanity_status: Literal["not_run", "verified_fix", "rejected_fix"] = "not_run"


class SignalProfile(BaseModel):
    task_id: str
    base_model: str
    rollouts_g: int
    pass_at_1: float
    pass_at_k: dict[int, float]
    pass_pow_k: dict[int, float]
    advantage_variance: float
    zero_grad_risk: bool
    bucket: CurriculumBucket


class RunManifest(BaseModel):
    run_id: str
    dasv_version: str
    timestamp_utc: str
    git_sha: str | None
    host_os: str
    host_arch: str
    docker_server_arch: str | None
    oci_image_digests: dict[str, str]
    dataset_sha256: str
    models_used: list[str]
    rng_seed: int
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    cli_args: list[str]


class AuditReport(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    manifest: RunManifest
    env_spec: EnvSpec
    grade: Literal["A", "B", "C", "D", "F"]
    hackability_score: float | None = None
    findings: list[AuditFinding] = Field(default_factory=list[AuditFinding])
    determinism: DeterminismMetrics | None = None
    mutation_survival_rate: float | None = None
    exploits: list[ExploitTrace] = Field(default_factory=list[ExploitTrace])
    signal_profile: SignalProfile | None = None
