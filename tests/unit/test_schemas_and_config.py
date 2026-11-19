"""Unit tests for Pydantic v2 schemas, strict dasv.toml config, and RunManifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from dasv.core.config import DasvConfig, load_config
from dasv.core.errors import EXIT_CONFIG_ERROR, ConfigError
from dasv.core.manifest import build_run_manifest, compute_dataset_sha256
from dasv.core.schemas import (
    AuditFinding,
    AuditReport,
    CurriculumBucket,
    DeterminismMetrics,
    EnvSpec,
    EnvStandard,
    ExploitTrace,
    Severity,
    SignalProfile,
    meets_fail_threshold,
)


def test_load_example_config() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_config(repo_root / "dasv.toml.example")
    assert isinstance(cfg, DasvConfig)
    assert cfg.project.default_tiers == [0, 1]
    assert cfg.project.fail_on == "high"
    assert cfg.sandbox.network_mode == "none"
    assert cfg.budget.max_cost_usd == 2.00
    assert cfg.llm.base_url == "http://localhost:11434/v1"


def test_config_rejects_extra_fields(tmp_path: Path) -> None:
    bad_toml = tmp_path / "dasv.toml"
    bad_toml.write_text("[project]\nfail_on = 'high'\nunknown_flag = true\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_config(bad_toml)
    assert exc_info.value.exit_code == EXIT_CONFIG_ERROR


def test_config_rejects_missing_and_malformed_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.toml")

    broken = tmp_path / "broken.toml"
    broken.write_text("[project\nfail_on = 'high'\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(broken)


def test_audit_report_roundtrip_and_manifest_determinism(hardened_harbor_env: Path) -> None:
    sha1 = compute_dataset_sha256(hardened_harbor_env)
    sha2 = compute_dataset_sha256(hardened_harbor_env)
    assert sha1 == sha2
    assert len(sha1) == 64

    manifest = build_run_manifest(hardened_harbor_env, cli_args=["audit", "--tiers", "0"])
    spec = EnvSpec(
        task_id="hardened-calculator",
        standard=EnvStandard.HARBOR,
        root_path=hardened_harbor_env,
        instruction_text="Implement add(a, b).",
    )
    report = AuditReport(
        manifest=manifest,
        env_spec=spec,
        grade="A",
        hackability_score=0.0,
        findings=[
            AuditFinding(
                rule_id="RULE-NATIVE-001",
                title="Advisory",
                severity=Severity.INFO,
                tier=0,
                source="native_delegate",
                evidence="harbor CLI absent",
                remediation="Install harbor",
            )
        ],
        determinism=DeterminismMetrics(
            oracle_trials=10,
            oracle_pass_rate=1.0,
            noop_trials=3,
            noop_pass_rate=0.0,
            replay_variance=0.0,
            pass_at_1=1.0,
            pass_pow_k={3: 1.0, 10: 1.0},
            latency_p50_ms=120.0,
            latency_p95_ms=180.0,
            peak_memory_mb=64.0,
        ),
        exploits=[
            ExploitTrace(
                exploit_id="exp-1",
                task_id="hardened-calculator",
                hack_class="test_tampering",
                reward_obtained=0.0,
                steps=[],
                gold_sanity_status="verified_fix",
            )
        ],
        signal_profile=SignalProfile(
            task_id="hardened-calculator",
            base_model="qwen2.5:3b",
            rollouts_g=8,
            pass_at_1=0.25,
            pass_at_k={1: 0.25, 8: 0.90},
            pass_pow_k={3: 0.015625},
            advantage_variance=0.1875,
            zero_grad_risk=False,
            bucket=CurriculumBucket.GOLDILOCKS,
        ),
    )

    serialized = report.model_dump_json()
    restored = AuditReport.model_validate_json(serialized)
    assert restored == report
    assert not meets_fail_threshold(report.findings, Severity.HIGH)
    assert meets_fail_threshold(report.findings, Severity.INFO)
