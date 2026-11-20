"""Unit & CLI tests for Tier 3 RLVR signal profiler & curriculum export (FEAT-17)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from dasv.adapters.harbor import HarborAdapter
from dasv.cli.app import app
from dasv.core.budget import BudgetGuard
from dasv.core.config import DasvConfig
from dasv.core.errors import (
    EXIT_BUDGET_EXCEEDED,
    EXIT_CONFIG_ERROR,
    EXIT_PASS,
    EXIT_RUNTIME_ERROR,
    BudgetExceededError,
)
from dasv.core.orchestrator import AuditContext, TieredAuditOrchestrator
from dasv.core.schemas import AuditReport, CurriculumBucket, EnvSpec, EnvStandard
from dasv.profile import (
    SignalProfiler,
    build_signal_profile,
    build_tier3_profile_hook,
    discover_taskset_specs,
    export_curriculum,
    load_rewards_from_traces_jsonl,
    pass_at_k,
    pass_pow_k,
    taskset_zero_grad_waste_pct,
)

runner = CliRunner()


def _make_harbor_task(parent: Path, name: str) -> Path:
    d = parent / name
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "task.toml").write_text(
        f'[metadata]\nname = "{name}"\n\n[environment]\nnetwork_mode = "no-network"\n',
        encoding="utf-8",
    )
    (d / "instruction.md").write_text(f"Solve {name}.\n", encoding="utf-8")
    digest = "0123456789abcdef" * 4
    (d / "environment" / "Dockerfile").write_text(
        f"FROM python:3.12-slim@sha256:{digest}\nWORKDIR /app\n",
        encoding="utf-8",
    )
    (d / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return d


def test_build_signal_profile_exact_estimators_and_buckets(tmp_path: Path) -> None:
    spec = EnvSpec(
        task_id="task-gold",
        standard=EnvStandard.HARBOR,
        root_path=tmp_path,
        instruction_text="Solve.",
    )
    # 2 successes out of 8 rollouts -> pass@1 = 0.25 (GOLDILOCKS)
    prof, entry, findings = build_signal_profile(
        spec,
        base_model="qwen2.5:3b",
        rewards=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert prof.bucket == CurriculumBucket.GOLDILOCKS
    assert prof.pass_at_1 == pytest.approx(0.25)
    assert prof.pass_at_k[4] == pytest.approx(pass_at_k(8, 2, 4), rel=1e-5)
    assert prof.pass_at_k[8] == pytest.approx(1.0)
    assert prof.pass_pow_k[4] == pytest.approx(pass_pow_k(8, 2, 4), rel=1e-5)
    assert prof.advantage_variance == pytest.approx(0.25 * 0.75)
    assert prof.zero_grad_risk is False
    assert entry.recommended_sampler_weight > 0.0
    assert findings == []

    # 0/8 -> STARVED_ZERO_GRAD + RULE-SIGL-001
    starved, s_entry, s_findings = build_signal_profile(
        spec, base_model="qwen2.5:3b", rewards=[0.0] * 8
    )
    assert starved.bucket == CurriculumBucket.STARVED_ZERO_GRAD
    assert starved.zero_grad_risk is True
    assert s_entry.recommended_sampler_weight == 0.0
    assert len(s_findings) == 1 and s_findings[0].rule_id == "RULE-SIGL-001"

    # 8/8 -> TRIVIAL_ZERO_GRAD + RULE-SIGL-001
    trivial, t_entry, t_findings = build_signal_profile(
        spec, base_model="qwen2.5:3b", rewards=[1.0] * 8
    )
    assert trivial.bucket == CurriculumBucket.TRIVIAL_ZERO_GRAD
    assert trivial.zero_grad_risk is True
    assert t_entry.recommended_sampler_weight == 0.0
    assert len(t_findings) == 1 and t_findings[0].rule_id == "RULE-SIGL-001"

    # Non-zero replay variance -> FLAKY + RULE-SIGL-001
    flaky, f_entry, f_findings = build_signal_profile(
        spec,
        base_model="qwen2.5:3b",
        rewards=[1.0, 0.0, 1.0, 0.0],
        replay_variance=0.05,
    )
    assert flaky.bucket == CurriculumBucket.FLAKY
    assert f_entry.recommended_sampler_weight == 0.0
    assert len(f_findings) == 1


async def test_signal_profiler_taskset_zero_grad_waste_and_exports(tmp_path: Path) -> None:
    taskset_dir = tmp_path / "taskset"
    taskset_dir.mkdir()
    _make_harbor_task(taskset_dir, "t1-goldilocks")
    _make_harbor_task(taskset_dir, "t2-trivial")
    _make_harbor_task(taskset_dir, "t3-starved")
    _make_harbor_task(taskset_dir, "t4-flaky")

    discovered = discover_taskset_specs(taskset_dir)
    specs = [spec for _adp, spec in discovered]
    assert [s.task_id for s in specs] == [
        "t1-goldilocks",
        "t2-trivial",
        "t3-starved",
        "t4-flaky",
    ]

    reward_table: dict[str, list[float]] = {
        "t1-goldilocks": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2/8 = 0.25
        "t2-trivial": [1.0] * 8,  # 8/8 = 1.0
        "t3-starved": [0.0] * 8,  # 0/8 = 0.0
        "t4-flaky": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 1/8 = 0.125, flaky
    }

    async def evaluator(spec: EnvSpec, rollout_idx: int, model: str, run_id: str) -> float:
        _ = (model, run_id)
        return reward_table[spec.task_id][rollout_idx - 1]

    jsonl_out = tmp_path / "out" / "calibrated.jsonl"
    yaml_out = tmp_path / "out" / "calibrated.yaml"
    out_dir = tmp_path / "out" / "profile_dir"

    profiler = SignalProfiler(rollout_evaluator=evaluator)
    report = await profiler.profile_specs(
        specs,
        base_model="qwen2.5:3b",
        rollouts_g=8,
        replay_variances={"t4-flaky": 0.09},
        out_dir=out_dir,
        export_curriculum_path=jsonl_out,
    )
    export_curriculum(report, yaml_out)

    expected_waste = taskset_zero_grad_waste_pct([0.25, 1.0, 0.0, 0.125], 8)
    assert report.zero_grad_waste_pct == pytest.approx(expected_waste, abs=0.02)
    assert report.bucket_counts["goldilocks"] == 1
    assert report.bucket_counts["trivial_zero_grad"] == 1
    assert report.bucket_counts["starved_zero_grad"] == 1
    assert report.bucket_counts["flaky"] == 1
    assert any(f.rule_id == "RULE-SIGL-002" for f in report.findings)
    assert (out_dir / "profile.json").is_file()

    # JSONL export strictly contains only GOLDILOCKS task by default
    jsonl_lines = [
        json.loads(ln) for ln in jsonl_out.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(jsonl_lines) == 1
    assert jsonl_lines[0]["task_id"] == "t1-goldilocks"
    assert jsonl_lines[0]["bucket"] == "goldilocks"

    # YAML export contains aggregate metadata + selected GOLDILOCKS list
    yaml_data: dict[str, Any] = yaml.safe_load(yaml_out.read_text(encoding="utf-8"))
    assert yaml_data["selected_count"] == 1
    assert yaml_data["tasks"][0]["task_id"] == "t1-goldilocks"


async def test_signal_profiler_budget_abort_preserves_partial_results(tmp_path: Path) -> None:
    specs = [
        EnvSpec(
            task_id=f"t{i}",
            standard=EnvStandard.HARBOR,
            root_path=tmp_path,
            instruction_text="",
        )
        for i in (1, 2)
    ]

    async def exploding_eval(spec: EnvSpec, rollout_idx: int, model: str, run_id: str) -> float:
        _ = (rollout_idx, model, run_id)
        if spec.task_id == "t2":
            raise BudgetExceededError("hard token ceiling reached")
        return 1.0 if rollout_idx == 1 else 0.0

    profiler = SignalProfiler(
        budget=BudgetGuard(max_cost_usd=1.0, max_tokens=1000, base_url="http://localhost:11434/v1"),
        rollout_evaluator=exploding_eval,
    )
    report = await profiler.profile_specs(specs, base_model="local", rollouts_g=4)
    assert report.budget_exceeded is True
    assert len(report.tasks) == 1
    assert report.tasks[0].task_id == "t1"
    assert any(f.rule_id == "RULE-BUDG-001" for f in report.findings)


async def test_tier3_orchestrator_hook_renders_signal_svg_curve_in_html(
    hardened_harbor_env: Path,
    tmp_path: Path,
) -> None:
    adapter = HarborAdapter()
    spec = adapter.load(hardened_harbor_env)
    out_dir = tmp_path / "t3_orch"
    html_path = out_dir / "report.html"

    async def goldilocks_eval(s: EnvSpec, rollout_idx: int, model: str, run_id: str) -> float:
        _ = (s, model, run_id)
        return 1.0 if rollout_idx in {1, 2} else 0.0  # 2/8 = 25%

    hook = build_tier3_profile_hook(
        base_model="qwen2.5:3b",
        rollouts_g=8,
        rollout_evaluator=goldilocks_eval,
    )
    ctx = AuditContext.create(
        config=DasvConfig(),
        adapter=adapter,
        env_spec=spec,
        out_dir=out_dir,
        native=False,
    )
    res = await TieredAuditOrchestrator(tier3_hook=hook).execute(ctx, {0, 3}, html_path=html_path)
    assert res.exit_code == EXIT_PASS
    assert res.report.signal_profile is not None
    assert res.report.signal_profile.bucket == CurriculumBucket.GOLDILOCKS
    assert (out_dir / "profile" / "profile.json").is_file()
    assert (out_dir / "profile" / "calibrated_curriculum.jsonl").is_file()
    html_text = html_path.read_text(encoding="utf-8")
    assert "RLVR Policy-Gradient Signal Profile (Tier 3)" in html_text
    assert "pass@k (cyan) vs pass^k (amber)" in html_text
    assert "http://" not in html_text and "https://" not in html_text


def test_cli_profile_and_audit_tier3_with_traces_and_errors(
    hardened_harbor_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    traces_file = tmp_path / "traces.jsonl"
    # 2/8 successes for hardened-calculator -> GOLDILOCKS
    records = [
        {"episode_id": f"ep-{i}", "task_id": "hardened-calculator", "reward": 1.0 if i < 2 else 0.0}
        for i in range(8)
    ]
    traces_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    loaded = load_rewards_from_traces_jsonl(traces_file)
    assert len(loaded["hardened-calculator"]) == 8

    # 1. `dasv profile --traces` exports calibrated JSONL cleanly at $0 (exit 0)
    curr_out = tmp_path / "calibrated.jsonl"
    prof_out = tmp_path / "prof_dir"
    res = runner.invoke(
        app,
        [
            "profile",
            str(hardened_harbor_env),
            "--model",
            "qwen2.5:3b",
            "--rollouts",
            "8",
            "--traces",
            str(traces_file),
            "--export-curriculum",
            str(curr_out),
            "--out-dir",
            str(prof_out),
        ],
    )
    assert res.exit_code == EXIT_PASS, res.stdout + res.stderr
    assert curr_out.is_file()
    exported = [json.loads(ln) for ln in curr_out.read_text(encoding="utf-8").splitlines() if ln]
    assert len(exported) == 1
    assert exported[0]["bucket"] == "goldilocks"

    # 2. `dasv audit --tiers 0,3 --traces ... --html ...` wires Tier 3 end-to-end!
    audit_out = tmp_path / "audit_t3"
    audit_html = audit_out / "report.html"
    res_t3 = runner.invoke(
        app,
        [
            "audit",
            str(hardened_harbor_env),
            "--tiers",
            "0,3",
            "--no-native",
            "--traces",
            str(traces_file),
            "--profile-model",
            "qwen2.5:3b",
            "--rollouts",
            "8",
            "--out-dir",
            str(audit_out),
            "--html",
            str(audit_html),
        ],
    )
    assert res_t3.exit_code == EXIT_PASS, res_t3.stdout + res_t3.stderr
    rep = AuditReport.model_validate_json((audit_out / "report.json").read_text(encoding="utf-8"))
    assert rep.signal_profile is not None
    assert rep.signal_profile.pass_at_1 == pytest.approx(0.25)
    assert "pass@k (cyan) vs pass^k (amber)" in audit_html.read_text(encoding="utf-8")

    # 3. Invalid curriculum bucket -> EXIT_CONFIG_ERROR (2)
    res_bad_b = runner.invoke(
        app,
        ["profile", str(hardened_harbor_env), "--include-buckets", "nonexistent_bucket"],
    )
    assert res_bad_b.exit_code == EXIT_CONFIG_ERROR

    # 4. Live mode without Docker -> EXIT_RUNTIME_ERROR (4)
    async def missing_docker(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_docker)
    res_no_docker = runner.invoke(app, ["profile", str(hardened_harbor_env)])
    assert res_no_docker.exit_code == EXIT_RUNTIME_ERROR
    _ = EXIT_BUDGET_EXCEEDED
