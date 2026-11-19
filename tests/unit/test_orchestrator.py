"""Unit tests for TieredAuditOrchestrator (SDD-01: Tiers 0->1, short-circuit, BudgetGuard)."""

from __future__ import annotations

import json
from pathlib import Path

from dasv.adapters.harbor import HarborAdapter
from dasv.core.config import DasvConfig
from dasv.core.errors import (
    EXIT_AUDIT_FAIL,
    EXIT_BUDGET_EXCEEDED,
    EXIT_PASS,
    EXIT_RUNTIME_ERROR,
    BudgetExceededError,
)
from dasv.core.orchestrator import AuditContext, TieredAuditOrchestrator, TierHookResult
from dasv.core.schemas import AuditFinding, AuditReport, Severity
from dasv.sandbox.janitor import ReapReport
from dasv.sandbox.provider import ExecResult, SandboxHandle
from tests.conftest import FakeSandboxProvider


def _make_clean_tier1_provider(*, oracle_passes: bool = True) -> FakeSandboxProvider:
    """FakeSandboxProvider simulating hardened runtime probes + clean-image Oracle/No-op grading."""
    container_applied_oracle: dict[str, bool] = {}

    def exec_handler(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (timeout_sec, workdir)
        joined = " ".join(cmd)
        if "os.access" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"exists": [], "writable": []}),
                stderr="",
                duration_ms=2.0,
            )
        if "egress_reachable" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"egress_reachable": [], "leaked_env_keys": []}),
                stderr="",
                duration_ms=2.0,
            )
        if "surviving" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"surviving": []}),
                stderr="",
                duration_ms=2.0,
            )
        if handle.plane == "execution":
            if "dasv_ep_" in joined or "solve.sh" in joined:
                container_applied_oracle[handle.container_id] = True
                return ExecResult(exit_code=0, stdout="solved\n", stderr="", duration_ms=5.0)
            if cmd[:2] == ["git", "diff"]:
                if container_applied_oracle.get(handle.container_id):
                    patch = (
                        "diff --git a/src/calculator.py b/src/calculator.py\n"
                        "--- a/src/calculator.py\n"
                        "+++ b/src/calculator.py\n"
                        "@@ -1 +1 @@\n-return 0\n+return a + b\n"
                    )
                    return ExecResult(exit_code=0, stdout=patch, stderr="", duration_ms=2.0)
                return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)
            return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

        # Verification plane: apply extracted patch and score test.sh
        if "git', 'apply'" in joined or "git apply" in joined:
            container_applied_oracle[handle.container_id] = True
            return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=2.0)
        if "test.sh" in joined or cmd[0] in {"sh", "pytest"}:
            if container_applied_oracle.get(handle.container_id) and oracle_passes:
                return ExecResult(exit_code=0, stdout="reward=1.0\n", stderr="", duration_ms=4.0)
            return ExecResult(exit_code=1, stdout="reward=0.0\n", stderr="fail\n", duration_ms=4.0)
        return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

    return FakeSandboxProvider(exec_handler=exec_handler, diff_entries=[])


async def test_orchestrator_tier0_and_tier1_happy_path(
    hardened_harbor_env: Path,
    tmp_path: Path,
) -> None:
    adapter = HarborAdapter()
    spec = adapter.load(hardened_harbor_env)
    provider = _make_clean_tier1_provider(oracle_passes=True)

    out_dir = tmp_path / "run_ok"
    sarif_path = out_dir / "report.sarif"
    html_path = out_dir / "report.html"

    ctx = AuditContext.create(
        config=DasvConfig(),
        adapter=adapter,
        env_spec=spec,
        out_dir=out_dir,
        provider=provider,
        native=False,
        oracle_replays=3,
        noop_replays=2,
    )
    res = await TieredAuditOrchestrator().execute(
        ctx,
        {0, 1},
        sarif_path=sarif_path,
        html_path=html_path,
    )

    assert res.exit_code == EXIT_PASS
    assert res.report.grade == "A"
    assert res.report.hackability_score == 0.0
    assert res.report.determinism is not None
    assert res.report.determinism.oracle_trials == 3
    assert res.report.determinism.oracle_pass_rate == 1.0
    assert res.report.determinism.noop_trials == 2
    assert res.report.determinism.noop_pass_rate == 0.0
    assert res.report.determinism.replay_variance == 0.0
    assert provider.active_handles == {}
    assert (out_dir / "report.json").is_file()
    assert (out_dir / "manifest.json").is_file()
    assert sarif_path.is_file()
    assert html_path.is_file()
    assert "hardened-calculator" in html_path.read_text(encoding="utf-8")


async def test_orchestrator_short_circuits_tier2_on_tier1_failure(
    hardened_harbor_env: Path,
    tmp_path: Path,
) -> None:
    adapter = HarborAdapter()
    spec = adapter.load(hardened_harbor_env)
    provider = _make_clean_tier1_provider(oracle_passes=False)

    tier2_called = False

    async def fake_tier2(ctx: AuditContext, findings: list[AuditFinding]) -> TierHookResult:
        nonlocal tier2_called
        _ = (ctx, findings)
        tier2_called = True
        return TierHookResult()

    ctx = AuditContext.create(
        config=DasvConfig(),
        adapter=adapter,
        env_spec=spec,
        out_dir=tmp_path / "run_sc",
        provider=provider,
        native=False,
        oracle_replays=2,
        noop_replays=1,
        short_circuit=True,
    )
    res = await TieredAuditOrchestrator(tier2_hook=fake_tier2).execute(ctx, {0, 1, 2})

    assert tier2_called is False
    assert res.short_circuited is True
    assert res.exit_code == EXIT_AUDIT_FAIL
    assert res.report.grade == "F"
    assert any(f.rule_id == "RULE-ORCH-001" for f in res.report.findings)
    assert provider.active_handles == {}


async def test_orchestrator_budget_exceeded_flushes_partial_report_and_reaps(
    hardened_harbor_env: Path,
    tmp_path: Path,
) -> None:
    adapter = HarborAdapter()
    spec = adapter.load(hardened_harbor_env)
    provider = _make_clean_tier1_provider(oracle_passes=True)

    async def exploding_tier2(ctx: AuditContext, findings: list[AuditFinding]) -> TierHookResult:
        _ = findings
        # Exceed hard $0.0001 cap via BudgetGuard
        await ctx.budget.reserve_or_raise("gpt-4o", 500_000, 100_000, is_local=False)
        raise BudgetExceededError("unreachable")

    cfg = DasvConfig()
    cfg = cfg.model_copy(update={"budget": cfg.budget.model_copy(update={"max_cost_usd": 0.0001})})
    out_dir = tmp_path / "run_budget"
    html_path = out_dir / "partial.html"

    ctx = AuditContext.create(
        config=cfg,
        adapter=adapter,
        env_spec=spec,
        out_dir=out_dir,
        provider=provider,
        native=False,
        oracle_replays=1,
        noop_replays=1,
    )
    res = await TieredAuditOrchestrator(tier2_hook=exploding_tier2).execute(
        ctx,
        {0, 1, 2},
        html_path=html_path,
    )

    assert res.budget_exceeded is True
    assert res.exit_code == EXIT_BUDGET_EXCEEDED
    assert any(f.rule_id == "RULE-BUDG-001" for f in res.report.findings)
    # Partial Tier 0/1 artifacts were still flushed to disk
    loaded = AuditReport.model_validate_json((out_dir / "report.json").read_text(encoding="utf-8"))
    assert loaded.determinism is not None
    assert loaded.determinism.oracle_pass_rate == 1.0
    assert html_path.is_file()
    assert provider.active_handles == {}


async def test_orchestrator_unhealthy_provider_exits_runtime_error_and_records_stubborn(
    hardened_harbor_env: Path,
    tmp_path: Path,
) -> None:
    adapter = HarborAdapter()
    spec = adapter.load(hardened_harbor_env)
    unhealthy = FakeSandboxProvider(healthy=False)

    out_dir = tmp_path / "run_err"
    ctx = AuditContext.create(
        config=DasvConfig(),
        adapter=adapter,
        env_spec=spec,
        out_dir=out_dir,
        provider=unhealthy,
        native=False,
    )

    async def fake_reap(run_id: str, *, grace_timeout_sec: float = 10.0) -> ReapReport:
        _ = grace_timeout_sec
        return ReapReport(run_id=run_id, stubborn_ids=["ctr-stuck-1"])

    ctx.janitor.reap_all = fake_reap  # type: ignore[method-assign]

    res = await TieredAuditOrchestrator().execute(ctx, {0, 1})
    assert res.exit_code == EXIT_RUNTIME_ERROR
    assert res.error_message is not None and "unavailable" in res.error_message.lower()
    assert any(
        f.rule_id == "RULE-HERM-005" and f.severity == Severity.HIGH for f in res.report.findings
    )
    assert (out_dir / "report.json").is_file()


async def test_orchestrator_tier1_with_mutations_populates_survival_rate(
    hardened_harbor_env: Path,
    tmp_path: Path,
) -> None:
    (hardened_harbor_env / "src").mkdir(exist_ok=True)
    (hardened_harbor_env / "src" / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    if a < 0:\n"
        "        raise ValueError('neg')\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    (hardened_harbor_env / "tests" / "test_calc.py").write_text(
        "def test_add() -> None:\n    assert 2 + 3 == 5\n",
        encoding="utf-8",
    )
    adapter = HarborAdapter()
    spec = adapter.load(hardened_harbor_env)
    provider = _make_clean_tier1_provider(oracle_passes=True)

    # Anti-patches fail clean-image grading when test.sh returns reward=0.0 on broken patches
    html_path = tmp_path / "mut_report.html"
    ctx = AuditContext.create(
        config=DasvConfig(),
        adapter=adapter,
        env_spec=spec,
        out_dir=tmp_path / "run_mut",
        provider=provider,
        native=False,
        oracle_replays=2,
        noop_replays=1,
        run_mutations=True,
    )
    res = await TieredAuditOrchestrator().execute(ctx, {0, 1}, html_path=html_path)
    assert res.report.mutation_survival_rate is not None
    assert "Mutation Survival" in html_path.read_text(encoding="utf-8")
    assert provider.active_handles == {}
