"""Tiered Audit Orchestrator with short-circuiting, BudgetGuard, and ContainerJanitor (SDD-01)."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from dasv.adapters.base import EnvironmentAdapter
from dasv.core.budget import BudgetGuard
from dasv.core.config import DasvConfig
from dasv.core.errors import (
    EXIT_AUDIT_FAIL,
    EXIT_BUDGET_EXCEEDED,
    EXIT_PASS,
    EXIT_RUNTIME_ERROR,
    BudgetExceededError,
    DasvError,
    NativeDelegateError,
    RuntimeSandboxError,
)
from dasv.core.manifest import build_run_manifest
from dasv.core.schemas import (
    AuditFinding,
    AuditReport,
    DeterminismMetrics,
    EnvSpec,
    ExploitTrace,
    Severity,
    SignalProfile,
    meets_fail_threshold,
)
from dasv.profile.stats import compute_grade, compute_hackability_score
from dasv.reporting.html_renderer import write_html_report
from dasv.reporting.json_writer import write_manifest_json, write_report_json
from dasv.reporting.sarif_writer import write_sarif
from dasv.sandbox.docker_provider import DockerSandboxProvider
from dasv.sandbox.janitor import ContainerJanitor, ReapReport
from dasv.sandbox.probes import run_runtime_boundary_probes, run_tier0_static_probes
from dasv.sandbox.provider import ContainerConfig, SandboxProvider
from dasv.verifier.clean_grader import CleanImageGrader, GradingContext
from dasv.verifier.determinism import DeterminismRunner

logger = logging.getLogger(__name__)


@dataclass
class TierHookResult:
    """Optional extension payload returned by pluggable Tier 2/3 stage hooks."""

    findings: list[AuditFinding] = field(default_factory=list[AuditFinding])
    exploits: list[ExploitTrace] = field(default_factory=list[ExploitTrace])
    mutation_survival_rate: float | None = None
    signal_profile: SignalProfile | None = None
    models_used: list[str] = field(default_factory=list[str])


TierHook = Callable[["AuditContext", list[AuditFinding]], Awaitable[TierHookResult]]


@dataclass
class AuditContext:
    """Execution state shared across Tier 0 -> Tier 3 audit stages."""

    config: DasvConfig
    adapter: EnvironmentAdapter
    env_spec: EnvSpec
    budget: BudgetGuard
    janitor: ContainerJanitor
    semaphore: asyncio.Semaphore
    run_id: str = field(default_factory=lambda: f"dasv-{uuid4().hex[:12]}")
    out_dir: Path | None = None
    provider: SandboxProvider | None = None
    cli_args: list[str] = field(default_factory=list[str])
    seed: int = 42
    native: bool = True
    require_native: bool = False
    oracle_replays: int | None = None
    noop_replays: int | None = None
    short_circuit: bool = True

    @classmethod
    def create(
        cls,
        *,
        config: DasvConfig,
        adapter: EnvironmentAdapter,
        env_spec: EnvSpec,
        out_dir: Path | None = None,
        provider: SandboxProvider | None = None,
        cli_args: list[str] | None = None,
        seed: int = 42,
        native: bool = True,
        require_native: bool = False,
        oracle_replays: int | None = None,
        noop_replays: int | None = None,
        short_circuit: bool = True,
        run_id: str | None = None,
    ) -> AuditContext:
        """Build a ready-to-run AuditContext wired with BudgetGuard, Janitor, and Semaphore."""
        budget = BudgetGuard(
            max_cost_usd=config.budget.max_cost_usd,
            max_tokens=config.budget.max_tokens,
            base_url=config.llm.base_url,
        )
        janitor = ContainerJanitor(provider=provider)
        semaphore = asyncio.Semaphore(config.sandbox.max_concurrent)
        return cls(
            config=config,
            adapter=adapter,
            env_spec=env_spec,
            budget=budget,
            janitor=janitor,
            semaphore=semaphore,
            run_id=run_id or f"dasv-{uuid4().hex[:12]}",
            out_dir=out_dir,
            provider=provider,
            cli_args=list(cli_args or []),
            seed=seed,
            native=native,
            require_native=require_native,
            oracle_replays=oracle_replays,
            noop_replays=noop_replays,
            short_circuit=short_circuit,
        )


@dataclass
class OrchestrationResult:
    """Final output bundle returned by `TieredAuditOrchestrator.execute()`."""

    report: AuditReport
    exit_code: int
    reap_report: ReapReport | None = None
    short_circuited: bool = False
    budget_exceeded: bool = False
    aborted_by_signal: bool = False
    error_message: str | None = None


def _find_dockerfile(spec: EnvSpec) -> Path | None:
    for candidate in (
        spec.root_path / "environment" / "Dockerfile",
        spec.root_path / "Dockerfile",
    ):
        if candidate.is_file():
            return candidate
    return None


class TieredAuditOrchestrator:
    """Cost-ordered async audit orchestrator with guaranteed cleanup and partial report flush."""

    def __init__(
        self,
        *,
        tier2_hook: TierHook | None = None,
        tier3_hook: TierHook | None = None,
        reap_grace_sec: float = 10.0,
    ) -> None:
        self._tier2_hook = tier2_hook
        self._tier3_hook = tier3_hook
        self._reap_grace_sec = reap_grace_sec

    async def _run_tier0(self, ctx: AuditContext) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        findings.extend(run_tier0_static_probes(ctx.env_spec))
        findings.extend(ctx.adapter.static_findings(ctx.env_spec))
        if ctx.native or ctx.require_native:
            native_findings = await ctx.adapter.run_native_check(
                ctx.env_spec,
                require_native=ctx.require_native,
            )
            findings.extend(native_findings)
        return findings

    async def _prepare_tier1_spec(
        self,
        ctx: AuditContext,
        provider: SandboxProvider,
        oci_digests: dict[str, str],
    ) -> EnvSpec:
        spec = ctx.env_spec
        if spec.image_ref or spec.image_digest:
            if spec.image_ref and spec.image_digest:
                oci_digests[spec.image_ref] = spec.image_digest
            return spec

        dockerfile = _find_dockerfile(spec)
        if dockerfile is None:
            raise RuntimeSandboxError(
                f"EnvSpec '{spec.task_id}' declares neither image_ref/image_digest nor a "
                "buildable Dockerfile; cannot execute Tier 1 container checks."
            )

        tag = f"dasv-{spec.task_id.lower().replace(' ', '-')}:{ctx.run_id}"
        built_digest = await provider.build_image(
            dockerfile.parent,
            dockerfile,
            tag,
            platform=None,
        )
        oci_digests[tag] = built_digest
        return spec.model_copy(update={"image_ref": tag, "image_digest": built_digest})

    async def _run_tier1(
        self,
        ctx: AuditContext,
        oci_digests: dict[str, str],
    ) -> tuple[list[AuditFinding], DeterminismMetrics, bool, str | None]:
        provider: SandboxProvider = ctx.provider or DockerSandboxProvider(janitor=ctx.janitor)
        health = await provider.health_check()
        if not health.healthy:
            raise RuntimeSandboxError(
                f"Sandbox runtime '{health.runtime}' is unavailable for Tier 1: {health.detail}. "
                "Start Docker/OrbStack or run with `--tiers 0` for static-only checks."
            )

        effective_spec = await self._prepare_tier1_spec(ctx, provider, oci_digests)
        findings: list[AuditFinding] = []

        # 1. Runtime boundary & hermeticity probes inside an isolated Execution Plane container.
        probe_image = effective_spec.image_ref or effective_spec.image_digest or ""
        probe_cfg = ContainerConfig(
            image=probe_image,
            network_mode="none",
            cpus=ctx.config.sandbox.cpus,
            memory=ctx.config.sandbox.memory,
            pids_limit=ctx.config.sandbox.pids_limit,
            plane="execution",
        )
        probe_handle = await provider.start(probe_cfg, run_id=ctx.run_id)
        try:
            if probe_handle.image_digest and probe_image:
                oci_digests[probe_image] = probe_handle.image_digest
            runtime_findings = await run_runtime_boundary_probes(
                provider,
                probe_handle,
                effective_spec,
                timeout_sec=min(30.0, float(ctx.config.sandbox.episode_timeout_sec)),
            )
            findings.extend(runtime_findings)
        finally:
            await provider.stop_and_remove(probe_handle)

        # 2. N-trial Oracle + M-trial No-op clean-image determinism replays.
        grader = CleanImageGrader(provider, sandbox_config=ctx.config.sandbox)
        grading_ctx = GradingContext(
            grader,
            run_id=ctx.run_id,
            timeout_sec=float(ctx.config.sandbox.episode_timeout_sec),
        )
        runner = DeterminismRunner(max_concurrent=ctx.config.sandbox.max_concurrent)
        oracle_n = (
            ctx.oracle_replays
            if ctx.oracle_replays is not None
            else ctx.config.determinism.oracle_replays
        )
        noop_n = (
            ctx.noop_replays
            if ctx.noop_replays is not None
            else ctx.config.determinism.noop_replays
        )
        det_result = await runner.run(
            ctx.adapter,
            effective_spec,
            grading_ctx,
            oracle_trials=oracle_n,
            noop_trials=noop_n,
            semaphore=ctx.semaphore,
        )
        findings.extend(det_result.findings)
        return findings, det_result.metrics, det_result.passed, health.server_arch

    async def execute(
        self,
        ctx: AuditContext,
        tiers: set[int],
        *,
        fail_on: Severity | None = None,
        sarif_path: Path | None = None,
        html_path: Path | None = None,
        install_signal_handlers: bool = False,
    ) -> OrchestrationResult:
        """Run requested tiers in ascending order, flush reports, and reap all containers."""
        threshold = fail_on or Severity(ctx.config.project.fail_on)
        findings: list[AuditFinding] = []
        determinism: DeterminismMetrics | None = None
        exploits: list[ExploitTrace] = []
        mutation_survival_rate: float | None = None
        signal_profile: SignalProfile | None = None
        models_used: list[str] = []
        oci_digests: dict[str, str] = {}
        if ctx.env_spec.image_ref and ctx.env_spec.image_digest:
            oci_digests[ctx.env_spec.image_ref] = ctx.env_spec.image_digest

        docker_arch: str | None = None
        short_circuited = False
        budget_exceeded = False
        aborted_by_signal = False
        fatal_error: DasvError | None = None
        reap_report: ReapReport | None = None

        current_task = asyncio.current_task()
        registered_signals: list[signal.Signals] = []
        loop = asyncio.get_running_loop()

        if install_signal_handlers and current_task is not None:

            def _on_signal(sig: signal.Signals) -> None:
                nonlocal aborted_by_signal
                aborted_by_signal = True
                logger.warning(
                    "Received %s; cancelling audit run %s cleanly.", sig.name, ctx.run_id
                )
                current_task.cancel()

            for sig in (signal.SIGINT, signal.SIGTERM):
                with suppress(NotImplementedError, RuntimeError, ValueError):
                    loop.add_signal_handler(sig, _on_signal, sig)
                    registered_signals.append(sig)

        try:
            # Tier 0: $0 static probes + adapter checks + native delegation.
            if 0 in tiers:
                findings.extend(await self._run_tier0(ctx))

            # Tier 1: $0 runtime boundary probes + N-trial Clean-Image Determinism.
            if 1 in tiers:
                t1_findings, determinism, det_passed, docker_arch = await self._run_tier1(
                    ctx, oci_digests
                )
                findings.extend(t1_findings)
                if ctx.short_circuit and not det_passed and ({2, 3} & tiers):
                    short_circuited = True
                    findings.append(
                        AuditFinding(
                            rule_id="RULE-ORCH-001",
                            title="Short-circuited higher tiers after Tier 1 determinism failure",
                            severity=Severity.INFO,
                            tier=1,
                            source="dasv",
                            evidence=(
                                "Tier 1 determinism gate failed (Oracle < 100%, No-op > 0%, or "
                                "non-zero replay variance); skipped Tier 2/3 LLM spend."
                            ),
                            remediation=(
                                "Fix Oracle/No-op determinism first, or pass `--no-short-circuit` "
                                "to force Tier 2/3 execution."
                            ),
                        )
                    )

            # Tier 2: Budget-capped adversarial red-team / rubric stress tests (M2 hook).
            if 2 in tiers and not short_circuited:
                if self._tier2_hook is not None:
                    res2 = await self._tier2_hook(ctx, findings)
                    findings.extend(res2.findings)
                    exploits.extend(res2.exploits)
                    if res2.mutation_survival_rate is not None:
                        mutation_survival_rate = res2.mutation_survival_rate
                    models_used.extend(res2.models_used)
                else:
                    findings.append(self._deferred_tier_finding(2))

            # Tier 3: Budget-capped RLVR signal profiling (M2/M3 hook).
            if 3 in tiers and not short_circuited:
                if self._tier3_hook is not None:
                    res3 = await self._tier3_hook(ctx, findings)
                    findings.extend(res3.findings)
                    if res3.signal_profile is not None:
                        signal_profile = res3.signal_profile
                    models_used.extend(res3.models_used)
                else:
                    findings.append(self._deferred_tier_finding(3))

        except BudgetExceededError as exc:
            budget_exceeded = True
            fatal_error = exc
            findings.append(
                AuditFinding(
                    rule_id="RULE-BUDG-001",
                    title="Hard budget circuit breaker triggered",
                    severity=Severity.INFO,
                    tier=2,
                    source="dasv",
                    evidence=str(exc),
                    remediation=(
                        "Partial Tier 0-1 results were preserved. Raise `--max-cost-usd` or "
                        "`--max-tokens`, or switch to a local $0 Ollama/MLX endpoint."
                    ),
                )
            )
        except (NativeDelegateError, RuntimeSandboxError, DasvError) as exc:
            fatal_error = exc
        except asyncio.CancelledError:
            aborted_by_signal = True
        finally:
            for sig in registered_signals:
                with suppress(NotImplementedError, RuntimeError, ValueError):
                    loop.remove_signal_handler(sig)

            # Guaranteed container cleanup on normal exit, exception, budget abort, or signal.
            try:
                reap_report = await asyncio.shield(
                    ctx.janitor.reap_all(ctx.run_id, grace_timeout_sec=self._reap_grace_sec)
                )
            except Exception as exc:
                logger.error("ContainerJanitor cleanup failed for run %s: %s", ctx.run_id, exc)
                reap_report = ReapReport(run_id=ctx.run_id, errors=[str(exc)])

            if reap_report is not None and reap_report.stubborn_ids:
                findings.append(
                    AuditFinding(
                        rule_id="RULE-HERM-005",
                        title="Stubborn container(s) survived teardown",
                        severity=Severity.HIGH,
                        tier=1,
                        source="dasv",
                        evidence=(
                            f"ContainerJanitor failed to remove container(s): "
                            f"{', '.join(reap_report.stubborn_ids)}"
                        ),
                        remediation=(
                            f"Inspect and remove lingering containers manually with "
                            f"`docker ps -a --filter label=dasv.run_id={ctx.run_id}`."
                        ),
                    )
                )

            budget_snap = await ctx.budget.snapshot()
            manifest = build_run_manifest(
                ctx.env_spec.root_path,
                cli_args=ctx.cli_args,
                docker_server_arch=docker_arch,
                oci_image_digests=oci_digests,
                models_used=sorted(set(models_used)),
                rng_seed=ctx.seed,
                total_tokens=budget_snap.committed_tokens,
                total_cost_usd=budget_snap.committed_cost_usd,
                run_id=ctx.run_id,
            )
            hack_score = compute_hackability_score(
                findings,
                mutation_survival_rate=mutation_survival_rate,
                exploits=exploits,
            )
            grade = compute_grade(
                findings,
                hackability_score=hack_score,
                determinism=determinism,
                mutation_survival_rate=mutation_survival_rate,
                exploits=exploits,
            )
            report = AuditReport(
                manifest=manifest,
                env_spec=ctx.env_spec,
                grade=grade,
                hackability_score=hack_score,
                findings=findings,
                determinism=determinism,
                mutation_survival_rate=mutation_survival_rate,
                exploits=exploits,
                signal_profile=signal_profile,
            )

            self._flush_outputs(
                report,
                out_dir=ctx.out_dir,
                sarif_path=sarif_path,
                html_path=html_path,
            )

        if budget_exceeded:
            exit_code = EXIT_BUDGET_EXCEEDED
        elif fatal_error is not None:
            exit_code = fatal_error.exit_code
        elif aborted_by_signal:
            exit_code = EXIT_RUNTIME_ERROR
        elif meets_fail_threshold(findings, threshold) or report.grade == "F":
            exit_code = EXIT_AUDIT_FAIL
        else:
            exit_code = EXIT_PASS

        return OrchestrationResult(
            report=report,
            exit_code=exit_code,
            reap_report=reap_report,
            short_circuited=short_circuited,
            budget_exceeded=budget_exceeded,
            aborted_by_signal=aborted_by_signal,
            error_message=str(fatal_error) if fatal_error is not None else None,
        )

    @staticmethod
    def _deferred_tier_finding(tier_num: int) -> AuditFinding:
        milestone = "M2" if tier_num == 2 else "M2/M3"
        return AuditFinding(
            rule_id="RULE-STAGE-001",
            title=f"Tier {tier_num} runner scheduled for Stage 6 ({milestone})",
            severity=Severity.INFO,
            tier=0,
            source="dasv",
            evidence=(
                f"Tier {tier_num} execution requires Stage 6 {milestone} LLM/signal runners; "
                "completed lower tiers were preserved."
            ),
            remediation="Run with `--tiers 0,1` for v0.1 checks or enable Tier 2/3 in M2/M3.",
        )

    @staticmethod
    def _flush_outputs(
        report: AuditReport,
        *,
        out_dir: Path | None,
        sarif_path: Path | None,
        html_path: Path | None,
    ) -> None:
        if out_dir is not None:
            write_report_json(report, out_dir / "report.json")
            write_manifest_json(report.manifest, out_dir / "manifest.json")
        if sarif_path is not None:
            write_sarif(report, sarif_path)
        if html_path is not None:
            write_html_report(report, html_path)
