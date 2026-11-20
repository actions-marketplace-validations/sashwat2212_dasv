"""RedTeamRunner & Tier 2 Orchestrator Hook (Hacker -> Fixer -> GoldSanityGate, SDD-02)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from dasv.adapters.base import EnvironmentAdapter
from dasv.core.budget import BudgetGuard
from dasv.core.config import SandboxConfig
from dasv.core.errors import BudgetExceededError, ConfigError, RuntimeSandboxError
from dasv.core.schemas import AuditFinding, EnvSpec, ExploitTrace, Severity
from dasv.redteam.fixer_agent import FixerAgent
from dasv.redteam.hacker_agent import (
    HACK_CLASSES,
    HackClass,
    HackerAgent,
    format_exploit_jsonl_line,
)
from dasv.redteam.llm_client import LLMClient, UnifiedLLMClient
from dasv.sandbox.docker_provider import DockerSandboxProvider
from dasv.sandbox.provider import SandboxProvider
from dasv.verifier.clean_grader import ensure_container_image, find_env_dockerfile
from dasv.verifier.gold_sanity import GoldSanityGate, apply_unified_diff_to_dir

if TYPE_CHECKING:
    from dasv.core.orchestrator import AuditContext, TierHook, TierHookResult


class RedTeamReport(BaseModel):
    """Structured output of a Tier 2 adversarial Hacker-Fixer red-team run."""

    task_id: str
    attacker_model: str
    fixer_model: str | None = None
    hack_classes_attempted: list[str] = Field(default_factory=list[str])
    exploits: list[ExploitTrace] = Field(default_factory=list[ExploitTrace])
    findings: list[AuditFinding] = Field(default_factory=list[AuditFinding])
    applied_fix_ids: list[str] = Field(default_factory=list[str])
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    budget_exceeded: bool = False

    @property
    def unpatched_exploit_count(self) -> int:
        return sum(1 for e in self.exploits if e.gold_sanity_status != "verified_fix")

    @property
    def passed(self) -> bool:
        return self.unpatched_exploit_count == 0 and not self.budget_exceeded


def parse_hack_classes(raw: str | None) -> list[HackClass]:
    """Parse a comma-separated list of hack classes (or return all 6 when None)."""
    if raw is None or not raw.strip():
        return list(HACK_CLASSES)
    selected: list[HackClass] = []
    valid_map: dict[str, HackClass] = {hc: hc for hc in HACK_CLASSES}
    for part in raw.split(","):
        cleaned = part.strip().lower()
        if not cleaned:
            continue
        if cleaned not in valid_map:
            allowed = ", ".join(HACK_CLASSES)
            raise ConfigError(f"Invalid hack class '{cleaned}'. Allowed hack classes: {allowed}.")
        hc = valid_map[cleaned]
        if hc not in selected:
            selected.append(hc)
    if not selected:
        raise ConfigError("At least one hack class must be specified.")
    return selected


class RedTeamRunner:
    """Orchestrates HackerAgent -> FixerAgent -> 3-Invariant GoldSanityGate under BudgetGuard."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        provider: SandboxProvider,
        budget: BudgetGuard,
        sandbox_config: SandboxConfig | None = None,
    ) -> None:
        self._llm = llm
        self._provider = provider
        self._budget = budget
        self._sandbox_config = sandbox_config or SandboxConfig()
        self._hacker = HackerAgent(llm, provider, sandbox_config=self._sandbox_config)
        self._fixer = FixerAgent(llm)
        self._gsg = GoldSanityGate(provider, sandbox_config=self._sandbox_config)

    async def run(
        self,
        spec: EnvSpec,
        adapter: EnvironmentAdapter,
        *,
        attacker_model: str,
        fixer_model: str | None = None,
        hack_classes: Sequence[HackClass] = HACK_CLASSES,
        max_turns: int = 12,
        propose_fix: bool = False,
        gold_sanity_replays: int = 5,
        apply_verified_fixes: bool = False,
        run_id: str = "dasv-redteam",
        out_dir: Path | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> RedTeamReport:
        """Execute adversarial attacks, optional Fixer + GoldSanityGate, and flush artifacts."""
        effective_spec = await ensure_container_image(spec, self._provider, run_id)
        effective_fixer_model = fixer_model or attacker_model
        attempted: list[str] = []
        exploits: list[ExploitTrace] = []
        findings: list[AuditFinding] = []
        applied_fix_ids: list[str] = []
        budget_tripped = False

        for idx, hack_class in enumerate(hack_classes, start=1):
            attempted.append(hack_class)
            try:
                attempt = await self._hacker.attack(
                    effective_spec,
                    hack_class=hack_class,
                    model=attacker_model,
                    run_id=f"{run_id}-h{idx}",
                    max_turns=max_turns,
                )
            except BudgetExceededError as exc:
                budget_tripped = True
                findings.append(
                    AuditFinding(
                        rule_id="RULE-BUDG-001",
                        title="Hard budget circuit breaker triggered during red-team loop",
                        severity=Severity.INFO,
                        tier=2,
                        source="dasv",
                        evidence=str(exc),
                        remediation="Partial red-team exploits preserved; raise budget or use "
                        "local Ollama/MLX endpoint.",
                    )
                )
                break

            if not attempt.confirmed_exploit or attempt.exploit_trace is None:
                continue

            trace = attempt.exploit_trace
            if propose_fix and attempt.extracted_artifact is not None:
                try:
                    candidate_diff = await self._fixer.propose_fix(
                        effective_spec,
                        trace,
                        model=effective_fixer_model,
                    )
                except BudgetExceededError as exc:
                    budget_tripped = True
                    findings.append(
                        AuditFinding(
                            rule_id="RULE-BUDG-001",
                            title="Hard budget circuit breaker triggered during Fixer generation",
                            severity=Severity.INFO,
                            tier=2,
                            source="dasv",
                            evidence=str(exc),
                            remediation="Partial red-team exploits preserved.",
                        )
                    )
                    exploits.append(trace)
                    findings.append(self._exploit_finding(trace))
                    break

                if candidate_diff is not None:
                    gsg_res = await self._gsg.verify_patch(
                        effective_spec,
                        adapter,
                        attempt.extracted_artifact,
                        candidate_diff,
                        run_id=f"{run_id}-gsg-{idx}",
                        oracle_replays=gold_sanity_replays,
                        noop_replays=2,
                        semaphore=semaphore,
                    )
                    trace = trace.model_copy(
                        update={
                            "proposed_fix_diff": candidate_diff,
                            "gold_sanity_status": gsg_res.status,
                        }
                    )
                    if apply_verified_fixes and gsg_res.status == "verified_fix":
                        applied_ok, _ = apply_unified_diff_to_dir(
                            effective_spec.root_path, candidate_diff
                        )
                        if applied_ok:
                            applied_fix_ids.append(trace.exploit_id)
                else:
                    trace = trace.model_copy(update={"gold_sanity_status": "rejected_fix"})

            exploits.append(trace)
            findings.append(self._exploit_finding(trace))

        snap = await self._budget.snapshot()
        report = RedTeamReport(
            task_id=spec.task_id,
            attacker_model=attacker_model,
            fixer_model=effective_fixer_model if propose_fix else None,
            hack_classes_attempted=attempted,
            exploits=exploits,
            findings=findings,
            applied_fix_ids=applied_fix_ids,
            total_tokens=snap.committed_tokens,
            total_cost_usd=snap.committed_cost_usd,
            budget_exceeded=budget_tripped,
        )

        if out_dir is not None:
            self._write_artifacts(out_dir, report)
        return report

    @staticmethod
    def _exploit_finding(trace: ExploitTrace) -> AuditFinding:
        if trace.gold_sanity_status == "verified_fix":
            return AuditFinding(
                rule_id="RULE-REDT-001",
                title=f"Adversarial exploit closed by verified Fixer patch ({trace.hack_class})",
                severity=Severity.INFO,
                tier=2,
                source="dasv",
                evidence=(
                    f"Exploit {trace.exploit_id} achieved reward={trace.reward_obtained:.2f} via "
                    f"'{trace.hack_class}', and a Fixer patch passed all 3 GoldSanityGate "
                    "invariants (status=verified_fix)."
                ),
                remediation="Apply the verified diff under fixes/ and re-run Tier 0-1 CI gates.",
            )
        return AuditFinding(
            rule_id="RULE-REDT-001",
            title=f"Confirmed adversarial reward-hacking exploit ({trace.hack_class})",
            severity=Severity.CRITICAL,
            tier=2,
            source="dasv",
            evidence=(
                f"HackerAgent achieved reward={trace.reward_obtained:.2f} via strategy "
                f"'{trace.hack_class}' (exploit_id={trace.exploit_id}, "
                f"gold_sanity={trace.gold_sanity_status})."
            ),
            remediation=(
                "Harden the verifier assertions and container boundaries to reject this exploit "
                "trace, then re-verify with GoldSanityGate."
            ),
        )

    @staticmethod
    def _write_artifacts(out_dir: Path, report: RedTeamReport) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "redteam.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

        jsonl_lines = [format_exploit_jsonl_line(ex) for ex in report.exploits]
        (out_dir / "exploits.jsonl").write_text(
            ("\n".join(jsonl_lines) + "\n") if jsonl_lines else "",
            encoding="utf-8",
        )

        fixes_dir = out_dir / "fixes"
        for ex in report.exploits:
            if ex.proposed_fix_diff:
                fixes_dir.mkdir(parents=True, exist_ok=True)
                (fixes_dir / f"{ex.exploit_id}.diff").write_text(
                    ex.proposed_fix_diff, encoding="utf-8"
                )


def build_tier2_redteam_hook(
    *,
    llm: LLMClient | None = None,
    attacker_model: str | None = None,
    fixer_model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    hack_classes: Sequence[HackClass] = HACK_CLASSES,
    propose_fix: bool = False,
    gold_sanity_replays: int = 5,
) -> TierHook:
    """Create a Tier 2 hook plugging `RedTeamRunner` directly into `TieredAuditOrchestrator`."""
    from dasv.core.orchestrator import TierHookResult
    from dasv.rubric.prorubric_audit import audit_rubric_static

    async def _hook(ctx: AuditContext, _prior_findings: list[AuditFinding]) -> TierHookResult:
        _targets, rubric_findings, _gate_specs = audit_rubric_static(
            ctx.env_spec.root_path,
            spec=ctx.env_spec,
        )
        has_container_target = bool(
            ctx.env_spec.image_ref
            or ctx.env_spec.image_digest
            or find_env_dockerfile(ctx.env_spec) is not None
        )
        if not has_container_target and ctx.env_spec.rubric_type in {"rubric_llm", "hybrid"}:
            return TierHookResult(findings=rubric_findings)

        provider: SandboxProvider = ctx.provider or DockerSandboxProvider(janitor=ctx.janitor)
        health = await provider.health_check()
        if not health.healthy:
            raise RuntimeSandboxError(
                f"Sandbox runtime '{health.runtime}' is unavailable for Tier 2: {health.detail}. "
                "Start Docker/OrbStack to execute Tier 2 red-team sandboxes."
            )
        effective_base_url = base_url or ctx.config.llm.base_url
        effective_attacker = attacker_model or ctx.config.llm.attacker_model
        effective_fixer = fixer_model or ctx.config.llm.fixer_model
        client: LLMClient = llm or UnifiedLLMClient(
            base_url=effective_base_url,
            budget=ctx.budget,
            api_key_env=api_key_env,
        )
        runner = RedTeamRunner(
            llm=client,
            provider=provider,
            budget=ctx.budget,
            sandbox_config=ctx.config.sandbox,
        )
        rt_report = await runner.run(
            ctx.env_spec,
            ctx.adapter,
            attacker_model=effective_attacker,
            fixer_model=effective_fixer,
            hack_classes=hack_classes,
            max_turns=ctx.config.budget.max_redteam_turns,
            propose_fix=propose_fix,
            gold_sanity_replays=gold_sanity_replays,
            run_id=f"{ctx.run_id}-t2",
            out_dir=(ctx.out_dir / "redteam") if ctx.out_dir is not None else None,
            semaphore=ctx.semaphore,
        )
        models = [effective_attacker]
        if propose_fix and effective_fixer not in models:
            models.append(effective_fixer)
        return TierHookResult(
            findings=[*rubric_findings, *rt_report.findings],
            exploits=rt_report.exploits,
            models_used=models,
            budget_exceeded=rt_report.budget_exceeded,
        )

    return _hook
