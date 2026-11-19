"""N-Trial Oracle, No-op & pass^k Determinism Runner (SDD-02, SDD-04, FEAT-10)."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from dasv.adapters.base import EnvironmentAdapter
from dasv.core.schemas import AuditFinding, DeterminismMetrics, EnvSpec, Severity
from dasv.profile.stats import pass_at_k, pass_pow_k


class TrialRecord(BaseModel):
    """Single isolated trial outcome for Oracle or No-op replay."""

    kind: Literal["oracle", "noop"]
    trial_index: int
    reward: float
    passed: bool
    duration_ms: float
    peak_memory_mb: float | None = None
    emulated_arch: bool = False
    tamper_attempts: list[str] = Field(default_factory=list)
    error: str | None = None


class DeterminismResult(BaseModel):
    """Aggregated Tier 1 determinism metrics, trial records, and audit findings."""

    metrics: DeterminismMetrics
    findings: list[AuditFinding] = Field(default_factory=list[AuditFinding])
    trials: list[TrialRecord] = Field(default_factory=list[TrialRecord])
    passed: bool


def _percentile(values: Sequence[float], q: float) -> float:
    """Compute linear-interpolated percentile q in [0, 100] over non-empty values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(ordered[low], 3)
    frac = rank - low
    return round(ordered[low] + frac * (ordered[high] - ordered[low]), 3)


def _population_variance(values: Sequence[float]) -> float:
    """Exact population variance (0.0 for <= 1 samples or identical rewards)."""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return 0.0 if var < 1e-12 else round(var, 8)


def _compute_pass_pow_checkpoints(n: int, c: int) -> dict[int, float]:
    """Compute pass^k reliability checkpoints for k in {1, 3, 5, 10, n} where k <= n."""
    if n < 1:
        return {}
    ks = sorted({k for k in (1, 3, 5, 10, n) if 1 <= k <= n})
    return {k: round(pass_pow_k(n, c, k), 6) for k in ks}


class DeterminismRunner:
    """Executes bounded-concurrency N-trial Oracle and M-trial No-op certification runs."""

    def __init__(self, *, max_concurrent: int = 4) -> None:
        self._max_concurrent = max(1, max_concurrent)

    async def _run_single_trial(
        self,
        adapter: EnvironmentAdapter,
        spec: EnvSpec,
        sandbox: Any,
        *,
        kind: Literal["oracle", "noop"],
        trial_index: int,
        semaphore: asyncio.Semaphore,
    ) -> TrialRecord:
        async with semaphore:
            t0 = time.perf_counter()
            try:
                if kind == "oracle":
                    reward, details = await adapter.run_oracle(spec, sandbox)
                else:
                    reward, details = await adapter.run_noop(spec, sandbox)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                reported_ms = details.get("duration_ms")
                duration_ms = (
                    float(reported_ms)
                    if isinstance(reported_ms, (int, float))
                    and not isinstance(reported_ms, bool)
                    and float(reported_ms) > 0.0
                    else elapsed_ms
                )
                peak_raw = details.get("peak_memory_mb")
                peak_mem = (
                    float(peak_raw)
                    if isinstance(peak_raw, (int, float)) and not isinstance(peak_raw, bool)
                    else None
                )
                emulated = bool(details.get("emulated_arch", False))
                raw_tamper = details.get("tamper_attempts")
                tamper = (
                    [str(p) for p in cast(list[Any], raw_tamper)]
                    if isinstance(raw_tamper, list)
                    else []
                )
                clamped_reward = max(0.0, min(1.0, float(reward)))
                passed = clamped_reward >= 1.0 - 1e-9 if kind == "oracle" else clamped_reward <= 0.0
                return TrialRecord(
                    kind=kind,
                    trial_index=trial_index,
                    reward=clamped_reward,
                    passed=passed,
                    duration_ms=round(duration_ms, 3),
                    peak_memory_mb=peak_mem,
                    emulated_arch=emulated,
                    tamper_attempts=tamper,
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return TrialRecord(
                    kind=kind,
                    trial_index=trial_index,
                    reward=0.0,
                    passed=False,
                    duration_ms=round(elapsed_ms, 3),
                    error=str(exc),
                )

    async def run(
        self,
        adapter: EnvironmentAdapter,
        spec: EnvSpec,
        sandbox: Any,
        *,
        oracle_trials: int = 10,
        noop_trials: int = 3,
        semaphore: asyncio.Semaphore | None = None,
    ) -> DeterminismResult:
        """Execute N Oracle trials and M No-op trials and emit DeterminismMetrics + findings."""
        sem = semaphore or asyncio.Semaphore(self._max_concurrent)
        effective_oracle_n = oracle_trials if (spec.has_oracle and oracle_trials > 0) else 0
        effective_noop_n = max(0, noop_trials)

        tasks = [
            self._run_single_trial(
                adapter, spec, sandbox, kind="oracle", trial_index=i, semaphore=sem
            )
            for i in range(effective_oracle_n)
        ] + [
            self._run_single_trial(
                adapter, spec, sandbox, kind="noop", trial_index=j, semaphore=sem
            )
            for j in range(effective_noop_n)
        ]
        trials: list[TrialRecord] = list(await asyncio.gather(*tasks)) if tasks else []

        oracle_records = [t for t in trials if t.kind == "oracle"]
        noop_records = [t for t in trials if t.kind == "noop"]

        oracle_passes = sum(1 for t in oracle_records if t.reward >= 1.0 - 1e-9)
        noop_passes = sum(1 for t in noop_records if t.reward > 0.0)

        oracle_pass_rate = round(oracle_passes / len(oracle_records), 6) if oracle_records else 0.0
        noop_pass_rate = round(noop_passes / len(noop_records), 6) if noop_records else 0.0

        oracle_var = _population_variance([t.reward for t in oracle_records])
        noop_var = _population_variance([t.reward for t in noop_records])
        replay_variance = max(oracle_var, noop_var)

        p_at_1 = (
            round(pass_at_k(len(oracle_records), oracle_passes, 1), 6) if oracle_records else 0.0
        )
        p_pow_k = _compute_pass_pow_checkpoints(len(oracle_records), oracle_passes)

        latencies = [t.duration_ms for t in trials]
        p50_ms = _percentile(latencies, 50.0)
        p95_ms = _percentile(latencies, 95.0)
        peak_mems = [t.peak_memory_mb for t in trials if t.peak_memory_mb is not None]
        peak_memory_mb = round(max(peak_mems), 3) if peak_mems else 0.0
        emulated_warning = any(t.emulated_arch for t in trials)

        metrics = DeterminismMetrics(
            oracle_trials=len(oracle_records),
            oracle_pass_rate=oracle_pass_rate,
            noop_trials=len(noop_records),
            noop_pass_rate=noop_pass_rate,
            replay_variance=replay_variance,
            pass_at_1=p_at_1,
            pass_pow_k=p_pow_k,
            latency_p50_ms=p50_ms,
            latency_p95_ms=p95_ms,
            peak_memory_mb=peak_memory_mb,
            emulated_arch_warning=emulated_warning,
        )

        findings: list[AuditFinding] = []
        if not spec.has_oracle and oracle_trials > 0:
            findings.append(
                AuditFinding(
                    rule_id="RULE-DETM-004",
                    title="Oracle entrypoint absent; Oracle determinism replays skipped",
                    severity=Severity.INFO,
                    tier=1,
                    source="dasv",
                    evidence=(
                        f"Environment '{spec.task_id}' has no oracle_entrypoint; "
                        "only No-op baseline replays executed."
                    ),
                    remediation=(
                        "Add a canonical oracle solution (e.g. solution/solve.sh) so Tier 1 can "
                        "certify 100% Oracle pass rate and pass^k stability."
                    ),
                )
            )

        if oracle_records and oracle_pass_rate < 1.0:
            failed_msgs = [t.error for t in oracle_records if t.error]
            extra = f" Errors: {failed_msgs[0]}" if failed_msgs else ""
            findings.append(
                AuditFinding(
                    rule_id="RULE-DETM-001",
                    title="Oracle solution failed clean-image certification trials",
                    severity=Severity.CRITICAL,
                    tier=1,
                    source="dasv",
                    file_path=spec.oracle_entrypoint,
                    evidence=(
                        f"Oracle passed {oracle_passes}/{len(oracle_records)} trials "
                        f"(pass_rate={oracle_pass_rate:.3f}, expected 1.000).{extra}"
                    ),
                    remediation=(
                        "Fix the task specification, oracle solution, or verifier harness so the "
                        "canonical Oracle achieves 100% pass rate in clean containers."
                    ),
                )
            )

        if noop_records and noop_pass_rate > 0.0:
            findings.append(
                AuditFinding(
                    rule_id="RULE-DETM-002",
                    title="No-op baseline obtained non-zero verifier reward",
                    severity=Severity.CRITICAL,
                    tier=1,
                    source="dasv",
                    file_path=spec.verifier_paths[0] if spec.verifier_paths else None,
                    evidence=(
                        f"No-op baseline scored reward > 0 in {noop_passes}/{len(noop_records)} "
                        f"trials (noop_pass_rate={noop_pass_rate:.3f}, expected 0.000)."
                    ),
                    remediation=(
                        "Add fail-to-pass assertions that verify the unsolved base state fails "
                        "with reward 0.0 before any agent action is applied."
                    ),
                )
            )

        if replay_variance > 0.0:
            findings.append(
                AuditFinding(
                    rule_id="RULE-DETM-003",
                    title="Non-zero replay variance detected across identical trials",
                    severity=Severity.HIGH,
                    tier=1,
                    source="dasv",
                    evidence=(
                        f"Replay variance={replay_variance:.6f} > 0.0 across "
                        f"{len(oracle_records)} Oracle / {len(noop_records)} No-op trials."
                    ),
                    remediation=(
                        "Eliminate nondeterminism (unseeded randomness, wall-clock races, "
                        "network calls, or shared state) from the environment and verifier."
                    ),
                )
            )

        tamper_paths = sorted({p for t in trials for p in t.tamper_attempts})
        if tamper_paths:
            findings.append(
                AuditFinding(
                    rule_id="RULE-BNDY-007",
                    title="Tamper attempt blocked during Tier 1 replay",
                    severity=Severity.CRITICAL,
                    tier=1,
                    source="dasv",
                    file_path=tamper_paths[0],
                    evidence=(
                        "Protected verifier path modification(s) stripped before grading: "
                        f"{', '.join(tamper_paths)}."
                    ),
                    remediation=(
                        "Ensure agent/oracle scripts never touch verifier files and keep "
                        "verifiers mounted read-only (:ro) in the Verification Plane."
                    ),
                )
            )

        passed = (
            (not oracle_records or oracle_pass_rate >= 1.0 - 1e-9)
            and noop_pass_rate <= 0.0
            and replay_variance <= 0.0
            and not tamper_paths
        )
        return DeterminismResult(
            metrics=metrics,
            findings=findings,
            trials=trials,
            passed=passed,
        )
