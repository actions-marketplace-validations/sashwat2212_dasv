"""CHERRL cross-family judge divergence & reference-free risk audit (`FEAT-16`, `RULE-RUBR-003`)."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from dasv.core.errors import BudgetExceededError, ConfigError
from dasv.core.schemas import AuditFinding, Severity
from dasv.rubric.perturbations import (
    RubricJudgeScorer,
    build_default_wrong_baseline,
    generate_verbosity_perturbations,
)

if TYPE_CHECKING:
    from dasv.rubric.prorubric_audit import RubricInspectionTarget

DEFAULT_TAU_THRESHOLD: float = 0.70
DEFAULT_DELTA_THRESHOLD: float = 0.15

_REFERENCE_ANCHOR_MARKERS: tuple[str, ...] = (
    "reference_answer",
    "reference_solution",
    "gold_answer",
    "gold_solution",
    "expected_output",
    "expected_answer",
    "ground_truth",
    "reference_text",
    "canonical_answer",
)


class JudgeSample(BaseModel):
    """Single `(task_prompt, candidate_response)` item evaluated across cross-family judges."""

    sample_id: str
    task_prompt: str
    candidate_response: str
    reference_answer: str | None = None


class JudgePairDivergence(BaseModel):
    """Pairwise rank correlation & agreement statistics between two distinct judges."""

    judge_a: str
    judge_b: str
    sample_count: int
    kendall_tau: float
    tau_ci_low: float
    tau_ci_high: float
    spearman_rho: float
    mean_abs_delta: float
    mean_score_gap: float
    divergent: bool
    degenerate_reason: str | None = None


class JudgeDivergenceResult(BaseModel):
    """Aggregated outcome of `FEAT-16` CHERRL cross-family judge divergence analysis."""

    judges: list[str]
    sample_count: int
    tau_threshold: float = DEFAULT_TAU_THRESHOLD
    delta_threshold: float = DEFAULT_DELTA_THRESHOLD
    is_reference_free: bool = False
    reference_anchor_evidence: list[str] = Field(default_factory=list[str])
    scores_by_judge: dict[str, list[float]] = Field(default_factory=dict[str, list[float]])
    pairs: list[JudgePairDivergence] = Field(default_factory=list[JudgePairDivergence])
    findings: list[AuditFinding] = Field(default_factory=list[AuditFinding])
    partial_budget_exceeded: bool = False

    @property
    def passed(self) -> bool:
        return not self.partial_budget_exceeded and not any(
            f.rule_id == "RULE-RUBR-003" for f in self.findings
        )


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute exact, tie-aware Kendall's tau-b rank correlation in pure Python.

    Returns 0.0 cleanly for degenerate inputs (`len < 2` or constant score arrays).
    """
    n = min(len(x), len(y))
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(n):
        xi = float(x[i])
        yi = float(y[i])
        for j in range(i + 1, n):
            dx = xi - float(x[j])
            dy = yi - float(y[j])
            if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
                ties_x += 1
                ties_y += 1
            elif abs(dx) <= 1e-12:
                ties_x += 1
            elif abs(dy) <= 1e-12:
                ties_y += 1
            elif dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1

    n0 = n * (n - 1) // 2
    denom_sq = (n0 - ties_x) * (n0 - ties_y)
    if denom_sq <= 0:
        return 0.0
    return (concordant - discordant) / math.sqrt(float(denom_sq))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Assign 1-based midranks for tie-aware Spearman rank correlation."""
    n = len(values)
    indexed = sorted(enumerate(float(v) for v in values), key=lambda pair: pair[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(indexed[j][1] - indexed[i][1]) <= 1e-12:
            j += 1
        # Average 1-based rank over slice [i, j)
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute exact, tie-aware Spearman's rho as Pearson correlation of midranks."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    rx = _average_ranks(x[:n])
    ry = _average_ranks(y[:n])
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    denom = math.sqrt(var_x * var_y)
    if denom <= 1e-15:
        return 0.0
    return max(-1.0, min(1.0, cov / denom))


def mean_abs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute mean absolute score difference `mean(|x_i - y_i|)`."""
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    return sum(abs(float(a) - float(b)) for a, b in zip(x[:n], y[:n], strict=True)) / n


def bootstrap_kendall_tau_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_resamples: int = 400,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap CI for Kendall's tau-b with fixed RNG seed."""
    n = min(len(x), len(y))
    point = kendall_tau_b(x[:n], y[:n])
    if n < 3 or n_resamples < 10:
        return round(point, 6), round(point, 6)

    rng = random.Random(seed)
    xs = [float(v) for v in x[:n]]
    ys = [float(v) for v in y[:n]]
    boot_vals: list[float] = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idxs]
        by = [ys[i] for i in idxs]
        boot_vals.append(kendall_tau_b(bx, by))
    boot_vals.sort()
    lo_idx = max(0, min(len(boot_vals) - 1, int(math.floor((alpha / 2.0) * len(boot_vals)))))
    hi_idx = max(
        0, min(len(boot_vals) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(boot_vals))) - 1)
    )
    return round(boot_vals[lo_idx], 6), round(boot_vals[hi_idx], 6)


def detect_reference_free_judge(
    target_path: Path,
    targets: Sequence[RubricInspectionTarget],
) -> tuple[bool, list[str]]:
    """Detect whether rubric/judge files operate reference-free without conjunctive gates."""
    evidence: list[str] = []
    has_conjunctive_gate = any(t.has_conjunctive_gate for t in targets)
    if has_conjunctive_gate:
        evidence.append("Static AST/schema inspection confirmed a conjunctive deterministic gate.")
        return False, evidence

    candidate_files: list[Path] = []
    root = target_path.resolve()
    if root.is_file():
        candidate_files = [root]
    else:
        for t in targets:
            p = (root / t.file_path).resolve()
            if p.is_file() and p not in candidate_files:
                candidate_files.append(p)
        if not candidate_files:
            candidate_files = [p for p in sorted(root.rglob("*.py")) if p.is_file()][:20]

    for fpath in candidate_files:
        try:
            text_low = fpath.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        for marker in _REFERENCE_ANCHOR_MARKERS:
            if marker in text_low:
                evidence.append(f"Found reference anchor marker '{marker}' in {fpath.name}.")
                return False, evidence

    evidence.append(
        "No conjunctive deterministic gate or reference-answer anchor detected in rubric files "
        "(purely reference-free LLM judge pipeline)."
    )
    return True, evidence


def load_judge_samples_jsonl(samples_path: Path) -> list[JudgeSample]:
    """Load `(prompt, response)` evaluation pairs from a JSONL file."""
    if not samples_path.is_file():
        raise ConfigError(f"Judge samples JSONL file not found: '{samples_path}'.")
    samples: list[JudgeSample] = []
    for lineno, raw_line in enumerate(
        samples_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"Malformed JSON in samples file '{samples_path}' line {lineno}: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            continue
        rec = cast(dict[str, Any], parsed)
        prompt = str(rec.get("task_prompt") or rec.get("prompt") or "").strip()
        response = str(rec.get("candidate_response") or rec.get("response") or "").strip()
        if not prompt or not response:
            continue
        sid = str(rec.get("sample_id") or rec.get("id") or f"sample-{len(samples) + 1}")
        ref_raw = rec.get("reference_answer")
        ref_ans = str(ref_raw) if isinstance(ref_raw, str) and ref_raw.strip() else None
        samples.append(
            JudgeSample(
                sample_id=sid,
                task_prompt=prompt,
                candidate_response=response,
                reference_answer=ref_ans,
            )
        )
    if not samples:
        raise ConfigError(
            f"No valid sample records with 'prompt'/'task_prompt' and "
            f"'response'/'candidate_response' found in '{samples_path}'."
        )
    return samples


def build_default_divergence_samples(
    task_prompt: str,
    *,
    baseline_response: str | None = None,
) -> list[JudgeSample]:
    """Build a deterministic 5-sample probe batch spanning grounded, partial, and fluff text."""
    wrong_base = (
        baseline_response.strip()
        if baseline_response and baseline_response.strip()
        else build_default_wrong_baseline(task_prompt)
    )
    perts = generate_verbosity_perturbations(wrong_base, task_prompt=task_prompt)
    samples = [
        JudgeSample(
            sample_id="probe-minimal-wrong",
            task_prompt=task_prompt,
            candidate_response=wrong_base,
        ),
        JudgeSample(
            sample_id="probe-concise-grounded",
            task_prompt=task_prompt,
            candidate_response=(
                "Direct solution addressing all required constraints with verified boundary "
                "conditions, explicit complexity analysis, and concrete edge-case handling."
            ),
        ),
    ]
    for pert in perts:
        samples.append(
            JudgeSample(
                sample_id=f"probe-{pert.kind}",
                task_prompt=task_prompt,
                candidate_response=pert.perturbed_response,
            )
        )
    return samples


def _evaluate_pair(
    judge_a: str,
    scores_a: Sequence[float],
    judge_b: str,
    scores_b: Sequence[float],
    *,
    tau_threshold: float,
    delta_threshold: float,
    seed: int,
) -> JudgePairDivergence:
    n = min(len(scores_a), len(scores_b))
    xs = [float(v) for v in scores_a[:n]]
    ys = [float(v) for v in scores_b[:n]]

    degenerate_reason: str | None = None
    if n < 3:
        degenerate_reason = f"fewer than 3 paired samples (n={n}); rank correlation underpowered"
    else:
        const_x = len({round(v, 9) for v in xs}) <= 1
        const_y = len({round(v, 9) for v in ys}) <= 1
        if const_x or const_y:
            degenerate_reason = "constant score vector from at least one judge (zero variance)"

    tau = round(kendall_tau_b(xs, ys), 6)
    rho = round(spearman_rho(xs, ys), 6)
    ci_lo, ci_hi = bootstrap_kendall_tau_ci(xs, ys, seed=seed)
    mad = round(mean_abs_delta(xs, ys), 6)
    gap = round(abs((sum(xs) / n) - (sum(ys) / n)), 6) if n > 0 else 0.0

    divergent = (n >= 2) and (tau < tau_threshold or mad > delta_threshold)
    return JudgePairDivergence(
        judge_a=judge_a,
        judge_b=judge_b,
        sample_count=n,
        kendall_tau=tau,
        tau_ci_low=ci_lo,
        tau_ci_high=ci_hi,
        spearman_rho=rho,
        mean_abs_delta=mad,
        mean_score_gap=gap,
        divergent=divergent,
        degenerate_reason=degenerate_reason,
    )


async def run_judge_divergence_audit(
    samples: Sequence[JudgeSample],
    judges: Sequence[tuple[str, RubricJudgeScorer]],
    *,
    tau_threshold: float = DEFAULT_TAU_THRESHOLD,
    delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
    is_reference_free: bool = False,
    reference_anchor_evidence: Sequence[str] = (),
    file_path: str | None = None,
    seed: int = 42,
) -> JudgeDivergenceResult:
    """Score `samples` across >= 2 judges and emit `RULE-RUBR-003` on divergence / ref-free risk."""
    if len(judges) < 2:
        raise ConfigError(
            "CHERRL judge divergence requires at least 2 distinct judges "
            "(e.g. `--judges modelA,modelB`)."
        )
    if not samples:
        raise ConfigError("At least 1 JudgeSample is required for judge divergence analysis.")

    judge_names = [name for name, _scorer in judges]
    scores_by_judge: dict[str, list[float]] = {name: [] for name in judge_names}
    findings: list[AuditFinding] = []
    budget_tripped = False

    # Evaluate sample-by-sample across all judges so partial budget aborts keep aligned pairs.
    for sample in samples:
        sample_scores: dict[str, float] = {}
        try:
            for name, scorer in judges:
                sc = await scorer.score(sample.task_prompt, sample.candidate_response)
                sample_scores[name] = round(max(0.0, min(1.0, float(sc))), 6)
        except BudgetExceededError as exc:
            budget_tripped = True
            findings.append(
                AuditFinding(
                    rule_id="RULE-BUDG-001",
                    title="Hard budget circuit breaker triggered during CHERRL judge divergence",
                    severity=Severity.INFO,
                    tier=2,
                    source="dasv",
                    file_path=file_path,
                    evidence=str(exc),
                    remediation=(
                        "Partial aligned judge scores were preserved. Raise `--max-cost-usd` or "
                        "`--max-tokens`, or run against local $0 Ollama/MLX judge endpoints."
                    ),
                )
            )
            break
        for name in judge_names:
            scores_by_judge[name].append(sample_scores[name])

    completed_n = min((len(scores_by_judge[name]) for name in judge_names), default=0)
    pairs: list[JudgePairDivergence] = []
    for i in range(len(judge_names)):
        for j in range(i + 1, len(judge_names)):
            ja = judge_names[i]
            jb = judge_names[j]
            pairs.append(
                _evaluate_pair(
                    ja,
                    scores_by_judge[ja],
                    jb,
                    scores_by_judge[jb],
                    tau_threshold=tau_threshold,
                    delta_threshold=delta_threshold,
                    seed=seed + i * 31 + j,
                )
            )

    divergent_pairs = [p for p in pairs if p.divergent]
    if divergent_pairs:
        sev = Severity.HIGH if is_reference_free else Severity.MEDIUM
        pair_summaries = "; ".join(
            f"{p.judge_a} vs {p.judge_b}: tau={p.kendall_tau:.3f} "
            f"[{p.tau_ci_low:.3f}, {p.tau_ci_high:.3f}], rho={p.spearman_rho:.3f}, "
            f"mean|delta|={p.mean_abs_delta:.3f} (n={p.sample_count})"
            for p in divergent_pairs
        )
        ref_note = (
            " Pipeline is purely reference-free (no deterministic gate or reference anchor), "
            "leaving RL training exposed to plausibility-basin hacking even under multi-judge "
            "ensembles (arXiv:2607.05904)."
            if is_reference_free
            else " Anchor/conjunctive gate detected, but judge rankings still diverge."
        )
        findings.append(
            AuditFinding(
                rule_id="RULE-RUBR-003",
                title=(
                    "Unanchored reference-free judge with cross-family divergence (CHERRL)"
                    if is_reference_free
                    else "Cross-family rubric judge divergence detected (CHERRL)"
                ),
                severity=sev,
                tier=2,
                source="dasv",
                file_path=file_path,
                line_number=None,
                evidence=(
                    f"Judge agreement breached thresholds (tau < {tau_threshold:.2f} or "
                    f"mean|delta| > {delta_threshold:.2f}): {pair_summaries}.{ref_note}"
                ),
                remediation=(
                    "Anchor judge prompts to pre-committed reference answers or deterministic "
                    "ProRubric conjunctive gates, and calibrate judge instructions until "
                    "cross-family Kendall tau >= 0.70 and mean|delta| <= 0.15."
                ),
            )
        )
    elif is_reference_free and completed_n >= 2:
        findings.append(
            AuditFinding(
                rule_id="RULE-RUBR-003",
                title="Unanchored reference-free LLM judge pipeline risk (CHERRL)",
                severity=Severity.HIGH,
                tier=2,
                source="dasv",
                file_path=file_path,
                line_number=None,
                evidence=(
                    f"Judges ({', '.join(judge_names)}) agreed on {completed_n} probe sample(s), "
                    "but no reference answer anchor or deterministic conjunctive gate was found. "
                    "Reference-free plausibility hacking persists even across multi-judge "
                    "ensembles (arXiv:2607.05904)."
                ),
                remediation=(
                    "Add a pre-committed reference answer or deterministic ProRubric conjunctive "
                    "gate (`all_required_pass AND no_hard_violation`) before RL training."
                ),
            )
        )

    return JudgeDivergenceResult(
        judges=judge_names,
        sample_count=completed_n,
        tau_threshold=tau_threshold,
        delta_threshold=delta_threshold,
        is_reference_free=is_reference_free,
        reference_anchor_evidence=list(reference_anchor_evidence),
        scores_by_judge=scores_by_judge,
        pairs=pairs,
        findings=findings,
        partial_budget_exceeded=budget_tripped,
    )
