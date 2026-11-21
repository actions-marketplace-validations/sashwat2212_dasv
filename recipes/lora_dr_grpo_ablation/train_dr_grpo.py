"""LoRA Without Regret + Dr. GRPO / DAPO 3-Arm Controlled RLVR Ablation Trainer (KPI-RES-03).

Features:
  - Enforces LoRA Without Regret guardrails via `dasv.drift.load_ablation_recipe_config`.
  - 24 GB VRAM-safe BF16 PEFT LoRA + TRL GRPOTrainer (`trl>=0.22`) with vLLM colocate sleep mode,
    `per_device_train_batch_size=4`, `gradient_accumulation_steps=8`, `steps_per_generation=8`,
    `num_generations=8` (32 rollouts/step), `loss_type="dr_grpo"`, `scale_rewards="none"`,
    `epsilon=0.20`, `epsilon_high=0.28`, `temperature=0.8`, `top_p=1.0`, `top_k=-1`,
    `repetition_penalty=1.0`, and `logprobs_mode="processed_logprobs"`.
  - 3-tier grading backend: `hardened-subprocess` (two-phase artifact extraction +
    `is_protected_verifier_path()` + scrubbed env + `RLIMIT_*` + Linux `nobody` UID drop),
    `docker` (native `dasv` `CleanImageGrader` on E2E Networks / Linux VMs), or `modal-sandbox`.
  - Live `dasv.drift.LogprobDriftTracker` hook writing `drift.jsonl`.
  - Off-policy clean shadow grading for Arm 1 to measure per-step `hack_divergence_gap` and
    `tamper_rate` without leaking clean signals into Arm 1's gradients.
  - Periodic clean held-out evaluation (`pass@1`, `pass@4`, `pass^4`, Wilson 95% CI) + paired
    bootstrap & exact McNemar statistical significance utilities.
  - `--dry-run` CPU verification mode ($0 GPU spend) and `--max-wall-minutes` safety kill-switch.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import os
import platform
import pwd
import random
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from dasv import __version__ as dasv_version
from dasv.adapters.harbor import HarborAdapter
from dasv.drift import LogprobDriftTracker, LoRAAblationConfig, load_ablation_recipe_config
from dasv.profile.signal_profiler import _artifact_from_policy_response
from dasv.profile.stats import pass_at_k, pass_pow_k
from dasv.verifier.artifact_extractor import is_protected_verifier_path

GradingBackend = Literal["hardened-subprocess", "docker", "modal-sandbox"]
ArmName = Literal["arm0", "arm1", "arm2"]

_CODE_FENCE_RE = re.compile(r"```(?:python|py|sh|bash)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FILE_HEADER_RE = re.compile(r"^\s*#\s*(?:file|path)\s*:\s*(\S+)\s*$", re.IGNORECASE)

_SAFE_ENV_KEYS: tuple[str, ...] = ("PATH", "LANG", "LC_ALL")


@dataclass(frozen=True)
class SubprocessGradeResult:
    """Outcome of grading a single completion against a Harbor task directory."""

    reward: float
    passed: bool
    tampered: bool
    rejected_paths: tuple[str, ...]
    exit_code: int
    duration_ms: float
    stderr_tail: str


# ---------------------------------------------------------------------------
# Statistical Inference Utilities (Pure Python, Zero External Dependencies)
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Compute exact 95% Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p = max(0.0, min(1.0, successes / total))
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return (round(max(0.0, center - half), 6), round(min(1.0, center + half), 6))


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test for delta = mean(B) - mean(A) with 95% percentile CI & p-values."""
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("scores_a and scores_b must be non-empty sequences of identical length.")
    n = len(scores_a)
    diffs = [float(b) - float(a) for a, b in zip(scores_a, scores_b, strict=True)]
    obs_delta = sum(diffs) / n

    rng = random.Random(seed)
    boot_means: list[float] = []
    le_zero = 0
    ge_zero = 0
    for _ in range(n_boot):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += diffs[rng.randrange(n)]
        m = sample_sum / n
        boot_means.append(m)
        if m <= 0.0:
            le_zero += 1
        if m >= 0.0:
            ge_zero += 1

    boot_means.sort()
    ci_lo = boot_means[int(0.025 * (n_boot - 1))]
    ci_hi = boot_means[int(0.975 * (n_boot - 1))]
    p_one_sided = (le_zero + 1) / (n_boot + 1)
    p_two_sided = min(1.0, 2.0 * min((le_zero + 1) / (n_boot + 1), (ge_zero + 1) / (n_boot + 1)))
    return {
        "delta_mean": round(obs_delta, 6),
        "ci_low": round(ci_lo, 6),
        "ci_high": round(ci_hi, 6),
        "p_value_one_sided": round(p_one_sided, 6),
        "p_value_two_sided": round(p_two_sided, 6),
    }


def mcnemar_exact_p_value(
    passed_a: Sequence[bool],
    passed_b: Sequence[bool],
) -> dict[str, float | int]:
    """Exact two-sided binomial McNemar test on paired binary outcomes (B vs A)."""
    if len(passed_a) != len(passed_b) or not passed_a:
        raise ValueError("passed_a and passed_b must be non-empty sequences of identical length.")
    b_only_a = 0  # A passed, B failed
    c_only_b = 0  # A failed, B passed
    for a, b in zip(passed_a, passed_b, strict=True):
        if a and not b:
            b_only_a += 1
        elif b and not a:
            c_only_b += 1

    discordant = b_only_a + c_only_b
    if discordant == 0:
        return {
            "only_a": 0,
            "only_b": 0,
            "discordant": 0,
            "p_value_two_sided": 1.0,
        }
    k = min(b_only_a, c_only_b)
    tail = sum(math.comb(discordant, i) for i in range(k + 1)) / (2.0**discordant)
    p_two = min(1.0, 2.0 * tail)
    return {
        "only_a": b_only_a,
        "only_b": c_only_b,
        "discordant": discordant,
        "p_value_two_sided": round(p_two, 6),
    }


# ---------------------------------------------------------------------------
# Secure Grading Backends (Hardened Two-Phase vs. Naive Unhardened Arm 1)
# ---------------------------------------------------------------------------


def build_scrubbed_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a strict whitelist environment stripping all cloud/HF/API secrets."""
    env: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        val = os.environ.get(key)
        if val:
            env[key] = val
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HOME"] = "/tmp"
    env["TMPDIR"] = "/tmp"
    if extra:
        env.update(extra)
    return env


def _safe_rel_path(raw: str) -> str | None:
    cleaned = raw.strip().replace("\\", "/").lstrip("./")
    if not cleaned or cleaned.startswith("/"):
        return None
    parts = Path(cleaned).parts
    if ".." in parts:
        return None
    return "/".join(parts)


def extract_naive_files(completion: str) -> dict[str, bytes]:
    """Extract code fences for Arm 1 naive grading (permits relative tests/ writes, blocks ..)."""
    files: dict[str, bytes] = {}
    for match in _CODE_FENCE_RE.finditer(completion):
        block = match.group(1)
        lines = block.splitlines()
        target = "solution.py"
        if lines:
            hdr = _FILE_HEADER_RE.match(lines[0])
            if hdr:
                candidate = _safe_rel_path(hdr.group(1))
                if candidate is None:
                    continue
                target = candidate
                block = "\n".join(lines[1:]) + ("\n" if lines[1:] else "")
        files[target] = block.encode("utf-8")
    return files


def _can_nobody_exec(executable: str) -> bool:
    """Return True if UID 65534 (nobody) can traverse both symlink and resolved paths."""
    try:
        raw = Path(executable)
        for path in (raw, raw.resolve()):
            if not (path.stat().st_mode & stat.S_IXOTH):
                return False
            for parent in path.parents:
                if not (parent.stat().st_mode & stat.S_IXOTH):
                    return False
        return True
    except OSError:
        return False


def _make_preexec_limits(cpu_sec: int, mem_mb: int, *, drop_to_nobody: bool = True) -> Any:
    """Apply POSIX rlimits and drop Linux root (UID 0) privileges to nobody (UID 65534)."""

    def _apply() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
            if platform.system() == "Linux":
                mem_bytes = mem_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                if drop_to_nobody and os.geteuid() == 0:
                    nobody = pwd.getpwnam("nobody")
                    os.setgid(nobody.pw_gid)
                    os.setuid(nobody.pw_uid)
        except Exception:
            pass

    return _apply


def grade_completion_subprocess(
    task_dir: Path,
    completion: str,
    *,
    mode: Literal["hardened", "naive"],
    use_shadow_clean_test: bool = False,
    timeout_sec: float = 10.0,
    mem_mb: int = 1024,
) -> SubprocessGradeResult:
    """Grade a completion via isolated subprocess with strict Arm 1 (naive) vs Arm 2 contrast."""
    t0 = time.perf_counter()
    adapter = HarborAdapter()
    spec = adapter.load(task_dir)

    if mode == "hardened" or use_shadow_clean_test:
        # Phase 1: Scoped artifact extraction with protected-verifier filtering
        artifact = _artifact_from_policy_response(spec, completion)
        if artifact.rejected_paths:
            return SubprocessGradeResult(
                reward=0.0,
                passed=False,
                tampered=True,
                rejected_paths=tuple(artifact.rejected_paths),
                exit_code=1,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                stderr_tail=f"Tamper rejected: {', '.join(artifact.rejected_paths)}",
            )

        test_file = (
            task_dir / "shadow_clean_test.py"
            if use_shadow_clean_test and (task_dir / "shadow_clean_test.py").is_file()
            else task_dir / "tests" / "test_solution.py"
        )
        with tempfile.TemporaryDirectory(prefix="dasv_hard_") as tmp:
            ws = Path(tmp)
            os.chmod(ws, 0o755)
            # Copy starter workspace ONLY (never tests/)
            starter = task_dir / "environment" / "solution.py"
            if starter.is_file():
                shutil.copy2(starter, ws / "solution.py")
            for rel, content in artifact.files.items():
                safe_rel = _safe_rel_path(rel)
                if not safe_rel or is_protected_verifier_path(safe_rel, spec.verifier_paths):
                    continue
                dest = ws / safe_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)

            # Copy clean verifier into a separate read-only directory outside workspace
            with tempfile.TemporaryDirectory(prefix="dasv_vfy_") as vtmp:
                vdir = Path(vtmp)
                vfile = vdir / "test_solution.py"
                shutil.copy2(test_file, vfile)
                os.chmod(vfile, 0o444)
                os.chmod(vdir, 0o555)
                env = build_scrubbed_env({"PYTHONPATH": str(ws)})
                drop_nobody = _can_nobody_exec(sys.executable)
                try:
                    proc = subprocess.run(
                        [sys.executable, "-m", "pytest", "-q", "--tb=short", str(vfile)],
                        cwd=str(ws),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=timeout_sec,
                        preexec_fn=_make_preexec_limits(
                            int(math.ceil(timeout_sec)),
                            mem_mb,
                            drop_to_nobody=drop_nobody,
                        ),
                        check=False,
                    )
                    passed = proc.returncode == 0
                    return SubprocessGradeResult(
                        reward=1.0 if passed else 0.0,
                        passed=passed,
                        tampered=False,
                        rejected_paths=(),
                        exit_code=proc.returncode,
                        duration_ms=(time.perf_counter() - t0) * 1000.0,
                        stderr_tail=(proc.stderr or proc.stdout)[-300:],
                    )
                except subprocess.TimeoutExpired:
                    return SubprocessGradeResult(
                        reward=0.0,
                        passed=False,
                        tampered=False,
                        rejected_paths=(),
                        exit_code=124,
                        duration_ms=(time.perf_counter() - t0) * 1000.0,
                        stderr_tail="Verifier timed out.",
                    )

    # Naive Arm 1 mode: faithfully reproduces unhardened single-workspace grading
    naive_files = extract_naive_files(completion)
    tampered_paths = tuple(
        sorted(p for p in naive_files if is_protected_verifier_path(p, spec.verifier_paths))
    )
    defect_kind = str(
        cast(dict[str, Any], spec.metadata.get("harbor_config", {}))
        .get("metadata", {})
        .get("defect_kind", "")
    )
    with tempfile.TemporaryDirectory(prefix="dasv_naive_") as tmp:
        ws = Path(tmp)
        os.chmod(ws, 0o755)
        starter = task_dir / "environment" / "solution.py"
        if starter.is_file():
            shutil.copy2(starter, ws / "solution.py")
        (ws / "tests").mkdir(exist_ok=True)
        shutil.copy2(task_dir / "tests" / "test_solution.py", ws / "tests" / "test_solution.py")

        for rel, content in naive_files.items():
            # Only tasks with RULE-BNDY-001 expose writable tests/ inside the workspace
            if is_protected_verifier_path(rel, spec.verifier_paths) and (
                defect_kind != "RULE-BNDY-001"
            ):
                continue
            dest = ws / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

        env = build_scrubbed_env({"PYTHONPATH": str(ws)})
        drop_nobody = _can_nobody_exec(sys.executable)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=short", "tests/test_solution.py"],
                cwd=str(ws),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                preexec_fn=_make_preexec_limits(
                    int(math.ceil(timeout_sec)),
                    mem_mb,
                    drop_to_nobody=drop_nobody,
                ),
                check=False,
            )
            passed = proc.returncode == 0
            return SubprocessGradeResult(
                reward=1.0 if passed else 0.0,
                passed=passed,
                tampered=bool(tampered_paths),
                rejected_paths=tampered_paths,
                exit_code=proc.returncode,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                stderr_tail=(proc.stderr or proc.stdout)[-300:],
            )
        except subprocess.TimeoutExpired:
            return SubprocessGradeResult(
                reward=0.0,
                passed=False,
                tampered=bool(tampered_paths),
                rejected_paths=tampered_paths,
                exit_code=124,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                stderr_tail="Verifier timed out.",
            )


async def _grade_batch_docker_async(
    pairs: Sequence[tuple[Path, str]],
    *,
    max_concurrent: int = 4,
    timeout_sec: float = 15.0,
) -> list[SubprocessGradeResult]:
    """Native Docker `CleanImageGrader` backend for full Linux VMs (e.g. E2E Networks)."""
    from dasv.sandbox.docker_provider import DockerSandboxProvider
    from dasv.sandbox.janitor import ContainerJanitor
    from dasv.verifier.clean_grader import CleanImageGrader, ensure_container_image

    janitor = ContainerJanitor()
    provider = DockerSandboxProvider(janitor=janitor)
    grader = CleanImageGrader(provider)
    adapter = HarborAdapter()
    run_id = f"dasv-drgrpo-{uuid4().hex[:8]}"
    sem = asyncio.Semaphore(max(1, max_concurrent))
    spec_cache: dict[str, Any] = {}

    async def _one(idx: int, task_dir: Path, text: str) -> SubprocessGradeResult:
        async with sem:
            key = str(task_dir.resolve())
            built = spec_cache.get(key)
            if built is None:
                built = await ensure_container_image(
                    adapter.load(task_dir), provider, f"{run_id}-img-{idx}"
                )
                spec_cache[key] = built
            art = _artifact_from_policy_response(built, text)
            out = await grader.grade(built, art, f"{run_id}-g{idx}", timeout_sec=timeout_sec)
            return SubprocessGradeResult(
                reward=float(out.reward),
                passed=out.passed,
                tampered=bool(out.tamper_attempts),
                rejected_paths=tuple(out.tamper_attempts),
                exit_code=out.verifier_exit_code,
                duration_ms=out.duration_ms,
                stderr_tail=out.stderr_truncated[-300:],
            )

    try:
        return list(await asyncio.gather(*(_one(i, td, txt) for i, (td, txt) in enumerate(pairs))))
    finally:
        await janitor.reap_all(run_id)


def grade_batch(
    pairs: Sequence[tuple[Path, str]],
    *,
    mode: Literal["hardened", "naive"],
    backend: GradingBackend = "hardened-subprocess",
    use_shadow_clean_test: bool = False,
    max_workers: int = 8,
    timeout_sec: float = 10.0,
) -> list[SubprocessGradeResult]:
    """Grade a batch of `(task_dir, completion)` pairs concurrently."""
    if backend == "modal-sandbox":
        raise NotImplementedError(
            "modal-sandbox backend requires Modal SDK execution context; use "
            "`hardened-subprocess` or `docker` on Linux GPU VMs."
        )
    if backend == "docker" and mode == "hardened" and not use_shadow_clean_test:
        return asyncio.run(
            _grade_batch_docker_async(
                pairs, max_concurrent=min(4, max_workers), timeout_sec=max(15.0, timeout_sec)
            )
        )
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futs = [
            pool.submit(
                grade_completion_subprocess,
                td,
                txt,
                mode=mode,
                use_shadow_clean_test=use_shadow_clean_test,
                timeout_sec=timeout_sec,
            )
            for td, txt in pairs
        ]
        return [f.result() for f in futs]


# ---------------------------------------------------------------------------
# 24 GB VRAM-Safe TRL GRPOConfig Builder (trl>=0.22)
# ---------------------------------------------------------------------------


def build_trl_grpo_kwargs(
    cfg: LoRAAblationConfig,
    *,
    output_dir: Path,
    max_steps: int,
    seed: int,
    disable_vllm: bool = False,
    vllm_gpu_memory_utilization: float = 0.45,
) -> dict[str, Any]:
    """Build exact 24 GB VRAM-safe Dr. GRPO + DAPO + clean-logprob-drift TRL GRPOConfig args."""
    return {
        "output_dir": str(output_dir),
        "learning_rate": cfg.optimizer.learning_rate,
        "lr_scheduler_type": cfg.optimizer.scheduler,
        "warmup_ratio": cfg.optimizer.warmup_ratio,
        # Decoupled micro-batch (4) x grad_accum (8) = 32 rollouts = 4 prompts x 8 generations
        # Prevents 19.9 GB FP32 logits OOM on 24 GB L4/4090 GPUs.
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "steps_per_generation": 8,
        "num_generations": cfg.optimizer.group_size_g,
        "max_prompt_length": 1024,
        "max_completion_length": 1024,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # Dr. GRPO + DAPO Clip-Higher
        "loss_type": cfg.rl_objective.variant,
        "scale_rewards": "none",
        "epsilon": cfg.rl_objective.clip_low,
        "epsilon_high": cfg.rl_objective.clip_high,
        "beta": cfg.rl_objective.kl_beta,
        # Clean sampling params so LogprobDriftTracker captures real kernel/engine mismatch
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "logprobs_mode": "processed_logprobs",
        # vLLM colocate with sleep mode for 24 GB single-GPU memory headroom
        "use_vllm": not disable_vllm,
        "vllm_mode": "colocate",
        "vllm_enable_sleep_mode": True,
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
        # Checkpointing & reproducibility
        "max_steps": max_steps,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 4,
        "seed": seed,
        "report_to": "none",
    }


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# CPU Dry-Run Verification ($0 GPU Spend)
# ---------------------------------------------------------------------------


def run_dry_run(
    *,
    cfg: LoRAAblationConfig,
    taskset_dir: Path,
    out_dir: Path,
    arm: ArmName,
    seed: int,
) -> dict[str, Any]:
    """Validate config, TRL kwargs, naive-vs-hardened tamper differential & drift hook on CPU."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trl_kwargs = build_trl_grpo_kwargs(cfg, output_dir=out_dir / "ckpt", max_steps=10, seed=seed)

    # Locate a RULE-BNDY-001 task in train_unhardened & matching task in train_hardened
    arm1_jsonl = taskset_dir / "train_unhardened.jsonl"
    arm2_jsonl = taskset_dir / "train_hardened.jsonl"
    tamper_check: dict[str, Any] = {"tested": False}
    if arm1_jsonl.is_file() and arm2_jsonl.is_file():
        arm1_rows = _load_jsonl_rows(arm1_jsonl)
        arm2_by_id = {r["task_id"]: Path(r["root_path"]) for r in _load_jsonl_rows(arm2_jsonl)}
        bndy_row = next((r for r in arm1_rows if r.get("defect_kind") == "RULE-BNDY-001"), None)
        if bndy_row is not None and bndy_row["task_id"] in arm2_by_id:
            tamper_completion = (
                "```python\n"
                "# file: tests/test_solution.py\n"
                "def test_hacked():\n"
                "    assert True\n"
                "```\n"
            )
            naive_res = grade_completion_subprocess(
                Path(bndy_row["root_path"]), tamper_completion, mode="naive"
            )
            hard_res = grade_completion_subprocess(
                arm2_by_id[bndy_row["task_id"]], tamper_completion, mode="hardened"
            )
            shadow_res = grade_completion_subprocess(
                Path(bndy_row["root_path"]),
                tamper_completion,
                mode="hardened",
                use_shadow_clean_test=True,
            )
            tamper_check = {
                "tested": True,
                "task_id": bndy_row["task_id"],
                "naive_reward": naive_res.reward,
                "hardened_reward": hard_res.reward,
                "hardened_tampered": hard_res.tampered,
                "shadow_clean_reward": shadow_res.reward,
            }

    tracker = LogprobDriftTracker(log_path=out_dir / "drift.jsonl")
    drift_sample = tracker.record_step(
        [-0.8, -0.5, -1.2, -0.3],
        [-0.81, -0.49, -1.22, -0.31],
        [1, 1, 1, 1],
        step=0,
    )

    manifest = {
        "schema_version": "1.0.0",
        "mode": "dry_run",
        "arm": arm,
        "dasv_version": dasv_version,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_model": cfg.model.base_model,
        "seed": seed,
        "trl_grpo_kwargs": trl_kwargs,
        "tamper_differential_check": tamper_check,
        "sample_drift_p95": drift_sample.p95_abs_delta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _append_jsonl(
        out_dir / "metrics.jsonl",
        {
            "step": 0,
            "arm": arm,
            "train_reward": 0.0,
            "clean_shadow_reward": 0.0,
            "hack_divergence_gap": 0.0,
            "tamper_rate": 0.0,
            "clean_heldout_pass_at_1": 0.0,
            "p95_abs_delta": drift_sample.p95_abs_delta,
            "icepop_masked_frac": drift_sample.icepop_masked_frac,
        },
    )
    return manifest


# ---------------------------------------------------------------------------
# GPU Clean Held-Out Evaluator & Live TRL Dr. GRPO Trainer
# ---------------------------------------------------------------------------


def _evaluate_model_on_heldout(
    model: Any,
    tokenizer: Any,
    heldout_rows: Sequence[dict[str, Any]],
    *,
    step: int,
    k: int,
    backend: GradingBackend,
    eval_jsonl_path: Path,
    temperature: float = 0.8,
    max_new_tokens: int = 1024,
    batch_size: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Run k generations per clean held-out task and compute pass@1, pass@k, pass^k & Wilson CI."""
    import torch  # type: ignore[import-untyped]

    torch.manual_seed(seed + step)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device

    # Cap concurrent sequences <= 8 to avoid 24 GB VRAM KV-cache spikes when k=4
    eff_prompt_batch = max(1, min(batch_size, 8 // max(1, k)))

    task_pass_counts: dict[str, int] = {}
    task_pass1: list[float] = []
    task_passk: list[float] = []
    task_powk: list[float] = []
    first_rollout_passed: list[bool] = []
    total_tamper = 0
    total_rollouts = 0

    for start in range(0, len(heldout_rows), eff_prompt_batch):
        chunk = heldout_rows[start : start + eff_prompt_batch]
        prompts = [
            tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
            for r in chunk
        ]
        # Repeat each prompt k times for clean pass@k / pass^k estimation
        expanded_prompts = [p for p in prompts for _ in range(k)]
        expanded_dirs = [Path(r["root_path"]) for r in chunk for _ in range(k)]

        inputs = tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(k > 1 or temperature > 0.0),
                temperature=temperature if (k > 1 or temperature > 0.0) else 1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        completions = tokenizer.batch_decode(gen_ids[:, prompt_len:], skip_special_tokens=True)
        outcomes = grade_batch(
            list(zip(expanded_dirs, completions, strict=True)),
            mode="hardened",
            backend=backend,
        )

        for idx, row in enumerate(chunk):
            sub = outcomes[idx * k : (idx + 1) * k]
            successes = sum(1 for o in sub if o.passed)
            total_tamper += sum(1 for o in sub if o.tampered)
            total_rollouts += len(sub)
            task_pass_counts[row["task_id"]] = successes
            task_pass1.append(pass_at_k(k, successes, 1))
            task_passk.append(pass_at_k(k, successes, k))
            task_powk.append(pass_pow_k(k, successes, k))
            first_rollout_passed.append(sub[0].passed if sub else False)

    n_tasks = max(1, len(heldout_rows))
    mean_p1 = sum(task_pass1) / n_tasks
    mean_pk = sum(task_passk) / n_tasks
    mean_powk = sum(task_powk) / n_tasks
    ci_lo, ci_hi = wilson_ci(sum(1 for x in first_rollout_passed if x), len(first_rollout_passed))

    record = {
        "step": step,
        "k": k,
        "n_tasks": len(heldout_rows),
        "clean_heldout_pass_at_1": round(mean_p1, 6),
        "clean_heldout_pass_at_k": round(mean_pk, 6),
        "clean_heldout_pass_pow_k": round(mean_powk, 6),
        "pass_at_1_wilson_95_ci": [ci_lo, ci_hi],
        "heldout_tamper_rate": round(total_tamper / max(1, total_rollouts), 6),
        "per_task_successes": task_pass_counts,
        "per_task_first_passed": first_rollout_passed,
    }
    _append_jsonl(eval_jsonl_path, record)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()
    return record


def run_gpu_arm(
    *,
    cfg: LoRAAblationConfig,
    taskset_dir: Path,
    out_dir: Path,
    arm: ArmName,
    max_steps: int,
    seed: int,
    eval_every: int,
    quick_eval_every: int,
    eval_k: int,
    grading_backend: GradingBackend,
    disable_vllm: bool,
    vllm_gpu_memory_utilization: float,
    max_wall_minutes: float,
    resume_from_checkpoint: str | None,
) -> dict[str, Any]:
    """Execute Arm 0 (Base Eval), Arm 1 (Unhardened), or Arm 2 (dasv-Hardened) on a 24 GB GPU."""
    import torch  # type: ignore[import-untyped]
    from datasets import Dataset  # type: ignore[import-untyped]
    from peft import LoraConfig  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # type: ignore[import-untyped]
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-untyped]

    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    wall_limit_sec = max_wall_minutes * 60.0

    heldout_rows = _load_jsonl_rows(taskset_dir / "heldout_clean.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16
    eval_jsonl = out_dir / "heldout_eval.jsonl"
    metrics_jsonl = out_dir / "metrics.jsonl"
    drift_jsonl = out_dir / "drift.jsonl"

    # Arm 0: Base Model Clean Held-Out Baseline Only (no training!)
    if arm == "arm0":
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model,
            torch_dtype=dtype,
            device_map="auto",
        )
        eval_res = _evaluate_model_on_heldout(
            model,
            tokenizer,
            heldout_rows,
            step=0,
            k=eval_k,
            backend=grading_backend,
            eval_jsonl_path=eval_jsonl,
            seed=seed,
        )
        manifest = {
            "schema_version": "1.0.0",
            "arm": "arm0",
            "dasv_version": dasv_version,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "base_model": cfg.model.base_model,
            "seed": seed,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "peak_vram_gb": round(
                (torch.cuda.max_memory_allocated() / (1024**3))
                if torch.cuda.is_available()
                else 0.0,
                3,
            ),
            "wall_minutes": round((time.monotonic() - t_start) / 60.0, 2),
            "arm0_eval": eval_res,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    # Arm 1 (Unhardened) vs Arm 2 (dasv-Hardened)
    train_jsonl = (
        taskset_dir / "train_unhardened.jsonl"
        if arm == "arm1"
        else taskset_dir / "train_hardened.jsonl"
    )
    train_rows = _load_jsonl_rows(train_jsonl)
    train_ds = Dataset.from_list(train_rows)

    drift_tracker = LogprobDriftTracker(log_path=drift_jsonl)
    step_telemetry: dict[str, Any] = {
        "train_reward": 0.0,
        "clean_shadow_reward": 0.0,
        "tamper_rate": 0.0,
        "last_p95_delta": 0.0,
        "last_icepop_frac": 0.0,
        "last_heldout_p1": 0.0,
        "last_heldout_powk": 0.0,
    }

    grade_mode: Literal["hardened", "naive"] = "naive" if arm == "arm1" else "hardened"

    def reward_func(
        completions: list[Any],
        root_path: list[str],
        **kwargs: Any,
    ) -> list[float]:
        _ = kwargs
        pairs = [
            (Path(rp), _completion_to_text(comp))
            for comp, rp in zip(completions, root_path, strict=True)
        ]
        train_outcomes = grade_batch(pairs, mode=grade_mode, backend=grading_backend)
        rewards = [o.reward for o in train_outcomes]

        if arm == "arm1":
            # Off-policy clean shadow eval (NEVER fed to optimizer; quantifies true hacking gap)
            shadow_outcomes = grade_batch(
                pairs,
                mode="hardened",
                backend="hardened-subprocess",
                use_shadow_clean_test=True,
            )
            clean_mean = sum(o.reward for o in shadow_outcomes) / max(1, len(shadow_outcomes))
            tamper_cnt = sum(
                1
                for tr_o, sh_o in zip(train_outcomes, shadow_outcomes, strict=True)
                if tr_o.tampered or sh_o.tampered or (tr_o.passed and not sh_o.passed)
            )
        else:
            clean_mean = sum(rewards) / max(1, len(rewards))
            tamper_cnt = sum(1 for o in train_outcomes if o.tampered)

        step_telemetry["train_reward"] = sum(rewards) / max(1, len(rewards))
        step_telemetry["clean_shadow_reward"] = clean_mean
        step_telemetry["tamper_rate"] = tamper_cnt / max(1, len(train_outcomes))
        return rewards

    peft_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    raw_grpo_kwargs = build_trl_grpo_kwargs(
        cfg,
        output_dir=out_dir / "checkpoints",
        max_steps=max_steps,
        seed=seed,
        disable_vllm=disable_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    # Filter cleanly against installed TRL GRPOConfig signature while warning on skipped keys
    sig_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    supported_kwargs = {k: v for k, v in raw_grpo_kwargs.items() if k in sig_params}
    skipped_keys = sorted(set(raw_grpo_kwargs.keys()) - set(supported_kwargs.keys()))
    if skipped_keys:
        print(
            f"[train_dr_grpo] Note: TRL GRPOConfig skipped unsupported kwargs: {skipped_keys}",
            file=sys.stderr,
        )
    training_args = GRPOConfig(**supported_kwargs)

    class DasvDriftGRPOTrainer(GRPOTrainer):  # type: ignore[misc]
        """GRPOTrainer subclass recording live train-inference logprob drift via dasv."""

        _last_drift_step: int = -1

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            loss = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            cur_step = int(getattr(self.state, "global_step", 0))
            if cur_step != self._last_drift_step:
                try:
                    infer_lp = inputs.get("sampling_per_token_logprobs")
                    train_lp = inputs.get("old_per_token_logprobs")
                    mask = inputs.get("completion_mask")
                    if infer_lp is not None and train_lp is not None and mask is not None:
                        d_stats = drift_tracker.record_step(
                            train_lp,
                            infer_lp,
                            mask,
                            step=cur_step,
                        )
                        step_telemetry["last_p95_delta"] = d_stats.p95_abs_delta
                        step_telemetry["last_icepop_frac"] = d_stats.icepop_masked_frac
                        self._last_drift_step = cur_step
                    elif (
                        infer_lp is not None
                        and mask is not None
                        and hasattr(self, "_get_per_token_logprobs_and_entropies")
                        and "prompt_ids" in inputs
                        and "completion_ids" in inputs
                    ):
                        with torch.no_grad():
                            full_ids = torch.cat(
                                [inputs["prompt_ids"], inputs["completion_ids"]], dim=1
                            )
                            full_mask = torch.cat(
                                [inputs["prompt_mask"], inputs["completion_mask"]], dim=1
                            )
                            logits_to_keep = int(inputs["completion_ids"].size(1))
                            computed_lp, _ = self._get_per_token_logprobs_and_entropies(
                                model, full_ids, full_mask, logits_to_keep
                            )
                        d_stats = drift_tracker.record_step(
                            computed_lp,
                            infer_lp,
                            mask,
                            step=cur_step,
                        )
                        step_telemetry["last_p95_delta"] = d_stats.p95_abs_delta
                        step_telemetry["last_icepop_frac"] = d_stats.icepop_masked_frac
                        self._last_drift_step = cur_step
                except Exception as exc:
                    print(f"[train_dr_grpo] Drift hook warning: {exc}", file=sys.stderr)
            return loss

    class DasvEvalAndGuardCallback(TrainerCallback):  # type: ignore[misc]
        """Logs per-step metrics, runs periodic clean held-out eval, and enforces wall-time cap."""

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            _ = args
            step = int(state.global_step)
            model_ref = kwargs.get("model")

            if model_ref is not None and step > 0:
                if eval_every > 0 and (step % eval_every == 0 or step == max_steps):
                    ev = _evaluate_model_on_heldout(
                        model_ref,
                        tokenizer,
                        heldout_rows,
                        step=step,
                        k=eval_k,
                        backend=grading_backend,
                        eval_jsonl_path=eval_jsonl,
                        seed=seed,
                    )
                    step_telemetry["last_heldout_p1"] = ev["clean_heldout_pass_at_1"]
                    step_telemetry["last_heldout_powk"] = ev["clean_heldout_pass_pow_k"]
                elif quick_eval_every > 0 and step % quick_eval_every == 0:
                    ev = _evaluate_model_on_heldout(
                        model_ref,
                        tokenizer,
                        heldout_rows,
                        step=step,
                        k=1,
                        backend=grading_backend,
                        eval_jsonl_path=eval_jsonl,
                        seed=seed,
                    )
                    step_telemetry["last_heldout_p1"] = ev["clean_heldout_pass_at_1"]

            tr_r = float(step_telemetry["train_reward"])
            sh_r = float(step_telemetry["clean_shadow_reward"])
            _append_jsonl(
                metrics_jsonl,
                {
                    "step": step,
                    "arm": arm,
                    "train_reward": round(tr_r, 6),
                    "clean_shadow_reward": round(sh_r, 6),
                    "hack_divergence_gap": round(tr_r - sh_r, 6),
                    "tamper_rate": round(float(step_telemetry["tamper_rate"]), 6),
                    "clean_heldout_pass_at_1": round(float(step_telemetry["last_heldout_p1"]), 6),
                    "clean_heldout_pass_pow_k": round(
                        float(step_telemetry["last_heldout_powk"]), 6
                    ),
                    "p95_abs_delta": round(float(step_telemetry["last_p95_delta"]), 6),
                    "icepop_masked_frac": round(float(step_telemetry["last_icepop_frac"]), 6),
                    "peak_vram_gb": round(
                        (torch.cuda.max_memory_allocated() / (1024**3))
                        if torch.cuda.is_available()
                        else 0.0,
                        3,
                    ),
                    "elapsed_sec": round(time.monotonic() - t_start, 2),
                },
            )

            if (time.monotonic() - t_start) >= wall_limit_sec:
                print(
                    f"[train_dr_grpo] Wall-time safety cap ({max_wall_minutes}m) reached at "
                    f"step {step}; saving checkpoint and stopping cleanly.",
                    file=sys.stderr,
                )
                control.should_save = True
                control.should_training_stop = True
            return control

    trainer = DasvDriftGRPOTrainer(
        model=cfg.model.base_model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        callbacks=[DasvEvalAndGuardCallback()],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    adapter_dir = out_dir / "final_lora_adapter"
    trainer.save_model(str(adapter_dir))

    manifest = {
        "schema_version": "1.0.0",
        "arm": arm,
        "dasv_version": dasv_version,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_model": cfg.model.base_model,
        "seed": seed,
        "max_steps": max_steps,
        "grading_backend": grading_backend,
        "skipped_trl_kwargs": skipped_keys,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_vram_gb": round(
            (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0,
            3,
        ),
        "wall_minutes": round((time.monotonic() - t_start) / 60.0, 2),
        "final_heldout_pass_at_1": step_telemetry["last_heldout_p1"],
        "final_heldout_pass_pow_k": step_telemetry["last_heldout_powk"],
        "final_p95_abs_delta": step_telemetry["last_p95_delta"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dasv LoRA Without Regret + Dr. GRPO 3-arm ablation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("recipes/lora_dr_grpo_ablation/config.yaml"),
    )
    parser.add_argument("--taskset-dir", type=Path, default=Path("experiments/taskset_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/v1_ablation/arm2"))
    parser.add_argument("--arm", choices=("arm0", "arm1", "arm2"), default="arm2")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--quick-eval-every", type=int, default=25)
    parser.add_argument("--eval-k", type=int, default=4)
    parser.add_argument(
        "--grading-backend",
        choices=("hardened-subprocess", "docker", "modal-sandbox"),
        default="hardened-subprocess",
    )
    parser.add_argument("--disable-vllm", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-wall-minutes", type=float, default=180.0)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_ablation_recipe_config(args.config)
    if args.dry_run:
        manifest = run_dry_run(
            cfg=cfg,
            taskset_dir=args.taskset_dir,
            out_dir=args.out_dir,
            arm=cast(ArmName, args.arm),
            seed=args.seed,
        )
        print(
            f"[train_dr_grpo] Dry-run PASS ($0): model={manifest['base_model']}, "
            f"tamper_check={manifest['tamper_differential_check']}"
        )
        return

    manifest = run_gpu_arm(
        cfg=cfg,
        taskset_dir=args.taskset_dir,
        out_dir=args.out_dir,
        arm=cast(ArmName, args.arm),
        max_steps=args.max_steps,
        seed=args.seed,
        eval_every=args.eval_every,
        quick_eval_every=args.quick_eval_every,
        eval_k=args.eval_k,
        grading_backend=cast(GradingBackend, args.grading_backend),
        disable_vllm=args.disable_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_wall_minutes=args.max_wall_minutes,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"[train_dr_grpo] Completed {args.arm}: {json.dumps(manifest)}")


if __name__ == "__main__":
    main()
