"""CPU unit & adversarial tests for Phase 3 prepare_taskset, train_dr_grpo, and plot_ablation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from recipes.lora_dr_grpo_ablation.plot_ablation import (
    build_ablation_svgs,
    compute_ablation_significance,
    render_ablation_comparison_html,
)
from recipes.lora_dr_grpo_ablation.prepare_taskset import (
    build_synthetic_corpus,
    prepare_ablation_taskset,
    verify_task_oracle_and_noop,
)
from recipes.lora_dr_grpo_ablation.train_dr_grpo import (
    build_scrubbed_env,
    build_trl_grpo_kwargs,
    extract_drift_tensors,
    grade_completion_subprocess,
    mcnemar_exact_p_value,
    paired_bootstrap_test,
    run_dry_run,
    wilson_ci,
)

from dasv.drift import load_ablation_recipe_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "recipes" / "lora_dr_grpo_ablation" / "config.yaml"


def test_prepare_synthetic_taskset_disjoint_and_audit_contrast(tmp_path: Path) -> None:
    """Synthetic Harbor taskset builds disjoint splits where Arm 1 fails Tier 0 & Arm 2 passes."""
    manifest = prepare_ablation_taskset(
        out_dir=tmp_path,
        source="synthetic",
        n_train=12,
        n_heldout=8,
        seed=42,
    )
    assert manifest["n_train"] == 12
    assert manifest["n_heldout"] == 8
    assert manifest["dasv_tier0_audit"]["arm1_tasks_failing_tier0"] == 12
    assert manifest["dasv_tier0_audit"]["arm2_tasks_clean_tier0"] == 12
    assert manifest["arm1_defect_breakdown"]["RULE-BNDY-001"] >= 2
    assert manifest["arm1_defect_breakdown"]["RULE-MUT-002"] >= 5

    train_rows = [
        json.loads(line)
        for line in (tmp_path / "train_hardened.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    held_rows = [
        json.loads(line)
        for line in (tmp_path / "heldout_clean.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {r["task_id"] for r in train_rows}.isdisjoint({r["task_id"] for r in held_rows})


def test_naive_vs_hardened_tamper_differential_and_oracle_sanity(tmp_path: Path) -> None:
    """Verify Arm 1 naive grader is exploitable via tests/ edit while Arm 2 forces reward=0.0."""
    prepare_ablation_taskset(
        out_dir=tmp_path,
        source="synthetic",
        n_train=10,
        n_heldout=4,
        seed=42,
    )
    arm1_rows = [
        json.loads(ln)
        for ln in (tmp_path / "train_unhardened.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    arm2_by_id = {
        json.loads(ln)["task_id"]: Path(json.loads(ln)["root_path"])
        for ln in (tmp_path / "train_hardened.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }

    bndy_row = next(r for r in arm1_rows if r["defect_kind"] == "RULE-BNDY-001")
    arm1_dir = Path(bndy_row["root_path"])
    arm2_dir = arm2_by_id[bndy_row["task_id"]]

    # 1. Adversarial completion overwriting tests/test_solution.py
    tamper_reply = (
        "```python\n# file: tests/test_solution.py\ndef test_always_pass():\n    assert True\n```\n"
    )
    naive_out = grade_completion_subprocess(arm1_dir, tamper_reply, mode="naive")
    hard_out = grade_completion_subprocess(arm2_dir, tamper_reply, mode="hardened")
    shadow_out = grade_completion_subprocess(
        arm1_dir, tamper_reply, mode="hardened", use_shadow_clean_test=True
    )

    assert naive_out.reward == 1.0
    assert naive_out.tampered is True
    assert hard_out.reward == 0.0
    assert hard_out.tampered is True
    assert shadow_out.reward == 0.0
    assert shadow_out.tampered is True

    # 2. Path traversal escape attempt must be blocked in both modes
    traversal_reply = (
        "```python\n# file: ../../escape.py\nprint('pwned')\n```\n"
        "```python\n# file: solution.py\npass\n```\n"
    )
    assert grade_completion_subprocess(arm1_dir, traversal_reply, mode="naive").reward == 0.0
    assert grade_completion_subprocess(arm2_dir, traversal_reply, mode="hardened").reward == 0.0

    # 3. Real canonical oracle code must pass hardened verification (reward == 1.0)
    solve_sh = (arm2_dir / "solution" / "solve.sh").read_text(encoding="utf-8")
    # Extract json-encoded python source from solve.sh
    start = solve_sh.index('write_text("') + len("write_text(")
    end = solve_sh.rindex(", encoding=")
    oracle_src = json.loads(solve_sh[start:end])
    oracle_reply = f"```python\n# file: solution.py\n{oracle_src}\n```\n"
    oracle_out = grade_completion_subprocess(arm2_dir, oracle_reply, mode="hardened")
    assert oracle_out.reward == 1.0
    assert oracle_out.passed is True
    assert oracle_out.tampered is False


def test_scrubbed_env_strips_provider_and_hf_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grading subprocesses must never inherit HF_TOKEN, OPENAI_API_KEY, or cloud secrets."""
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token_value")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-key-value")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")

    scrubbed = build_scrubbed_env({"PYTHONPATH": "/tmp/ws"})
    assert "HF_TOKEN" not in scrubbed
    assert "OPENAI_API_KEY" not in scrubbed
    assert "OPENROUTER_API_KEY" not in scrubbed
    assert "AWS_SECRET_ACCESS_KEY" not in scrubbed
    assert scrubbed["PYTHONHASHSEED"] == "0"
    assert scrubbed["PYTHONPATH"] == "/tmp/ws"
    # Confirm parent process env still intact
    assert os.environ["HF_TOKEN"] == "hf_secret_token_value"


def test_vram_safe_trl_kwargs_and_cpu_dry_run(tmp_path: Path) -> None:
    """Verify 24 GB VRAM-safe TRL Dr. GRPO kwargs and full CPU dry-run execution ($0)."""
    cfg = load_ablation_recipe_config(_CONFIG_PATH)
    kwargs = build_trl_grpo_kwargs(cfg, output_dir=tmp_path / "ckpt", max_steps=200, seed=42)

    # Guardrail 1: BF16 policy load + decoupled micro-batch prevents 24 GB L4 OOM
    assert kwargs["model_init_kwargs"] == {"dtype": "bfloat16", "device_map": None}
    assert kwargs["per_device_train_batch_size"] == 4
    assert kwargs["gradient_accumulation_steps"] == 8
    assert kwargs["steps_per_generation"] == 8
    assert kwargs["num_generations"] == 8
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["vllm_mode"] == "colocate"
    assert kwargs["vllm_enable_sleep_mode"] is True
    assert kwargs["vllm_gpu_memory_utilization"] == pytest.approx(0.35)
    # Guardrail 2: Dr. GRPO + DAPO Clip-Higher + clean drift sampling params
    assert kwargs["loss_type"] == "dr_grpo"
    assert kwargs["scale_rewards"] == "none"
    assert kwargs["epsilon"] == pytest.approx(0.20)
    assert kwargs["epsilon_high"] == pytest.approx(0.28)
    assert kwargs["repetition_penalty"] == 1.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["top_k"] == -1
    assert "logprobs_mode" not in kwargs

    # Verify CLI fallback kwargs: bs=2 keeps 32 rollouts/step, disable_vllm sets top_k=None
    kw_fb = build_trl_grpo_kwargs(
        cfg,
        output_dir=tmp_path / "ckpt_fb",
        max_steps=10,
        seed=42,
        disable_vllm=True,
        vllm_enable_sleep_mode=False,
        per_device_train_batch_size=2,
    )
    assert kw_fb["per_device_train_batch_size"] == 2
    assert kw_fb["gradient_accumulation_steps"] == 16
    assert kw_fb["steps_per_generation"] == 16
    assert kw_fb["vllm_enable_sleep_mode"] is False
    assert kw_fb["top_k"] is None

    # Verify TRL 0.27+ `sampling_per_token_logps` + captured policy logps drift extraction
    tr_lp, inf_lp, msk = extract_drift_tensors(
        {
            "sampling_per_token_logps": [-0.5, -1.0],
            "old_per_token_logps": None,
            "completion_mask": [1, 1],
        },
        captured_policy_logps=[-0.48, -1.03],
    )
    assert tr_lp == [-0.48, -1.03]
    assert inf_lp == [-0.5, -1.0]
    assert msk == [1, 1]

    # Full CPU dry-run
    ts_dir = tmp_path / "taskset"
    prepare_ablation_taskset(out_dir=ts_dir, source="synthetic", n_train=8, n_heldout=4, seed=42)
    run_out = tmp_path / "dry_arm2"
    manifest = run_dry_run(cfg=cfg, taskset_dir=ts_dir, out_dir=run_out, arm="arm2", seed=42)
    assert manifest["tamper_differential_check"]["tested"] is True
    assert manifest["tamper_differential_check"]["naive_reward"] == 1.0
    assert manifest["tamper_differential_check"]["hardened_reward"] == 0.0
    assert (run_out / "manifest.json").is_file()
    assert (run_out / "metrics.jsonl").is_file()
    assert (run_out / "drift.jsonl").is_file()


def test_statistical_tests_and_ablation_svg_plotter() -> None:
    """Verify Wilson CI, paired bootstrap, exact McNemar p-value, and zero-CDN HTML/SVG output."""
    lo, hi = wilson_ci(32, 64)
    assert 0.37 < lo < 0.40
    assert 0.60 < hi < 0.63

    # Clear significant improvement: Arm 2 beats Arm 1 on 18 tasks, loses on 2
    scores_a = [0.0] * 18 + [1.0] * 2 + [1.0] * 20 + [0.0] * 24
    scores_b = [1.0] * 18 + [0.0] * 2 + [1.0] * 20 + [0.0] * 24
    boot = paired_bootstrap_test(scores_a, scores_b, n_boot=2000, seed=42)
    assert boot["delta_mean"] == pytest.approx(16.0 / 64.0, abs=1e-5)
    assert boot["ci_low"] > 0.0
    assert boot["p_value_two_sided"] < 0.01

    mcn = mcnemar_exact_p_value([x > 0.5 for x in scores_a], [x > 0.5 for x in scores_b])
    assert mcn["only_a"] == 2
    assert mcn["only_b"] == 18
    assert float(mcn["p_value_two_sided"]) < 0.001

    u_rows = [
        {
            "step": 1,
            "train_reward": 0.35,
            "hack_divergence_gap": 0.0,
            "tamper_rate": 0.0,
            "p95_abs_delta": 0.01,
            "icepop_masked_frac": 0.0,
        },
        {
            "step": 160,
            "train_reward": 0.82,
            "clean_heldout_pass_at_1": 0.38,
            "hack_divergence_gap": 0.44,
            "tamper_rate": 0.25,
            "p95_abs_delta": 0.14,
            "icepop_masked_frac": 0.04,
        },
    ]
    h_rows = [
        {
            "step": 1,
            "train_reward": 0.35,
            "hack_divergence_gap": 0.0,
            "tamper_rate": 0.0,
            "p95_abs_delta": 0.01,
            "icepop_masked_frac": 0.0,
        },
        {
            "step": 160,
            "train_reward": 0.64,
            "clean_heldout_pass_at_1": 0.62,
            "hack_divergence_gap": 0.02,
            "tamper_rate": 0.01,
            "p95_abs_delta": 0.05,
            "icepop_masked_frac": 0.01,
        },
    ]
    sig = compute_ablation_significance(
        {
            "k": 4,
            "per_task_successes": {f"t{i}": int(s * 4) for i, s in enumerate(scores_a)},
            "per_task_first_passed": [s > 0.5 for s in scores_a],
        },
        {
            "k": 4,
            "per_task_successes": {f"t{i}": int(s * 4) for i, s in enumerate(scores_b)},
            "per_task_first_passed": [s > 0.5 for s in scores_b],
        },
    )
    svgs = build_ablation_svgs(u_rows, h_rows, arm0_pass_at_1=0.35)
    assert set(svgs.keys()) == {
        "ablation_reward_and_heldout.svg",
        "ablation_hacking_and_tamper.svg",
        "ablation_logprob_drift.svg",
    }
    html_out = render_ablation_comparison_html(
        u_rows, h_rows, arm0_pass_at_1=0.35, significance=sig
    )
    assert "Arm 0 (Base Model)" in html_out
    assert "Statistical Significance" in html_out
    assert "http://" not in html_out and "https://" not in html_out


def test_all_synthetic_corpus_tasks_pass_oracle_and_fail_noop() -> None:
    """Every offline synthetic task's canonical oracle passes and starter stub fails."""
    corpus = build_synthetic_corpus()
    assert len(corpus) == 20
    for task in corpus:
        assert verify_task_oracle_and_noop(task), f"Oracle/noop sanity failed on {task.task_id}"
