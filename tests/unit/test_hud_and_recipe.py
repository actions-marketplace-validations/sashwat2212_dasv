"""Unit tests for LoRA Dr.GRPO recipe guardrails, ablation plotter, and HUD adapter (FEAT-04/18)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from dasv.adapters.hud import HudAdapter
from dasv.adapters.registry import AdapterRegistry
from dasv.core.errors import ConfigError
from dasv.core.schemas import EnvStandard
from dasv.drift import load_ablation_recipe_config
from dasv.sandbox.probes import run_tier0_static_probes

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_DIR = REPO_ROOT / "recipes" / "lora_dr_grpo_ablation"


def test_lora_dr_grpo_recipe_config_passes_guardrails_and_rejects_violations(
    tmp_path: Path,
) -> None:
    """Validate shipped config.yaml and verify LoRA Without Regret rules reject bad configs."""
    cfg = load_ablation_recipe_config(RECIPE_DIR / "config.yaml")
    assert cfg.lora.r == 16
    assert cfg.optimizer.prompts_per_step < 32
    assert cfg.rl_objective.variant == "dr_grpo"
    assert cfg.rl_objective.clip_high >= cfg.rl_objective.clip_low

    # 1. Attention-only LoRA (missing MLP projections) MUST be rejected
    bad_lora_cfg = yaml.safe_load((RECIPE_DIR / "config.yaml").read_text(encoding="utf-8"))
    bad_lora_cfg["lora"]["target_modules"] = ["q_proj", "v_proj"]
    bad_lora_path = tmp_path / "bad_lora.yaml"
    bad_lora_path.write_text(yaml.safe_dump(bad_lora_cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="Rule 1"):
        load_ablation_recipe_config(bad_lora_path)

    # 2. Effective prompt batch >= 32 MUST be rejected (Rule 4)
    bad_batch_cfg = yaml.safe_load((RECIPE_DIR / "config.yaml").read_text(encoding="utf-8"))
    bad_batch_cfg["optimizer"]["prompts_per_step"] = 16
    bad_batch_cfg["optimizer"]["gradient_accumulation_steps"] = 2
    bad_batch_path = tmp_path / "bad_batch.yaml"
    bad_batch_path.write_text(yaml.safe_dump(bad_batch_cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="Rule 4"):
        load_ablation_recipe_config(bad_batch_path)


def test_ablation_plotter_renders_zero_cdn_svg_html(tmp_path: Path) -> None:
    """Verify plot_ablation.py renders offline zero-CDN SVG curves from synthetic Arm 1/2 JSONL."""
    spec = importlib.util.spec_from_file_location("plot_ablation", RECIPE_DIR / "plot_ablation.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    arm1_path = tmp_path / "arm1.jsonl"
    arm2_path = tmp_path / "arm2.jsonl"
    arm1_rows = [
        {"step": 0, "train_reward": 0.20, "clean_heldout_pass_at_1": 0.20, "p95_abs_delta": 0.01},
        {"step": 100, "train_reward": 0.92, "clean_heldout_pass_at_1": 0.22, "p95_abs_delta": 0.14},
    ]
    arm2_rows = [
        {"step": 0, "train_reward": 0.20, "clean_heldout_pass_at_1": 0.20, "p95_abs_delta": 0.01},
        {"step": 100, "train_reward": 0.64, "clean_heldout_pass_at_1": 0.61, "p95_abs_delta": 0.03},
    ]
    arm1_path.write_text("\n".join(json.dumps(r) for r in arm1_rows) + "\n", encoding="utf-8")
    arm2_path.write_text("\n".join(json.dumps(r) for r in arm2_rows) + "\n", encoding="utf-8")

    plotter = cast(Any, mod)
    html_out = plotter.render_ablation_comparison_html(
        plotter.load_arm_metrics(arm1_path),
        plotter.load_arm_metrics(arm2_path),
    )
    assert "<svg" in html_out
    assert "Arm 1 (Unhardened)" in html_out
    assert "Arm 2 (dasv-Hardened)" in html_out
    assert "http://" not in html_out and "https://" not in html_out


def test_hud_adapter_entry_point_detection_loading_and_static_probes(tmp_path: Path) -> None:
    """Verify FEAT-04 HudAdapter plugin discovery, hud.toml & pyproject loading, and T0 probes."""
    # 1. hud.toml-based HUD env with pinned image and tasks/grade.py
    hud_dir = tmp_path / "hud_task"
    (hud_dir / "tasks").mkdir(parents=True)
    (hud_dir / "hud.toml").write_text(
        (
            'name = "hud-mcp-demo"\n'
            'image = "python:3.12-slim@sha256:' + ("a" * 64) + '"\n'
            'network_mode = "no-network"\n'
        ),
        encoding="utf-8",
    )
    (hud_dir / "README.md").write_text("Complete the MCP browser workflow.\n", encoding="utf-8")
    (hud_dir / "tasks" / "grade.py").write_text("def grade(): return 1.0\n", encoding="utf-8")

    registry = AdapterRegistry(load_entry_points=True)
    adapter = registry.resolve(hud_dir)
    assert isinstance(adapter, HudAdapter)
    spec = adapter.load(hud_dir)
    assert spec.standard == EnvStandard.HUD
    assert spec.task_id == "hud-mcp-demo"
    assert spec.verifier_paths == ["tasks/grade.py"]
    assert "MCP browser workflow" in spec.instruction_text
    assert run_tier0_static_probes(spec) == []

    # 2. pyproject.toml with hud-python dependency also auto-detects as HUD
    hud_py_dir = tmp_path / "hud_py_pkg"
    hud_py_dir.mkdir()
    (hud_py_dir / "pyproject.toml").write_text(
        '[project]\nname = "hud-pkg-env"\ndependencies = ["hud-python>=0.2"]\n',
        encoding="utf-8",
    )
    assert HudAdapter().detect(hud_py_dir) is True
    spec_py = HudAdapter().load(hud_py_dir)
    assert spec_py.task_id == "hud-pkg-env"
