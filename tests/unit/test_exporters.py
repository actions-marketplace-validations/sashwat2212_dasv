"""Unit & CLI schema round-trip + golden determinism tests for FEAT-18 exporters."""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from dasv.cli.app import app
from dasv.core.errors import EXIT_CONFIG_ERROR, EXIT_PASS, ConfigError
from dasv.core.schemas import CurriculumBucket, ExploitTrace
from dasv.export import (
    OpenEnvTrainingTrace,
    PrimeRLSplitManifest,
    TRLDatasetRow,
    VerlDatasetRow,
    export_openenv_traces,
    export_prime_rl,
    export_trl,
    export_verl,
    load_curriculum_entries,
    load_exploit_traces,
)
from dasv.profile.signal_profiler import CurriculumEntry

runner = CliRunner()


def _sample_curriculum_entries(tmp_path: Path) -> tuple[Path, Path, list[CurriculumEntry]]:
    task_dir = tmp_path / "calc_task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text(
        "Implement safe integer division divide(a, b) raising ValueError when b == 0.\n",
        encoding="utf-8",
    )
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    entries = [
        CurriculumEntry(
            task_id="hardened-div",
            root_path=str(task_dir),
            standard="harbor",
            bucket=CurriculumBucket.GOLDILOCKS,
            base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
            rollouts_g=8,
            pass_at_1=0.25,
            pass_at_k={1: 0.25, 4: 0.785714, 8: 1.0},
            pass_pow_k={1: 0.25, 4: 0.00390625, 8: 0.00001526},
            advantage_variance=0.1875,
            zero_grad_prob=0.10011292,
            zero_grad_risk=False,
            recommended_sampler_weight=0.75,
        ),
        CurriculumEntry(
            task_id="trivial-hello",
            root_path=str(task_dir),
            standard="harbor",
            bucket=CurriculumBucket.TRIVIAL_ZERO_GRAD,
            base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
            rollouts_g=8,
            pass_at_1=1.0,
            pass_at_k={1: 1.0, 8: 1.0},
            pass_pow_k={1: 1.0, 8: 1.0},
            advantage_variance=0.0,
            zero_grad_prob=1.0,
            zero_grad_risk=True,
            recommended_sampler_weight=0.0,
        ),
    ]
    jsonl_path = tmp_path / "curriculum.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(e.model_dump(mode="json")) for e in entries) + "\n",
        encoding="utf-8",
    )
    yaml_path = tmp_path / "curriculum.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"tasks": [e.model_dump(mode="json") for e in entries]}),
        encoding="utf-8",
    )
    return jsonl_path, yaml_path, entries


def _sample_exploits_jsonl(tmp_path: Path) -> Path:
    trace = ExploitTrace(
        exploit_id="exp-hardco-deadbeef",
        task_id="hardened-div",
        hack_class="hardcoding_lookup",
        reward_obtained=1.0,
        steps=[
            {
                "turn": 1,
                "tool_call_id": "call_1",
                "action": {"tool": "write_file", "path": "div.py", "exit_code": 0},
                "observation_preview": "Wrote div.py",
                "fence_nonce": "abc123",
                "loss_mask": 0,
            },
            {
                "event": "clean_image_grade",
                "reward": 1.0,
                "artifact_sha256": "0" * 64,
                "verifier_exit_code": 0,
                "tamper_attempts": ["tests/test.sh"],
            },
        ],
        proposed_fix_diff=None,
        gold_sanity_status="not_run",
    )
    path = tmp_path / "exploits.jsonl"
    path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    return path


def test_trl_verl_prime_rl_and_openenv_schema_roundtrips_and_golden_determinism(
    tmp_path: Path,
) -> None:
    """Verify schema round-trip, determinism, reward_func syntax, TOML & loss_mask invariants."""
    jsonl_path, yaml_path, _entries = _sample_curriculum_entries(tmp_path)
    exploits_path = _sample_exploits_jsonl(tmp_path)

    gold_jsonl = load_curriculum_entries(jsonl_path, include_buckets=[CurriculumBucket.GOLDILOCKS])
    gold_yaml = load_curriculum_entries(yaml_path, include_buckets=[CurriculumBucket.GOLDILOCKS])
    assert len(gold_jsonl) == 1
    assert gold_jsonl == gold_yaml

    # 1. TRL export: deterministic manifest SHA + valid Python reward_func wrapper
    trl_out_1 = tmp_path / "trl_1" / "dataset.jsonl"
    trl_out_2 = tmp_path / "trl_2" / "dataset.jsonl"
    res_trl_1 = export_trl(gold_jsonl, trl_out_1)
    res_trl_2 = export_trl(gold_jsonl, trl_out_2)
    assert res_trl_1.manifest_sha == res_trl_2.manifest_sha
    assert trl_out_1.read_text(encoding="utf-8") == trl_out_2.read_text(encoding="utf-8")
    trl_rows = [json.loads(ln) for ln in trl_out_1.read_text(encoding="utf-8").splitlines() if ln]
    assert len(trl_rows) == 1
    parsed_trl = TRLDatasetRow.model_validate(trl_rows[0])
    assert parsed_trl.task_id == "hardened-div"
    assert "safe integer division" in parsed_trl.prompt[0]["content"]
    assert res_trl_1.companion_paths[0].name == "trl_reward_func.py"
    ast.parse(res_trl_1.companion_paths[0].read_text(encoding="utf-8"))

    # 2. veRL export: HybridFlow RLHFDataset JSONL schema with dasv_manifest_sha
    verl_out = tmp_path / "verl.jsonl"
    res_verl = export_verl(gold_jsonl, verl_out)
    assert res_verl.record_count == 1
    verl_row = VerlDatasetRow.model_validate(
        json.loads(verl_out.read_text(encoding="utf-8").strip())
    )
    assert verl_row.data_source == "dasv/harbor"
    assert verl_row.reward_model.style == "rule"
    assert verl_row.extra_info["dasv_manifest_sha"] == res_verl.manifest_sha

    # 3. prime-rl export: JSON split manifest + tomllib-parseable rl_config.toml
    prime_out = tmp_path / "prime" / "split.json"
    res_prime = export_prime_rl(gold_jsonl, prime_out)
    manifest = PrimeRLSplitManifest.model_validate_json(prime_out.read_text(encoding="utf-8"))
    assert manifest.selected_task_ids == ["hardened-div"]
    toml_path = res_prime.companion_paths[0]
    parsed_toml = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    assert parsed_toml["environment"]["task_ids"] == ["hardened-div"]
    assert parsed_toml["training"]["objective"] == "dr_grpo"

    # 4. OpenEnv TrainingTrace export: exact loss_mask (1=policy action, 0=env observation)
    traces = load_exploit_traces(exploits_path)
    oe_out = tmp_path / "openenv_traces.jsonl"
    res_oe = export_openenv_traces(exploits=traces, entries=gold_jsonl, out_path=oe_out)
    assert res_oe.record_count == 2
    oe_lines = [json.loads(ln) for ln in oe_out.read_text(encoding="utf-8").splitlines() if ln]
    ex_trace = OpenEnvTrainingTrace.model_validate(oe_lines[0])
    assert ex_trace.source == "exploit_trace"
    assert ex_trace.tamper_flags == ["tests/test.sh"]
    assert ex_trace.turns[0].role == "assistant" and ex_trace.turns[0].loss_mask == 1
    assert ex_trace.turns[1].role == "tool" and ex_trace.turns[1].loss_mask == 0
    assert ex_trace.loss_mask == [1, 0]

    seed_trace = OpenEnvTrainingTrace.model_validate(oe_lines[1])
    assert seed_trace.source == "curriculum_seed"
    assert seed_trace.turns[0].role == "user" and seed_trace.turns[0].loss_mask == 0

    with pytest.raises(ConfigError):
        export_openenv_traces(exploits=(), entries=(), out_path=tmp_path / "empty.jsonl")


def test_cli_export_all_four_formats_and_error_contracts(tmp_path: Path) -> None:
    """Verify `dasv export` CLI across trl, verl, prime_rl, openenv + exit 2 on invalid args."""
    curr_jsonl, _curr_yaml, _entries = _sample_curriculum_entries(tmp_path)
    exploits_jsonl = _sample_exploits_jsonl(tmp_path)

    for fmt, out_name in (
        ("trl", "out_trl.jsonl"),
        ("verl", "out_verl.jsonl"),
        ("prime_rl", "out_prime.json"),
    ):
        out_p = tmp_path / out_name
        res = runner.invoke(
            app,
            ["export", "--format", fmt, "--curriculum", str(curr_jsonl), "--out", str(out_p)],
        )
        assert res.exit_code == EXIT_PASS, res.stdout + res.stderr
        assert out_p.is_file()

    oe_out = tmp_path / "out_openenv.jsonl"
    res_oe = runner.invoke(
        app,
        [
            "export",
            "--format",
            "openenv",
            "--exploits",
            str(exploits_jsonl),
            "--curriculum",
            str(curr_jsonl),
            "--out",
            str(oe_out),
        ],
    )
    assert res_oe.exit_code == EXIT_PASS, res_oe.stdout + res_oe.stderr
    assert oe_out.is_file()

    # Invalid format -> EXIT_CONFIG_ERROR (2)
    assert (
        runner.invoke(
            app,
            [
                "export",
                "--format",
                "unknown_trainer",
                "--curriculum",
                str(curr_jsonl),
                "--out",
                str(tmp_path / "x.jsonl"),
            ],
        ).exit_code
        == EXIT_CONFIG_ERROR
    )
    # Missing --curriculum for TRL -> EXIT_CONFIG_ERROR (2)
    assert (
        runner.invoke(
            app,
            ["export", "--format", "trl", "--out", str(tmp_path / "x.jsonl")],
        ).exit_code
        == EXIT_CONFIG_ERROR
    )
