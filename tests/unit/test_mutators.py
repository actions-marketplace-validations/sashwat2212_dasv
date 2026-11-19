"""Unit tests for $0 Anti-Patch and ImpossibleBench mutation suite (SDD-02, FEAT-11)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dasv.adapters.harbor import HarborAdapter
from dasv.core.schemas import EnvSpec, Severity
from dasv.mutate import (
    MutationSuiteRunner,
    OneOffMutation,
    generate_anti_patches,
    generate_conflicting_spec_variants,
    generate_one_off_mutations,
)
from dasv.sandbox.provider import ExecResult, SandboxHandle
from tests.conftest import FakeSandboxProvider


def _populate_python_task(env_dir: Path) -> EnvSpec:
    (env_dir / "src").mkdir(exist_ok=True)
    (env_dir / "src" / "calculator.py").write_text(
        (
            '"""Calculator module."""\n\n'
            "def divide(a: float, b: float) -> float:\n"
            '    """Divide a by b safely."""\n'
            "    if b == 0:\n"
            '        raise ValueError("zero divisor")\n'
            "    return a / b\n"
        ),
        encoding="utf-8",
    )
    (env_dir / "tests" / "test_calc.py").write_text(
        (
            "import unittest\n\n"
            "def test_basic() -> None:\n"
            "    assert 10 / 2 == 5\n"
            "    assert 10 > 2\n\n"
            "class CalcCase(unittest.TestCase):\n"
            "    def test_unit(self) -> None:\n"
            "        self.assertEqual(2 + 2, 4)\n"
        ),
        encoding="utf-8",
    )
    return HarborAdapter().load(env_dir)


def test_all_five_anti_patch_mutators_generate_valid_python_and_diffs(
    hardened_harbor_env: Path,
) -> None:
    spec = _populate_python_task(hardened_harbor_env)
    mutations = generate_anti_patches(spec)

    mutator_ids = [m.mutator_id for m in mutations]
    assert mutator_ids == [
        "MUT-AP-001",
        "MUT-AP-002",
        "MUT-AP-003",
        "MUT-AP-004",
        "MUT-AP-005",
    ]
    for mut in mutations:
        assert mut.target_file == "src/calculator.py"
        assert "tests/" not in mut.target_file
        assert mut.patch_diff.startswith("diff --git a/src/calculator.py b/src/calculator.py\n")
        assert mut.mutated_source is not None
        # Every synthesized Python source MUST compile cleanly
        ast.parse(mut.mutated_source)

    by_id = {m.mutator_id: m.mutated_source or "" for m in mutations}
    assert "return None" in by_id["MUT-AP-002"]
    assert "except Exception:" in by_id["MUT-AP-003"]
    assert "sys.exit(0)" in by_id["MUT-AP-004"]
    assert "ValueError" not in by_id["MUT-AP-005"]


def test_impossible_bench_one_off_and_conflicting_generation(
    hardened_harbor_env: Path,
) -> None:
    spec = _populate_python_task(hardened_harbor_env)

    one_offs = generate_one_off_mutations(spec, max_mutations=5)
    assert len(one_offs) == 3
    sources = [m.mutated_test_source for m in one_offs]
    for src in sources:
        ast.parse(src)
    assert any("!=" in s for s in sources)
    assert any("<=" in s for s in sources)
    assert any("assertNotEqual" in s for s in sources)

    variants = generate_conflicting_spec_variants(spec)
    assert len(variants) == 1
    variant = variants[0]
    ast.parse(variant.mutated_test_source)
    assert "test_dasv_impossible_conflicting_spec" in variant.mutated_test_source

    # Executing the injected conflicting test MUST raise AssertionError (guaranteed unsolvable)
    ns: dict[str, object] = {}
    exec(compile(variant.mutated_test_source, "<conflicting>", "exec"), ns)
    fn = ns["test_dasv_impossible_conflicting_spec"]
    assert callable(fn)
    with pytest.raises(AssertionError, match="ImpossibleBench"):
        fn()


async def test_mutation_runner_flags_weak_verifier_and_dead_assertions(
    hardened_harbor_env: Path,
    tmp_path: Path,
) -> None:
    spec = _populate_python_task(hardened_harbor_env)

    # 1. Weak verifier: lets MUT-AP-002 (stub) & MUT-AP-003 (swallow) pass (survival=40% CRITICAL)
    patch_seen_by_ctr: dict[str, str] = {}

    def weak_exec(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (timeout_sec, workdir)
        joined = " ".join(cmd)
        if "git', 'apply'" in joined:
            # Capture base64-decoded payload snippet via cmd argument inspection
            patch_seen_by_ctr[handle.container_id] = joined
            return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)
        if "test.sh" in joined:
            # First two containers (MUT-AP-001, MUT-AP-002) erroneously pass reward=1.0
            idx_num = int(handle.container_id.split("-")[-1])
            passed = idx_num in {1, 2}
            return ExecResult(
                exit_code=0 if passed else 1,
                stdout="reward=1.0\n" if passed else "reward=0.0\n",
                stderr="",
                duration_ms=2.0,
            )
        return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

    async def dead_one_off_eval(_spec: EnvSpec, mut: OneOffMutation) -> float:
        # First inverted assertion is dead (Oracle still passes with reward=1.0)
        return 1.0 if mut.mutation_id == "MUT-ONEOFF-001" else 0.0

    weak_provider = FakeSandboxProvider(exec_handler=weak_exec)
    runner = MutationSuiteRunner(provider=weak_provider, one_off_evaluator=dead_one_off_eval)
    out_dir = tmp_path / "weak_mut"
    report = await runner.run(spec, out_dir=out_dir)

    assert report.passed is False
    assert report.mutation_survival_rate == pytest.approx(2 / 5, rel=1e-4)
    assert report.dead_assertions_count == 1
    assert any(
        f.rule_id == "RULE-MUT-002" and f.severity == Severity.CRITICAL for f in report.findings
    )
    assert any(f.rule_id == "RULE-MUT-001" and f.severity == Severity.HIGH for f in report.findings)
    assert (out_dir / "mutations.json").is_file()
    assert (out_dir / "anti_patches" / "01_MUT-AP-001.diff").is_file()
    assert weak_provider.active_handles == {}

    # 2. Strong verifier: rejects all anti-patches & fails Oracle on every inverted assertion
    def strict_exec(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (handle, timeout_sec, workdir)
        joined = " ".join(cmd)
        if "test.sh" in joined:
            return ExecResult(exit_code=1, stdout="reward=0.0\n", stderr="fail", duration_ms=2.0)
        return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

    async def strict_one_off_eval(_spec: EnvSpec, _mut: OneOffMutation) -> float:
        return 0.0

    strict_provider = FakeSandboxProvider(exec_handler=strict_exec)
    strict_runner = MutationSuiteRunner(
        provider=strict_provider, one_off_evaluator=strict_one_off_eval
    )
    clean_rep = await strict_runner.run(spec)
    assert clean_rep.passed is True
    assert clean_rep.mutation_survival_rate == 0.0
    assert clean_rep.dead_assertions_count == 0
    assert clean_rep.findings == []
    assert strict_provider.active_handles == {}
