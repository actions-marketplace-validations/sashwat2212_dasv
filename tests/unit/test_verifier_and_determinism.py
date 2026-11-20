"""Unit tests for ArtifactExtractor, CleanImageGrader, adapter hooks, and DeterminismRunner."""

from __future__ import annotations

import ast
import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest

from dasv.adapters.base import AdapterCapabilities
from dasv.adapters.harbor import HarborAdapter
from dasv.adapters.openenv import OpenEnvAdapter
from dasv.core.errors import RuntimeSandboxError
from dasv.core.schemas import EnvSpec, EnvStandard, Severity
from dasv.sandbox.provider import ExecResult, SandboxHandle
from dasv.verifier import (
    VERIFIER_MOUNT_PATH,
    ArtifactExtractor,
    CleanImageGrader,
    DeterminismRunner,
    ExtractedArtifact,
    GradingContext,
    compute_artifact_sha256,
    filter_git_patch,
    is_protected_verifier_path,
    parse_verifier_reward,
    resolve_verifier_command,
    resolve_verifier_host_dir,
)
from tests.conftest import FakeSandboxProvider


def test_protected_verifier_path_and_reward_parsing() -> None:
    vps = ["tests/", "tests/test.sh", "grader/score.py"]
    assert is_protected_verifier_path("tests/test.sh", vps) is True
    assert is_protected_verifier_path("./tests/unit/test_x.py", vps) is True
    assert is_protected_verifier_path("tests\\sub\\test.py", vps) is True
    assert is_protected_verifier_path("grader/score.py", vps) is True
    assert is_protected_verifier_path("../tests/test.sh", vps) is True
    assert is_protected_verifier_path("src/../../etc/passwd", vps) is True
    assert is_protected_verifier_path("/etc/passwd", vps) is True
    assert is_protected_verifier_path("src/app.py\x00.txt", vps) is True

    assert is_protected_verifier_path("src/calculator.py", vps) is False
    assert is_protected_verifier_path("README.md", vps) is False

    assert parse_verifier_reward("all good\n", 0, False) == 1.0
    assert parse_verifier_reward("failed\n", 1, False) == 0.0
    assert parse_verifier_reward('{"reward": 0.75}\n', 1, False) == 0.75
    assert parse_verifier_reward("DASV_REWARD=0.5\n", 0, False) == 0.5
    assert parse_verifier_reward('{"reward": 1.0}\n', 0, True) == 0.0


async def test_artifact_extractor_git_patch_and_file_globs(tmp_path: Path) -> None:
    spec = EnvSpec(
        task_id="extract-demo",
        standard=EnvStandard.HARBOR,
        root_path=tmp_path,
        instruction_text="Fix calculator.",
        image_ref="python:3.12-slim",
        verifier_paths=["tests/", "tests/test.sh"],
    )

    sample_patch = (
        "diff --git a/src/calc.py b/src/calc.py\n"
        "--- a/src/calc.py\n"
        "+++ b/src/calc.py\n"
        "@@ -1 +1 @@\n"
        "-return 0\n"
        "+return a + b\n"
    )
    b64_src = base64.b64encode(b"def add(a, b): return a + b\n").decode("ascii")
    b64_test = base64.b64encode(b"assert True\n").decode("ascii")

    def fake_exec(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (handle, timeout_sec, workdir)
        if cmd[:2] == ["git", "diff"]:
            return ExecResult(exit_code=0, stdout=sample_patch, stderr="", duration_ms=3.0)
        return ExecResult(
            exit_code=0,
            stdout=json.dumps({"src/calc.py": b64_src, "tests/test.sh": b64_test}),
            stderr="",
            duration_ms=3.0,
        )

    provider = FakeSandboxProvider(exec_handler=fake_exec)
    handle = SandboxHandle(
        container_id="exec-1",
        image_digest="python:3.12-slim",
        resolved_arch="arm64",
        emulated_arch=False,
        plane="execution",
        run_id="r-extract",
    )

    extractor = ArtifactExtractor()
    git_art = await extractor.extract(provider, handle, spec, mode="git_patch")
    assert git_art.mode == "git_patch"
    assert git_art.patch_bytes == sample_patch.encode("utf-8")
    assert git_art.rejected_paths == []
    assert git_art.sha256 == compute_artifact_sha256("git_patch", git_art.patch_bytes, {})

    glob_art = await extractor.extract(
        provider, handle, spec, mode="file_globs", globs=("**/*.py", "**/*.sh")
    )
    assert glob_art.mode == "file_globs"
    assert list(glob_art.files.keys()) == ["src/calc.py"]
    assert glob_art.rejected_paths == ["tests/test.sh"]


async def test_clean_image_grader_mounts_ro_and_cleans_up_on_error(
    hardened_harbor_env: Path,
) -> None:
    adapter = HarborAdapter()
    spec = adapter.load(hardened_harbor_env)

    def verifier_exec(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (handle, timeout_sec, workdir)
        if cmd[0] == "sh" and cmd[1] == f"{VERIFIER_MOUNT_PATH}/test.sh":
            return ExecResult(
                exit_code=0,
                stdout="2 passed\n",
                stderr="",
                duration_ms=12.5,
                peak_memory_mb=48.0,
            )
        return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

    provider = FakeSandboxProvider(exec_handler=verifier_exec)
    grader = CleanImageGrader(provider)
    patch = b"diff --git a/src/calc.py b/src/calc.py\n--- a/src/calc.py\n+++ b/src/calc.py\n"
    artifact = ExtractedArtifact(
        mode="git_patch",
        patch_bytes=patch,
        files={},
        sha256=compute_artifact_sha256("git_patch", patch, {}),
    )

    outcome = await grader.grade(spec, artifact, run_id="run-clean-1")
    assert outcome.passed is True
    assert outcome.reward == 1.0
    assert outcome.peak_memory_mb == 48.0
    assert len(provider.started_configs) == 1
    started_cfg, started_run = provider.started_configs[0]
    assert started_run == "run-clean-1"
    assert started_cfg.plane == "verification"
    assert started_cfg.network_mode == "none"
    assert any(
        cpath == VERIFIER_MOUNT_PATH and mode == "ro"
        for _hpath, cpath, mode in started_cfg.bind_mounts
    )
    assert provider.active_handles == {}

    # Even if provider.exec raises mid-grade, Verification Plane container MUST be removed.
    def boom_exec(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (handle, cmd, timeout_sec, workdir)
        raise RuntimeError("transient verifier crash")

    failing_provider = FakeSandboxProvider(exec_handler=boom_exec)
    failing_grader = CleanImageGrader(failing_provider)
    with pytest.raises(RuntimeError, match="transient verifier crash"):
        await failing_grader.grade(spec, artifact, run_id="run-crash")
    assert failing_provider.active_handles == {}
    assert len(failing_provider.removed_ids) == 1


async def test_harbor_and_openenv_adapter_hooks_via_clean_grader(
    hardened_harbor_env: Path,
    minimal_openenv: Path,
) -> None:
    harbor_adapter = HarborAdapter()
    harbor_spec = harbor_adapter.load(hardened_harbor_env)

    state: dict[str, bool] = {"oracle_ran": False}

    def exec_handler(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (timeout_sec, workdir)
        joined = " ".join(cmd)
        if handle.plane == "execution":
            if "dasv_ep_" in joined:
                state["oracle_ran"] = True
                return ExecResult(exit_code=0, stdout="solved", stderr="", duration_ms=4.0)
            if cmd[:2] == ["git", "diff"]:
                if state["oracle_ran"]:
                    state["oracle_ran"] = False
                    diff = (
                        "diff --git a/src/calculator.py b/src/calculator.py\n"
                        "--- a/src/calculator.py\n"
                        "+++ b/src/calculator.py\n"
                        "@@ -1 +1 @@\n-pass\n+return a + b\n"
                    )
                    return ExecResult(exit_code=0, stdout=diff, stderr="", duration_ms=2.0)
                return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)
            return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

        # Verification plane: pass only if git apply ran for this container
        if "git', 'apply'" in joined:
            state[handle.container_id] = True
            return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=2.0)
        if cmd[0] in {"sh", "pytest"}:
            ok = state.get(handle.container_id, False)
            return ExecResult(
                exit_code=0 if ok else 1,
                stdout="passed" if ok else "failed",
                stderr="",
                duration_ms=10.0,
                peak_memory_mb=32.0,
            )
        return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

    provider = FakeSandboxProvider(exec_handler=exec_handler)
    ctx = GradingContext(CleanImageGrader(provider), run_id="run-hooks")

    oracle_reward, oracle_details = await harbor_adapter.run_oracle(harbor_spec, ctx)
    assert oracle_reward == 1.0
    assert oracle_details["passed"] is True

    noop_reward, noop_details = await harbor_adapter.run_noop(harbor_spec, ctx)
    assert noop_reward == 0.0
    assert noop_details["passed"] is False
    assert provider.active_handles == {}

    # OpenEnv fixture without oracle entrypoint should raise on run_oracle, succeed on run_noop
    openenv_adapter = OpenEnvAdapter()
    openenv_spec = openenv_adapter.load(minimal_openenv)
    with pytest.raises(RuntimeSandboxError, match="oracle_entrypoint"):
        await openenv_adapter.run_oracle(openenv_spec, ctx)
    oe_noop_reward, _ = await openenv_adapter.run_noop(openenv_spec, ctx)
    assert oe_noop_reward == 0.0


class _ScriptedDeterminismAdapter:
    standard = EnvStandard.HARBOR
    capabilities = AdapterCapabilities()

    def __init__(
        self,
        oracle_rewards: list[float],
        noop_rewards: list[float],
        *,
        emulated: bool = False,
        delay_sec: float = 0.0,
    ) -> None:
        self._oracle_rewards = list(oracle_rewards)
        self._noop_rewards = list(noop_rewards)
        self._emulated = emulated
        self._delay_sec = delay_sec
        self._o_idx = 0
        self._n_idx = 0
        self.active = 0
        self.max_active = 0

    def detect(self, path: Path) -> bool:
        _ = path
        return True

    def load(self, path: Path) -> EnvSpec:
        raise NotImplementedError

    def static_findings(self, spec: EnvSpec) -> list[Any]:
        _ = spec
        return []

    async def run_native_check(
        self, spec: EnvSpec, timeout_sec: float = 60.0, require_native: bool = False
    ) -> list[Any]:
        _ = (spec, timeout_sec, require_native)
        return []

    async def run_oracle(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        _ = (spec, sandbox)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay_sec > 0:
                await asyncio.sleep(self._delay_sec)
            idx = self._o_idx
            self._o_idx += 1
            reward = self._oracle_rewards[idx % len(self._oracle_rewards)]
            return reward, {
                "duration_ms": 10.0 + idx * 5.0,
                "peak_memory_mb": 40.0 + idx,
                "emulated_arch": self._emulated,
                "tamper_attempts": [],
            }
        finally:
            self.active -= 1

    async def run_noop(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        _ = (spec, sandbox)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay_sec > 0:
                await asyncio.sleep(self._delay_sec)
            idx = self._n_idx
            self._n_idx += 1
            reward = self._noop_rewards[idx % len(self._noop_rewards)]
            return reward, {
                "duration_ms": 8.0 + idx,
                "peak_memory_mb": 20.0,
                "emulated_arch": self._emulated,
                "tamper_attempts": [],
            }
        finally:
            self.active -= 1


async def test_determinism_runner_clean_flaky_and_leaky_noop(tmp_path: Path) -> None:
    spec = EnvSpec(
        task_id="detm-task",
        standard=EnvStandard.HARBOR,
        root_path=tmp_path,
        instruction_text="Determinism check",
        has_oracle=True,
        oracle_entrypoint="solution/solve.sh",
        verifier_paths=["tests/test.sh"],
    )

    # 1. Clean 10/10 Oracle, 3/3 No-op -> PASS, bounded concurrency <= 3
    clean_adapter = _ScriptedDeterminismAdapter([1.0] * 10, [0.0] * 3, delay_sec=0.01)
    runner = DeterminismRunner(max_concurrent=3)
    res_clean = await runner.run(clean_adapter, spec, None, oracle_trials=10, noop_trials=3)
    assert res_clean.passed is True
    assert res_clean.findings == []
    assert clean_adapter.max_active <= 3
    assert res_clean.metrics.oracle_trials == 10
    assert res_clean.metrics.oracle_pass_rate == 1.0
    assert res_clean.metrics.noop_trials == 3
    assert res_clean.metrics.noop_pass_rate == 0.0
    assert res_clean.metrics.replay_variance == 0.0
    assert res_clean.metrics.pass_at_1 == 1.0
    assert res_clean.metrics.pass_pow_k == {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0}
    assert res_clean.metrics.peak_memory_mb == 49.0

    # 2. Flaky Oracle (9/10) -> pass^10 drops to 0.348678, variance = 0.09
    flaky_adapter = _ScriptedDeterminismAdapter([1.0] * 9 + [0.0], [0.0] * 3, emulated=True)
    res_flaky = await runner.run(flaky_adapter, spec, None, oracle_trials=10, noop_trials=3)
    assert res_flaky.passed is False
    assert res_flaky.metrics.oracle_pass_rate == pytest.approx(0.9)
    assert res_flaky.metrics.replay_variance == pytest.approx(0.09)
    assert res_flaky.metrics.pass_pow_k[10] == pytest.approx(0.9**10, rel=1e-4)
    assert res_flaky.metrics.emulated_arch_warning is True
    rule_ids = {f.rule_id for f in res_flaky.findings}
    assert {"RULE-DETM-001", "RULE-DETM-003"}.issubset(rule_ids)

    # 3. Leaky No-op (1/3 scores 1.0) -> CRITICAL RULE-DETM-002
    leaky_adapter = _ScriptedDeterminismAdapter([1.0] * 4, [0.0, 1.0, 0.0])
    res_leaky = await runner.run(leaky_adapter, spec, None, oracle_trials=4, noop_trials=3)
    assert res_leaky.passed is False
    assert res_leaky.metrics.noop_pass_rate == pytest.approx(1 / 3, rel=1e-4)
    assert any(
        f.rule_id == "RULE-DETM-002" and f.severity == Severity.CRITICAL for f in res_leaky.findings
    )

    # 4. Missing oracle emits RULE-DETM-004 INFO and still runs No-op
    no_oracle_spec = spec.model_copy(update={"has_oracle": False, "oracle_entrypoint": None})
    res_no_oracle = await runner.run(
        clean_adapter, no_oracle_spec, None, oracle_trials=5, noop_trials=2
    )
    assert res_no_oracle.metrics.oracle_trials == 0
    assert res_no_oracle.metrics.noop_trials == 2
    assert any(
        f.rule_id == "RULE-DETM-004" and f.severity == Severity.INFO for f in res_no_oracle.findings
    )


def test_filter_git_patch_handles_quoted_paths_and_renames() -> None:
    patch = (
        b'diff --git "a/tests/my test.py" "b/tests/my test.py"\n'
        b'--- "a/tests/my test.py"\n'
        b'+++ "b/tests/my test.py"\n'
        b"@@ -1 +1 @@\n-assert False\n+assert True\n"
        b"diff --git a/src/ok.py b/src/ok.py\n"
        b"--- a/src/ok.py\n"
        b"+++ b/src/ok.py\n"
        b"@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    )
    sanitized, rejected = filter_git_patch(patch, ["tests/"])
    assert b"src/ok.py" in sanitized
    assert b"my test.py" not in sanitized
    assert rejected == ["tests/my test.py"]


def test_resolve_verifier_command_prefers_existing_explicit_test_path(tmp_path: Path) -> None:
    """Regression: skip missing tests/test.sh and honor existing [verifier].test_path."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text("assert True\n", encoding="utf-8")

    spec = EnvSpec(
        task_id="py-verifier-task",
        standard=EnvStandard.HARBOR,
        root_path=tmp_path,
        instruction_text="Implement calculator.",
        verifier_paths=["tests/", "tests/test.sh", "tests/test_calc.py"],
        metadata={"harbor_config": {"verifier": {"test_path": "tests/test_calc.py"}}},
    )
    host_dir = resolve_verifier_host_dir(spec)
    assert host_dir == tests_dir
    assert resolve_verifier_command(spec, host_dir) == [
        "pytest",
        f"{VERIFIER_MOUNT_PATH}/test_calc.py",
    ]

    # Even without explicit [verifier].test_path, missing tests/test.sh must not be selected.
    implicit_spec = spec.model_copy(update={"metadata": {}})
    assert resolve_verifier_command(implicit_spec, host_dir) == [
        "pytest",
        f"{VERIFIER_MOUNT_PATH}/test_calc.py",
    ]


async def test_clean_image_grader_and_extractor_inline_scripts_are_valid_python(
    hardened_harbor_env: Path,
) -> None:
    """Regression: all inline `python3 -c` scripts in Extractor & Grader must parse cleanly."""
    spec = HarborAdapter().load(hardened_harbor_env)
    captured_scripts: list[str] = []

    def exec_capture(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (handle, timeout_sec, workdir)
        if len(cmd) >= 3 and cmd[0] == "python3" and cmd[1] == "-c":
            captured_scripts.append(cmd[2])
            return ExecResult(exit_code=0, stdout="{}\n", stderr="", duration_ms=1.0)
        return ExecResult(exit_code=0, stdout="ok\n", stderr="", duration_ms=1.0)

    provider = FakeSandboxProvider(exec_handler=exec_capture)
    grader = CleanImageGrader(provider)
    files = {"src/calc.py": b"def add(a, b):\n    return a + b\n"}
    glob_artifact = ExtractedArtifact(
        mode="file_globs",
        files=files,
        sha256=compute_artifact_sha256("file_globs", None, files),
    )
    patch = b"diff --git a/src/calc.py b/src/calc.py\n--- a/src/calc.py\n+++ b/src/calc.py\n"
    patch_artifact = ExtractedArtifact(
        mode="git_patch",
        patch_bytes=patch,
        files={},
        sha256=compute_artifact_sha256("git_patch", patch, {}),
    )

    out_globs = await grader.grade(spec, glob_artifact, run_id="run-globs-syntax")
    out_patch = await grader.grade(spec, patch_artifact, run_id="run-patch-syntax")
    assert out_globs.passed is True
    assert out_patch.passed is True

    fake_handle = SandboxHandle(
        container_id="exec-syntax",
        image_digest="python:3.12-slim",
        resolved_arch="arm64",
        emulated_arch=False,
        plane="execution",
        run_id="run-extract-syntax",
    )
    await ArtifactExtractor().extract(provider, fake_handle, spec, mode="file_globs")

    assert len(captured_scripts) == 3
    for script in captured_scripts:
        ast.parse(script)
