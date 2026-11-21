"""Unit tests for Harbor/OpenEnv/verifiers v1/HUD adapters, entry_points & NativeDelegateRunner."""

from __future__ import annotations

import os
import sys
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

from dasv.adapters.base import AdapterCapabilities
from dasv.adapters.native import NativeDelegateRunner
from dasv.adapters.registry import AdapterRegistry
from dasv.core.errors import (
    EXIT_CONFIG_ERROR,
    EXIT_NATIVE_ERROR,
    ConfigError,
    NativeDelegateError,
)
from dasv.core.schemas import AuditFinding, EnvSpec, EnvStandard, Severity


def test_auto_detect_harbor_openenv_verifiers_v1_and_hud(
    hardened_harbor_env: Path,
    minimal_openenv: Path,
    minimal_verifiers_v1: Path,
    tmp_path: Path,
) -> None:
    registry = AdapterRegistry(load_entry_points=False)

    harbor_adapter = registry.resolve(hardened_harbor_env)
    harbor_spec = harbor_adapter.load(hardened_harbor_env)
    assert harbor_spec.standard == EnvStandard.HARBOR
    assert harbor_spec.task_id == "hardened-calculator"
    assert harbor_spec.has_oracle is True
    assert harbor_spec.oracle_entrypoint == "solution/solve.sh"
    assert harbor_spec.metadata["network_mode"] == "no-network"

    openenv_adapter = registry.resolve(minimal_openenv)
    openenv_spec = openenv_adapter.load(minimal_openenv)
    assert openenv_spec.standard == EnvStandard.OPENENV
    assert openenv_spec.task_id == "gridworld-v1"
    assert openenv_adapter.capabilities.supports_interactive_reset is True
    assert openenv_spec.image_digest is not None

    vf_adapter = registry.resolve(minimal_verifiers_v1)
    vf_spec = vf_adapter.load(minimal_verifiers_v1)
    assert vf_spec.standard == EnvStandard.VERIFIERS_V1
    assert vf_spec.rubric_type == "hybrid"
    assert vf_adapter.static_findings(vf_spec) == []

    hud_dir = tmp_path / "hud_task"
    hud_dir.mkdir()
    (hud_dir / "hud.toml").write_text('name = "hud-crm-gym"\n', encoding="utf-8")
    hud_adapter = registry.resolve(hud_dir)
    hud_spec = hud_adapter.load(hud_dir)
    assert hud_spec.standard == EnvStandard.HUD
    assert hud_spec.task_id == "hud-crm-gym"


def test_legacy_verifiers_v0_triggers_deprecation_finding(legacy_verifiers_v0: Path) -> None:
    registry = AdapterRegistry(load_entry_points=False)
    adapter = registry.resolve(legacy_verifiers_v0)
    spec = adapter.load(legacy_verifiers_v0)
    findings = adapter.static_findings(spec)
    assert len(findings) == 1
    assert findings[0].rule_id == "RULE-ADPT-001"
    assert findings[0].severity == Severity.HIGH


def test_ambiguous_env_raises_config_error_unless_forced(hardened_harbor_env: Path) -> None:
    (hardened_harbor_env / "openenv.yaml").write_text("name: clash\n", encoding="utf-8")
    registry = AdapterRegistry(load_entry_points=False)

    with pytest.raises(ConfigError) as exc_info:
        registry.resolve(hardened_harbor_env)
    assert exc_info.value.exit_code == EXIT_CONFIG_ERROR

    forced = registry.resolve(hardened_harbor_env, forced_format="harbor")
    assert forced.standard == EnvStandard.HARBOR


class _DummyPluginAdapter:
    standard: EnvStandard = EnvStandard.HUD
    capabilities: AdapterCapabilities = AdapterCapabilities()

    def detect(self, path: Path) -> bool:
        return (path / "dummy.marker").is_file()

    def load(self, path: Path) -> EnvSpec:
        return EnvSpec(
            task_id="dummy-plugin",
            standard=EnvStandard.HUD,
            root_path=path.resolve(),
            instruction_text="dummy",
        )

    def static_findings(self, spec: EnvSpec) -> list[AuditFinding]:
        return []

    async def run_native_check(
        self, spec: EnvSpec, timeout_sec: float = 60.0, require_native: bool = False
    ) -> list[AuditFinding]:
        return []

    async def run_oracle(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        return 1.0, {}

    async def run_noop(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]:
        return 0.0, {}


def test_entry_point_plugin_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeEntryPoint(EntryPoint):
        def load(self) -> Any:
            return _DummyPluginAdapter

    fake_ep = _FakeEntryPoint(name="dummy_plugin", value="fake:Dummy", group="dasv.adapters")

    def _fake_entry_points(group: str) -> list[EntryPoint]:
        return [fake_ep] if group == "dasv.adapters" else []

    monkeypatch.setattr("dasv.adapters.registry.entry_points", _fake_entry_points)

    registry = AdapterRegistry(load_entry_points=True)
    assert "dummy_plugin" in registry.list_names()

    plugin_dir = tmp_path / "custom_env"
    plugin_dir.mkdir()
    (plugin_dir / "dummy.marker").write_text("ok", encoding="utf-8")
    adapter = registry.resolve(plugin_dir)
    assert adapter.load(plugin_dir).task_id == "dummy-plugin"


async def test_native_delegate_missing_binary(tmp_path: Path) -> None:
    runner = NativeDelegateRunner()
    res = await runner.run(
        "nonexistent_dasv_validator_xyz",
        ["check"],
        cwd=tmp_path,
        require_native=False,
    )
    assert res.found is False
    assert len(res.findings) == 1
    assert res.findings[0].rule_id == "RULE-NATIVE-001"
    assert res.findings[0].severity == Severity.INFO

    with pytest.raises(NativeDelegateError) as exc_info:
        await runner.run(
            "nonexistent_dasv_validator_xyz",
            ["check"],
            cwd=tmp_path,
            require_native=True,
        )
    assert exc_info.value.exit_code == EXIT_NATIVE_ERROR


async def test_native_delegate_exit_codes_timeout_and_env_scrub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = NativeDelegateRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-must-not-leak")
    assert os.environ["OPENAI_API_KEY"] == "sk-secret-must-not-leak"

    ok_res = await runner.run(
        sys.executable,
        ["-c", "import os, sys; sys.exit(1 if 'OPENAI_API_KEY' in os.environ else 0)"],
        cwd=tmp_path,
    )
    assert ok_res.found is True
    assert ok_res.exit_code == 0
    assert ok_res.findings == []

    fail_res = await runner.run(
        sys.executable,
        ["-c", "import sys; print('schema invalid', file=sys.stderr); sys.exit(2)"],
        cwd=tmp_path,
    )
    assert fail_res.exit_code == 2
    assert len(fail_res.findings) == 1
    assert fail_res.findings[0].rule_id == "RULE-NATIVE-003"
    assert fail_res.findings[0].severity == Severity.HIGH

    timeout_res = await runner.run(
        sys.executable,
        ["-c", "import time; time.sleep(5)"],
        cwd=tmp_path,
        timeout_sec=0.1,
    )
    assert timeout_res.exit_code is None
    assert timeout_res.findings[0].rule_id == "RULE-NATIVE-002"


async def test_verifiers_v1_and_hud_container_oracle_noop_and_non_container_tier1(
    minimal_verifiers_v1: Path,
    tmp_path: Path,
) -> None:
    """Commit A1: verifiers_v1 & HUD support clean-image oracle/noop & pure-Python T1 advisory."""
    from dasv.adapters.hud import HudAdapter
    from dasv.adapters.verifiers_v1 import VerifiersV1Adapter
    from dasv.core.config import DasvConfig
    from dasv.core.errors import EXIT_PASS
    from dasv.core.orchestrator import AuditContext, TieredAuditOrchestrator
    from dasv.sandbox.provider import ExecResult, SandboxHandle
    from dasv.verifier import CleanImageGrader, GradingContext
    from tests.conftest import FakeSandboxProvider

    # 1. Pure-Python verifiers_v1 (no image / no Dockerfile) skips Tier 1 with RULE-ADPT-002 INFO
    vf_adapter = VerifiersV1Adapter()
    vf_spec = vf_adapter.load(minimal_verifiers_v1)
    ctx = AuditContext.create(
        config=DasvConfig(),
        adapter=vf_adapter,
        env_spec=vf_spec,
        native=False,
    )
    res = await TieredAuditOrchestrator().execute(ctx, {0, 1})
    assert res.exit_code == EXIT_PASS
    assert any(
        f.rule_id == "RULE-ADPT-002" and f.severity == Severity.INFO for f in res.report.findings
    )

    # 2. Containerized HUD env with image + solution/solve.sh executes clean-image oracle & noop
    hud_dir = tmp_path / "hud_container_env"
    (hud_dir / "solution").mkdir(parents=True)
    (hud_dir / "tests").mkdir()
    (hud_dir / "hud.toml").write_text(
        'name = "hud-demo"\nimage = "python:3.12-slim"\nnetwork_mode = "no-network"\n',
        encoding="utf-8",
    )
    (hud_dir / "solution" / "solve.sh").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    (hud_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    hud_adapter = HudAdapter()
    hud_spec = hud_adapter.load(hud_dir)
    assert hud_spec.has_oracle is True
    assert hud_spec.oracle_entrypoint == "solution/solve.sh"

    state: dict[str, bool] = {"ran_oracle": False}

    def fake_exec(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (timeout_sec, workdir)
        joined = " ".join(cmd)
        if handle.plane == "execution":
            if "dasv_ep_" in joined:
                state["ran_oracle"] = True
            if cmd[:2] == ["git", "diff"] and state["ran_oracle"]:
                state["ran_oracle"] = False
                return ExecResult(
                    exit_code=0,
                    stdout=(
                        "diff --git a/app.py b/app.py\n"
                        "--- a/app.py\n"
                        "+++ b/app.py\n"
                        "@@ -1 +1 @@\n-0\n+1\n"
                    ),
                    stderr="",
                    duration_ms=2.0,
                )
            return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)
        if "git', 'apply'" in joined:
            state[handle.container_id] = True
            return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)
        ok = state.get(handle.container_id, False)
        return ExecResult(exit_code=0 if ok else 1, stdout="", stderr="", duration_ms=1.0)

    provider = FakeSandboxProvider(exec_handler=fake_exec)
    gctx = GradingContext(CleanImageGrader(provider), run_id="hud-t1")
    o_reward, _ = await hud_adapter.run_oracle(hud_spec, gctx)
    n_reward, _ = await hud_adapter.run_noop(hud_spec, gctx)
    assert o_reward == 1.0
    assert n_reward == 0.0
