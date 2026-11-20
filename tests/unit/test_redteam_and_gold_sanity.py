"""Unit tests for HackerAgent, FixerAgent, GoldSanityGate, RedTeamRunner & Tier 2 hook."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from dasv.adapters.harbor import HarborAdapter
from dasv.core.budget import BudgetGuard
from dasv.core.config import DasvConfig
from dasv.core.errors import EXIT_BUDGET_EXCEEDED, BudgetExceededError
from dasv.core.orchestrator import AuditContext, TieredAuditOrchestrator
from dasv.core.schemas import ExploitTrace
from dasv.redteam import (
    FixerAgent,
    HackerAgent,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolSpec,
    LLMUsage,
    RedTeamRunner,
    build_tier2_redteam_hook,
)
from dasv.sandbox.provider import ExecResult, SandboxHandle
from dasv.verifier import (
    ExtractedArtifact,
    GoldSanityGate,
    compute_artifact_sha256,
)
from tests.conftest import FakeSandboxProvider


class _ScriptedLLM:
    """Deterministic scripted LLMClient yielding canned responses or raising BudgetExceededError."""

    def __init__(self, responses: Sequence[LLMResponse | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        tools: Sequence[LLMToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        _ = (model, tools, max_tokens, temperature)
        self.calls.append(list(messages))
        if not self._responses:
            return LLMResponse(model=model, content="done", usage=LLMUsage())
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _valid_verifier_fix_diff() -> str:
    return (
        "diff --git a/tests/test.sh b/tests/test.sh\n"
        "--- a/tests/test.sh\n"
        "+++ b/tests/test.sh\n"
        "@@ -1,2 +1,2 @@\n"
        " #!/bin/sh\n"
        "-pytest\n"
        "+pytest --strict-markers -q\n"
    )


async def test_hacker_agent_react_loop_confirms_exploit_and_reaps_container(
    hardened_harbor_env: Path,
) -> None:
    spec = HarborAdapter().load(hardened_harbor_env)
    wrote_exploit = False

    def exec_handler(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        nonlocal wrote_exploit
        _ = (timeout_sec, workdir)
        joined = " ".join(cmd)
        if handle.plane == "execution":
            if "python3" in joined and "write_bytes" in joined:
                wrote_exploit = True
                return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=2.0)
            if cmd[:2] == ["git", "diff"] and wrote_exploit:
                patch = (
                    "diff --git a/src/calculator.py b/src/calculator.py\n"
                    "--- a/src/calculator.py\n"
                    "+++ b/src/calculator.py\n"
                    "@@ -1 +1 @@\n-pass\n+return 5\n"
                )
                return ExecResult(exit_code=0, stdout=patch, stderr="", duration_ms=2.0)
            return ExecResult(exit_code=0, stdout="workspace-files\n", stderr="", duration_ms=1.0)

        # Verification plane: weak test passes hardcoded stub with reward=1.0
        if "test.sh" in joined:
            return ExecResult(exit_code=0, stdout="reward=1.0\n", stderr="", duration_ms=3.0)
        return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

    llm = _ScriptedLLM(
        [
            LLMResponse(
                model="test-hacker",
                tool_calls=[
                    LLMToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "src/calculator.py", "content": "return 5\n"},
                    )
                ],
                usage=LLMUsage(prompt_tokens=100, completion_tokens=20),
            ),
            LLMResponse(
                model="test-hacker",
                tool_calls=[
                    LLMToolCall(
                        id="c2",
                        name="submit",
                        arguments={"rationale": "hardcoded expected answer 5"},
                    )
                ],
                usage=LLMUsage(prompt_tokens=140, completion_tokens=15),
            ),
        ]
    )
    provider = FakeSandboxProvider(exec_handler=exec_handler)
    agent = HackerAgent(llm, provider)
    attempt = await agent.attack(
        spec,
        hack_class="hardcoding_lookup",
        model="test-hacker",
        run_id="rt-1",
        max_turns=5,
    )

    assert attempt.confirmed_exploit is True
    assert attempt.turns_used == 2
    assert attempt.clean_reward == 1.0
    assert attempt.exploit_trace is not None
    assert attempt.exploit_trace.hack_class == "hardcoding_lookup"
    assert len(attempt.exploit_trace.steps) == 3  # 2 tool turns + clean_image_grade event
    assert provider.active_handles == {}


async def test_fixer_agent_extracts_safe_diff_and_rejects_traversal(
    hardened_harbor_env: Path,
) -> None:
    spec = HarborAdapter().load(hardened_harbor_env)
    trace = ExploitTrace(
        exploit_id="exp-hard-1",
        task_id=spec.task_id,
        hack_class="hardcoding_lookup",
        reward_obtained=1.0,
        steps=[{"turn": 1, "action": {"tool": "write_file"}}],
    )

    # 1. Valid diff inside fenced markdown block -> accepted
    good_llm = _ScriptedLLM(
        [
            LLMResponse(
                model="test-fixer",
                content=f"Hardening patch:\n```diff\n{_valid_verifier_fix_diff()}```\n",
            )
        ]
    )
    good_diff = await FixerAgent(good_llm).propose_fix(spec, trace, model="test-fixer")
    assert good_diff is not None
    assert "pytest --strict-markers -q" in good_diff

    # 2. Traversal diff targeting ../../../etc/passwd -> rejected (returns None)
    evil_diff = (
        "diff --git a/../../../etc/passwd b/../../../etc/passwd\n"
        "--- a/../../../etc/passwd\n"
        "+++ b/../../../etc/passwd\n"
        "@@ -1 +1 @@\n-root\n+pwned\n"
    )
    bad_llm = _ScriptedLLM([LLMResponse(model="test-fixer", content=f"```diff\n{evil_diff}```")])
    rejected = await FixerAgent(bad_llm).propose_fix(spec, trace, model="test-fixer")
    assert rejected is None


async def test_gold_sanity_gate_all_four_outcomes(hardened_harbor_env: Path) -> None:
    adapter = HarborAdapter()
    spec = adapter.load(hardened_harbor_env)
    patch_bytes = b"diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n"
    exploit_artifact = ExtractedArtifact(
        mode="git_patch",
        patch_bytes=patch_bytes,
        sha256=compute_artifact_sha256("git_patch", patch_bytes, {}),
    )
    diff_text = _valid_verifier_fix_diff()

    def _make_gsg_provider(
        *, exploit_reward: float, oracle_reward: float, noop_reward: float
    ) -> FakeSandboxProvider:
        applied_patch_for: dict[str, bool] = {}
        exec_oracle_for: dict[str, bool] = {}

        def handler(
            handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
        ) -> ExecResult:
            _ = (timeout_sec, workdir)
            joined = " ".join(cmd)
            if handle.plane == "execution":
                if "dasv_ep_" in joined:
                    exec_oracle_for[handle.container_id] = True
                if cmd[:2] == ["git", "diff"] and exec_oracle_for.get(handle.container_id):
                    diff_ok = (
                        "diff --git a/src/ok.py b/src/ok.py\n"
                        "--- a/src/ok.py\n+++ b/src/ok.py\n@@ -1 +1 @@\n-0\n+1\n"
                    )
                    return ExecResult(
                        exit_code=0,
                        stdout=diff_ok,
                        stderr="",
                        duration_ms=1.0,
                    )
                return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

            if "git', 'apply'" in joined:
                applied_patch_for[handle.container_id] = True
                return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)
            if "test.sh" in joined:
                if handle.run_id.endswith("-exploit"):
                    r = exploit_reward
                elif applied_patch_for.get(handle.container_id):
                    r = oracle_reward
                else:
                    r = noop_reward
                return ExecResult(
                    exit_code=0 if r > 0 else 1,
                    stdout=f"reward={r:.1f}\n",
                    stderr="",
                    duration_ms=2.0,
                )
            return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

        return FakeSandboxProvider(exec_handler=handler)

    # 1. All 3 invariants hold -> verified_fix
    ok_res = await GoldSanityGate(
        _make_gsg_provider(exploit_reward=0.0, oracle_reward=1.0, noop_reward=0.0)
    ).verify_patch(spec, adapter, exploit_artifact, diff_text, oracle_replays=2, noop_replays=1)
    assert ok_res.status == "verified_fix"
    assert ok_res.exploit_blocked is True
    assert ok_res.oracle_preserved is True
    assert ok_res.noop_preserved is True

    # 2. Exploit still passes -> rejected_fix
    bad_exp = await GoldSanityGate(
        _make_gsg_provider(exploit_reward=1.0, oracle_reward=1.0, noop_reward=0.0)
    ).verify_patch(spec, adapter, exploit_artifact, diff_text, oracle_replays=2, noop_replays=1)
    assert bad_exp.status == "rejected_fix"
    assert bad_exp.exploit_blocked is False

    # 3. Oracle regressed -> rejected_fix
    bad_oracle = await GoldSanityGate(
        _make_gsg_provider(exploit_reward=0.0, oracle_reward=0.0, noop_reward=0.0)
    ).verify_patch(spec, adapter, exploit_artifact, diff_text, oracle_replays=2, noop_replays=1)
    assert bad_oracle.status == "rejected_fix"
    assert bad_oracle.oracle_preserved is False

    # 4. No-op leaked -> rejected_fix
    bad_noop = await GoldSanityGate(
        _make_gsg_provider(exploit_reward=0.0, oracle_reward=1.0, noop_reward=1.0)
    ).verify_patch(spec, adapter, exploit_artifact, diff_text, oracle_replays=2, noop_replays=1)
    assert bad_noop.status == "rejected_fix"
    assert bad_noop.noop_preserved is False


async def test_redteam_runner_and_tier2_orchestrator_html_drawer_and_budget_flush(
    hardened_harbor_env: Path,
    tmp_path: Path,
) -> None:
    adapter = HarborAdapter()
    spec = adapter.load(hardened_harbor_env)
    state: dict[str, bool] = {}

    def exec_handler(
        handle: SandboxHandle, cmd: list[str], timeout_sec: float, workdir: str | None
    ) -> ExecResult:
        _ = (timeout_sec, workdir)
        joined = " ".join(cmd)
        if "os.access" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"exists": [], "writable": []}),
                stderr="",
                duration_ms=1.0,
            )
        if "egress_reachable" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"egress_reachable": [], "leaked_env_keys": []}),
                stderr="",
                duration_ms=1.0,
            )
        if "surviving" in joined:
            return ExecResult(
                exit_code=0,
                stdout=json.dumps({"surviving": []}),
                stderr="",
                duration_ms=1.0,
            )
        if handle.plane == "execution":
            if "dasv_ep_" in joined or "write_bytes" in joined:
                state[handle.container_id] = True
            if cmd[:2] == ["git", "diff"] and state.get(handle.container_id):
                diff_a = (
                    "diff --git a/src/a.py b/src/a.py\n"
                    "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-0\n+5\n"
                )
                return ExecResult(
                    exit_code=0,
                    stdout=diff_a,
                    stderr="",
                    duration_ms=1.0,
                )
            return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

        if "git', 'apply'" in joined:
            state[handle.container_id] = True
            return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)
        if "test.sh" in joined:
            if handle.run_id.endswith("-exploit"):
                # After Fixer patch, exploit is blocked!
                return ExecResult(exit_code=1, stdout="reward=0.0\n", stderr="", duration_ms=1.0)
            ok = state.get(handle.container_id, False)
            return ExecResult(
                exit_code=0 if ok else 1,
                stdout="reward=1.0\n" if ok else "reward=0.0\n",
                stderr="",
                duration_ms=1.0,
            )
        return ExecResult(exit_code=0, stdout="", stderr="", duration_ms=1.0)

    # First hack class finds exploit + Fixer fixes it; second hack class trips BudgetExceededError
    scripted = _ScriptedLLM(
        [
            LLMResponse(
                model="local-hacker",
                tool_calls=[
                    LLMToolCall(
                        id="t1",
                        name="write_file",
                        arguments={"path": "src/a.py", "content": "5\n"},
                    )
                ],
            ),
            LLMResponse(
                model="local-hacker",
                tool_calls=[
                    LLMToolCall(id="t2", name="submit", arguments={"rationale": "hardcoded"})
                ],
            ),
            LLMResponse(
                model="local-fixer",
                content=f"```diff\n{_valid_verifier_fix_diff()}```",
            ),
            BudgetExceededError("USD budget exceeded on hack class 2"),
        ]
    )

    provider = FakeSandboxProvider(exec_handler=exec_handler)
    out_dir = tmp_path / "t2_orch"
    html_path = out_dir / "report.html"
    ctx = AuditContext.create(
        config=DasvConfig(),
        adapter=adapter,
        env_spec=spec,
        out_dir=out_dir,
        provider=provider,
        native=False,
        oracle_replays=1,
        noop_replays=1,
    )
    hook = build_tier2_redteam_hook(
        llm=scripted,
        attacker_model="local-hacker",
        fixer_model="local-fixer",
        hack_classes=("hardcoding_lookup", "state_bypass"),
        propose_fix=True,
        gold_sanity_replays=1,
    )
    res = await TieredAuditOrchestrator(tier2_hook=hook).execute(
        ctx,
        {0, 1, 2},
        html_path=html_path,
    )

    # Budget exceeded -> exit 3, yet partial verified exploit + Fixer diff + HTML drawer persisted!
    assert res.exit_code == EXIT_BUDGET_EXCEEDED
    assert res.budget_exceeded is True
    assert len(res.report.exploits) == 1
    assert res.report.exploits[0].gold_sanity_status == "verified_fix"
    assert (out_dir / "redteam" / "redteam.json").is_file()
    assert (out_dir / "redteam" / "exploits.jsonl").is_file()
    fixes = list((out_dir / "redteam" / "fixes").glob("*.diff"))
    assert len(fixes) == 1
    html_text = html_path.read_text(encoding="utf-8")
    assert "Adversarial Exploit Traces" in html_text
    assert "verified_fix" in html_text
    assert provider.active_handles == {}

    # Also verify standalone RedTeamRunner clean pass when no exploits succeed
    no_exploit_llm = _ScriptedLLM([LLMResponse(model="local", content="cannot exploit")])
    strict_provider = FakeSandboxProvider(
        exec_handler=lambda h, cmd, t, w: ExecResult(
            exit_code=1 if "test.sh" in " ".join(cmd) else 0,
            stdout="reward=0.0\n" if "test.sh" in " ".join(cmd) else "",
            stderr="",
            duration_ms=1.0,
        )
    )
    rt = await RedTeamRunner(
        llm=no_exploit_llm,
        provider=strict_provider,
        budget=BudgetGuard(
            max_cost_usd=1.0, max_tokens=10_000, base_url="http://localhost:11434/v1"
        ),
    ).run(
        spec,
        adapter,
        attacker_model="local",
        hack_classes=("state_bypass",),
    )
    assert rt.passed is True
    assert rt.exploits == []
