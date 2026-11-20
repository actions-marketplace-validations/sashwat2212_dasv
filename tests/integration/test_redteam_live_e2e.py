"""End-to-end live Docker + real localhost HTTP tests & Phase A regressions for Commit 5."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from dasv.adapters.harbor import HarborAdapter
from dasv.cli.app import app
from dasv.core.budget import BudgetGuard
from dasv.core.errors import (
    EXIT_AUDIT_FAIL,
    EXIT_BUDGET_EXCEEDED,
    EXIT_CONFIG_ERROR,
    EXIT_PASS,
    EXIT_RUNTIME_ERROR,
)
from dasv.core.schemas import AuditReport
from dasv.redteam import (
    LLMMessage,
    UnifiedLLMClient,
    extract_unified_diff,
)
from dasv.sandbox.docker_provider import DockerSandboxProvider
from dasv.sandbox.janitor import ContainerJanitor
from dasv.sandbox.provider import ContainerConfig
from dasv.verifier import (
    CleanImageGrader,
    ExtractedArtifact,
    GoldSanityGate,
    apply_unified_diff_to_dir,
    compute_artifact_sha256,
    ensure_container_image,
)

runner = CliRunner()

_GOOD_HARDENING_DIFF = (
    "diff --git a/tests/test.sh b/tests/test.sh\n"
    "--- a/tests/test.sh\n"
    "+++ b/tests/test.sh\n"
    "@@ -1,2 +1,4 @@\n"
    " #!/bin/sh\n"
    "+set -e\n"
    "-python3 -c 'import calculator; assert calculator.add(2, 3) == 5'\n"
    "+python3 -c 'import calculator; assert calculator.add(2, 3) == 5; "
    "assert calculator.add(10, 20) == 30; assert calculator.add(-1, 4) == 3'\n"
    "+echo DASV_REWARD=1.0\n"
)

_BROKEN_ORACLE_DIFF = (
    "diff --git a/tests/test.sh b/tests/test.sh\n"
    "--- a/tests/test.sh\n"
    "+++ b/tests/test.sh\n"
    "@@ -1,2 +1,2 @@\n"
    " #!/bin/sh\n"
    "-python3 -c 'import calculator; assert calculator.add(2, 3) == 5'\n"
    "+python3 -c 'import calculator; assert calculator.add(2, 3) == 999'\n"
)


def _create_live_weak_harbor_env(base_dir: Path) -> Path:
    """Create a real Harbor task using locally cached python:3.12-slim with a weak verifier."""
    env_dir = base_dir / "live_weak_calc"
    (env_dir / "environment").mkdir(parents=True)
    (env_dir / "solution").mkdir()
    (env_dir / "tests").mkdir()

    (env_dir / "environment" / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n    return 0\n",
        encoding="utf-8",
    )
    (env_dir / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nWORKDIR /app\nCOPY calculator.py /app/calculator.py\n",
        encoding="utf-8",
    )
    (env_dir / "task.toml").write_text(
        (
            "[metadata]\n"
            'name = "live-weak-calc"\n\n'
            "[environment]\n"
            'network_mode = "no-network"\n\n'
            "[verifier]\n"
            'test_path = "tests/test.sh"\n'
        ),
        encoding="utf-8",
    )
    (env_dir / "instruction.md").write_text(
        "Implement general integer addition add(a, b) in calculator.py.\n",
        encoding="utf-8",
    )
    (env_dir / "solution" / "solve.sh").write_text(
        (
            "#!/bin/sh\n"
            "cat <<'EOF' > /app/calculator.py\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "EOF\n"
        ),
        encoding="utf-8",
    )
    # Intentionally weak verifier: only asserts add(2, 3) == 5 (vulnerable to hardcoding 5!)
    (env_dir / "tests" / "test.sh").write_text(
        "#!/bin/sh\npython3 -c 'import calculator; assert calculator.add(2, 3) == 5'\n",
        encoding="utf-8",
    )
    return env_dir


@contextmanager
def _mock_openai_http_server(
    *,
    fixer_diff: str = _GOOD_HARDENING_DIFF,
    fail_first_with_429: bool = False,
) -> Generator[tuple[str, list[dict[str, Any]]], None, None]:
    """Run a real localhost ThreadingHTTPServer speaking OpenAI /v1/chat/completions."""
    recorded: list[dict[str, Any]] = []
    state = {"count": 0}

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            body = cast(dict[str, Any], json.loads(raw_body))
            recorded.append({"path": self.path, "body": body})
            state["count"] += 1

            if fail_first_with_429 and state["count"] == 1:
                err_payload = json.dumps({"error": "rate_limited"}).encode("utf-8")
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_payload)))
                self.end_headers()
                self.wfile.write(err_payload)
                return

            msgs = cast(list[dict[str, Any]], body.get("messages", []))
            model = str(body.get("model", "local-mock"))
            has_tools = "tools" in body

            if has_tools:
                tool_msgs = [m for m in msgs if m.get("role") == "tool"]
                if len(tool_msgs) == 0:
                    # Step 1: Execute shell command checking workspace + verifying --network=none
                    tc_args = {
                        "cmd": (
                            'python3 -c "import socket; '
                            "print('NET_OK' if False else 'CHECKING'); "
                            "s=socket.socket(); s.settimeout(0.5); "
                            "print('EGRESS_BLOCKED' if s.connect_ex(('1.1.1.1', 53)) != 0 "
                            "else 'EGRESS_OPEN')\""
                        )
                    }
                    tool_call = {
                        "id": "call_run_1",
                        "type": "function",
                        "function": {"name": "run_cmd", "arguments": json.dumps(tc_args)},
                    }
                elif len(tool_msgs) == 1:
                    # Step 2: Read workspace file
                    tool_call = {
                        "id": "call_read_2",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "calculator.py"}),
                        },
                    }
                elif len(tool_msgs) == 2:
                    # Step 3: Write hardcoded cheat returning 5
                    tool_call = {
                        "id": "call_write_3",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {
                                    "path": "calculator.py",
                                    "content": "def add(a: int, b: int) -> int:\n    return 5\n",
                                }
                            ),
                        },
                    }
                else:
                    # Step 4: Submit exploit
                    tool_call = {
                        "id": "call_submit_4",
                        "type": "function",
                        "function": {
                            "name": "submit",
                            "arguments": json.dumps(
                                {"rationale": "hardcoded constant 5 passes add(2,3)"}
                            ),
                        },
                    }

                resp_obj: dict[str, Any] = {
                    "id": f"chatcmpl-{state['count']}",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 60, "completion_tokens": 20},
                }
            else:
                # Fixer response: include a decoy python fence before the real diff fence
                fixer_content = (
                    "Root cause:\n```python\ndef add(a, b): return 5\n```\n"
                    f"Hardening patch:\n```diff\n{fixer_diff}```\n"
                )
                resp_obj = {
                    "id": f"chatcmpl-fix-{state['count']}",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": fixer_content},
                        }
                    ],
                    "usage": {"prompt_tokens": 110, "completion_tokens": 45},
                }

            raw_resp = json.dumps(resp_obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw_resp)))
            self.end_headers()
            self.wfile.write(raw_resp)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1", recorded
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Phase A Non-Docker Regressions (always run in CI & locally)
# ---------------------------------------------------------------------------


def test_apply_unified_diff_isolated_from_parent_monorepo_git(tmp_path: Path) -> None:
    """Regression: git apply in a subdirectory of a parent git repo must patch cwd, not parent."""
    monorepo = tmp_path / "monorepo"
    task_dir = monorepo / "tasks" / "calc"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/sh\npython3 -c 'import calculator; assert calculator.add(2, 3) == 5'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=str(monorepo), check=True, capture_output=True)

    ok, err = apply_unified_diff_to_dir(task_dir, _GOOD_HARDENING_DIFF)
    assert ok is True, err
    updated = (task_dir / "tests" / "test.sh").read_text(encoding="utf-8")
    assert "assert calculator.add(10, 20) == 30" in updated


def test_extract_unified_diff_prefers_diff_fence_over_earlier_code_fence() -> None:
    """Regression: ignore earlier ```python``` explanation blocks when extracting ```diff```."""
    llm_text = (
        "Bad code:\n```python\ndef add(a, b):\n    return 5\n```\n\n"
        f"Fix:\n```diff\n{_GOOD_HARDENING_DIFF}```\n"
    )
    extracted = extract_unified_diff(llm_text)
    assert extracted is not None
    assert extracted.startswith("diff --git a/tests/test.sh")


async def test_unified_llm_client_real_localhost_http_and_retry() -> None:
    """Verify UnifiedLLMClient over real localhost TCP/HTTP including 429 retry & usage commit."""
    with _mock_openai_http_server(fail_first_with_429=True) as (base_url, recorded):
        budget = BudgetGuard(max_cost_usd=1.0, max_tokens=5000, base_url=base_url)
        client = UnifiedLLMClient(
            base_url=base_url,
            budget=budget,
            max_retries=2,
            backoff_base_sec=0.01,
        )
        resp = await client.complete(
            [LLMMessage(role="user", content="Propose a fix")],
            model="local-mock",
            max_tokens=256,
        )
        assert len(recorded) == 2
        assert recorded[0]["path"] == "/v1/chat/completions"
        assert "diff --git a/tests/test.sh" in resp.content
        snap = await budget.snapshot()
        assert snap.committed_tokens == 155
        assert snap.reserved_tokens == 0


def test_cli_audit_cloud_base_url_enforces_budget_cap(
    hardened_harbor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: `dasv audit --tiers 2 --base-url https://...` must NOT treat cloud as local."""
    from dasv.sandbox.provider import RuntimeHealth

    async def healthy_docker(self: DockerSandboxProvider) -> RuntimeHealth:
        _ = self
        return RuntimeHealth(healthy=True, runtime="docker", host_arch="arm64", detail="ok")

    async def fake_start(self: DockerSandboxProvider, cfg: ContainerConfig, run_id: str) -> object:
        from dasv.sandbox.provider import SandboxHandle

        _ = (self, cfg)
        return SandboxHandle(
            container_id="fake-cloud-budget-1",
            image_digest="img",
            resolved_arch="arm64",
            emulated_arch=False,
            plane="execution",
            run_id=run_id,
        )

    async def fake_stop(self: DockerSandboxProvider, handle: object) -> None:
        _ = (self, handle)

    async def fake_build(
        self: DockerSandboxProvider,
        context_dir: Path,
        dockerfile: Path,
        tag: str,
        platform: str | None = None,
    ) -> str:
        _ = (self, context_dir, dockerfile, tag, platform)
        return "sha256:fake"

    monkeypatch.setattr(DockerSandboxProvider, "health_check", healthy_docker)
    monkeypatch.setattr(DockerSandboxProvider, "start", fake_start)
    monkeypatch.setattr(DockerSandboxProvider, "stop_and_remove", fake_stop)
    monkeypatch.setattr(DockerSandboxProvider, "build_image", fake_build)

    # Cloud endpoint + tiny USD cap must trip BudgetGuard BEFORE sending HTTP (EXIT_BUDGET_EXCEEDED)
    res = runner.invoke(
        app,
        [
            "audit",
            str(hardened_harbor_env),
            "--tiers",
            "2",
            "--base-url",
            "https://api.openai.com/v1",
            "--redteam-model",
            "openai/gpt-4o",
            "--max-cost-usd",
            "0.000001",
            "--hack-classes",
            "hardcoding_lookup",
        ],
    )
    assert res.exit_code == EXIT_BUDGET_EXCEEDED, res.stdout + res.stderr
    assert "RULE-BUDG-001" in res.stdout


# ---------------------------------------------------------------------------
# Phase A Live OrbStack Docker + Real Localhost HTTP E2E Tests
# ---------------------------------------------------------------------------


@pytest.mark.docker
async def test_live_docker_janitor_reaps_child_run_id_containers() -> None:
    """Verify ContainerJanitor.reap_all(parent) reaps child `<parent>-h1` containers cleanly."""
    janitor = ContainerJanitor()
    provider = DockerSandboxProvider(janitor=janitor)
    parent_run = "dasv-live-child-reap"
    await provider.start(
        ContainerConfig(image="python:3.12-slim", network_mode="none"),
        run_id=f"{parent_run}-h1",
    )
    await provider.start(
        ContainerConfig(image="python:3.12-slim", network_mode="none"),
        run_id=f"{parent_run}-gsg-1-exploit",
    )
    reap = await janitor.reap_all(parent_run)
    assert len(reap.removed_ids) == 2
    assert reap.stubborn_ids == []
    assert await janitor.list_labeled_containers(parent_run) == []


@pytest.mark.docker
async def test_live_docker_gold_sanity_gate_verified_and_rejected(tmp_path: Path) -> None:
    """Verify GoldSanityGate against real Docker containers for both verified & rejected fixes."""
    env_dir = _create_live_weak_harbor_env(tmp_path)
    adapter = HarborAdapter()
    raw_spec = adapter.load(env_dir)
    janitor = ContainerJanitor()
    provider = DockerSandboxProvider(janitor=janitor)
    run_id = "dasv-live-gsg"
    built_spec = await ensure_container_image(raw_spec, provider, run_id)

    exploit_files = {"calculator.py": b"def add(a: int, b: int) -> int:\n    return 5\n"}
    exploit_artifact = ExtractedArtifact(
        mode="file_globs",
        files=exploit_files,
        sha256=compute_artifact_sha256("file_globs", None, exploit_files),
    )

    try:
        # Confirm exploit initially scores 1.0 against the weak verifier
        grader = CleanImageGrader(provider)
        pre_outcome = await grader.grade(built_spec, exploit_artifact, f"{run_id}-pre")
        assert pre_outcome.reward == 1.0

        gsg = GoldSanityGate(provider)

        # 1. Good hardening diff -> blocks exploit (0.0), preserves Oracle (1.0) & No-op (0.0)
        good_res = await gsg.verify_patch(
            built_spec,
            adapter,
            exploit_artifact,
            _GOOD_HARDENING_DIFF,
            run_id=f"{run_id}-good",
            oracle_replays=2,
            noop_replays=1,
        )
        assert good_res.status == "verified_fix", good_res
        assert good_res.exploit_blocked is True
        assert good_res.oracle_preserved is True
        assert good_res.noop_preserved is True

        # 2. Broken diff that breaks Oracle -> rejected_fix
        bad_res = await gsg.verify_patch(
            built_spec,
            adapter,
            exploit_artifact,
            _BROKEN_ORACLE_DIFF,
            run_id=f"{run_id}-bad",
            oracle_replays=1,
            noop_replays=1,
        )
        assert bad_res.status == "rejected_fix"
        assert bad_res.oracle_preserved is False
    finally:
        reap = await janitor.reap_all(run_id)
        assert reap.stubborn_ids == []
        if built_spec.image_ref and built_spec.image_ref.startswith("dasv-"):
            subprocess.run(["docker", "rmi", "-f", built_spec.image_ref], capture_output=True)


@pytest.mark.docker
def test_live_docker_and_http_cli_redteam_and_audit_exit_codes(tmp_path: Path) -> None:
    """Full live E2E CLI check: `dasv redteam` + `dasv audit --tiers 0,1,2` + exits 0/1/2/3/4."""
    env_dir = _create_live_weak_harbor_env(tmp_path)

    try:
        with _mock_openai_http_server(fixer_diff=_GOOD_HARDENING_DIFF) as (base_url, recorded):
            # 1. `dasv redteam` WITHOUT --propose-fix -> confirms exploit, exit 1 (EXIT_AUDIT_FAIL)
            out_unpatched = tmp_path / "rt_unpatched"
            res_fail = runner.invoke(
                app,
                [
                    "redteam",
                    str(env_dir),
                    "--base-url",
                    base_url,
                    "--model",
                    "local-mock",
                    "--hack-classes",
                    "hardcoding_lookup",
                    "--out-dir",
                    str(out_unpatched),
                ],
            )
            assert res_fail.exit_code == EXIT_AUDIT_FAIL, res_fail.stdout + res_fail.stderr
            rt_data = json.loads((out_unpatched / "redteam.json").read_text(encoding="utf-8"))
            assert len(rt_data["exploits"]) == 1
            assert rt_data["exploits"][0]["reward_obtained"] == 1.0
            assert rt_data["exploits"][0]["gold_sanity_status"] == "not_run"
            assert (out_unpatched / "exploits.jsonl").is_file()
            # Verify --network=none actually blocked egress inside real HackerAgent run_cmd!
            all_tool_obs = [
                str(m.get("content", ""))
                for req in recorded
                for m in cast(list[dict[str, Any]], req["body"].get("messages", []))
                if m.get("role") == "tool"
            ]
            assert any("EGRESS_BLOCKED" in obs for obs in all_tool_obs)
            assert any("def add(a: int, b: int)" in obs for obs in all_tool_obs)

            # 2. `dasv redteam --propose-fix` (dry-run default) -> VERIFIED_FIX, exit 0, untouched
            out_fixed = tmp_path / "rt_fixed"
            orig_test_sh = (env_dir / "tests" / "test.sh").read_text(encoding="utf-8")
            res_pass = runner.invoke(
                app,
                [
                    "redteam",
                    str(env_dir),
                    "--base-url",
                    base_url,
                    "--model",
                    "local-mock",
                    "--hack-classes",
                    "hardcoding_lookup",
                    "--propose-fix",
                    "--gold-sanity-replays",
                    "2",
                    "--out-dir",
                    str(out_fixed),
                ],
            )
            assert res_pass.exit_code == EXIT_PASS, res_pass.stdout + res_pass.stderr
            fixed_data = json.loads((out_fixed / "redteam.json").read_text(encoding="utf-8"))
            assert len(fixed_data["exploits"]) == 1
            assert fixed_data["exploits"][0]["gold_sanity_status"] == "verified_fix"
            assert list((out_fixed / "fixes").glob("*.diff"))
            # Without --apply, source tests/test.sh MUST remain untouched
            assert (env_dir / "tests" / "test.sh").read_text(encoding="utf-8") == orig_test_sh

            # 3. `dasv audit --tiers 0,1,2 --propose-fix --html ... --sarif ...` -> live pipeline
            audit_out = tmp_path / "audit_e2e"
            audit_html = audit_out / "report.html"
            audit_sarif = audit_out / "report.sarif"
            res_audit = runner.invoke(
                app,
                [
                    "audit",
                    str(env_dir),
                    "--tiers",
                    "0,1,2",
                    "--no-native",
                    "--replays",
                    "2",
                    "--noop-replays",
                    "1",
                    "--base-url",
                    base_url,
                    "--redteam-model",
                    "local-mock",
                    "--hack-classes",
                    "hardcoding_lookup",
                    "--propose-fix",
                    "--gold-sanity-replays",
                    "1",
                    "--fail-on",
                    "critical",
                    "--out-dir",
                    str(audit_out),
                    "--sarif",
                    str(audit_sarif),
                    "--html",
                    str(audit_html),
                ],
            )
            # Note: Dockerfile uses unpinned python:3.12-slim (HIGH RULE-HERM-001), so with
            # --fail-on critical and exploit closed by verified_fix -> Grade D, exit 0!
            assert res_audit.exit_code == EXIT_PASS, res_audit.stdout + res_audit.stderr
            report = AuditReport.model_validate_json(
                (audit_out / "report.json").read_text(encoding="utf-8")
            )
            assert report.determinism is not None
            assert report.determinism.oracle_pass_rate == 1.0
            assert report.determinism.noop_pass_rate == 0.0
            assert len(report.exploits) == 1
            assert report.exploits[0].gold_sanity_status == "verified_fix"
            html_text = audit_html.read_text(encoding="utf-8")
            assert "Adversarial Exploit Traces" in html_text
            assert "verified_fix" in html_text
            assert "Trajectory (" in html_text
            assert "http://" not in html_text and "https://" not in html_text
            assert audit_sarif.is_file()

            # 4. Exit code 3 (EXIT_BUDGET_EXCEEDED) via tight --max-tokens cap
            res_budget = runner.invoke(
                app,
                [
                    "redteam",
                    str(env_dir),
                    "--base-url",
                    base_url,
                    "--model",
                    "local-mock",
                    "--hack-classes",
                    "hardcoding_lookup",
                    "--max-tokens",
                    "10",
                ],
            )
            assert res_budget.exit_code == EXIT_BUDGET_EXCEEDED, (
                res_budget.stdout + res_budget.stderr
            )

            # 5. Exit code 2 (EXIT_CONFIG_ERROR) via invalid hack class
            res_cfg = runner.invoke(
                app,
                ["redteam", str(env_dir), "--hack-classes", "nonexistent_attack"],
            )
            assert res_cfg.exit_code == EXIT_CONFIG_ERROR

            # 6. Exit code 4 (EXIT_RUNTIME_ERROR) via unreachable LLM endpoint
            res_rt = runner.invoke(
                app,
                [
                    "redteam",
                    str(env_dir),
                    "--base-url",
                    "http://127.0.0.1:1/v1",
                    "--model",
                    "local-mock",
                    "--hack-classes",
                    "hardcoding_lookup",
                ],
            )
            assert res_rt.exit_code == EXIT_RUNTIME_ERROR

            # 7. `--apply` applies verified fix directly to target env_dir
            res_apply = runner.invoke(
                app,
                [
                    "redteam",
                    str(env_dir),
                    "--base-url",
                    base_url,
                    "--model",
                    "local-mock",
                    "--hack-classes",
                    "hardcoding_lookup",
                    "--propose-fix",
                    "--gold-sanity-replays",
                    "1",
                    "--apply",
                ],
            )
            assert res_apply.exit_code == EXIT_PASS, res_apply.stdout + res_apply.stderr
            assert "assert calculator.add(10, 20) == 30" in (
                env_dir / "tests" / "test.sh"
            ).read_text(encoding="utf-8")

        # Verify ZERO orphaned dasv-managed containers remain after all runs
        ps = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=dasv.managed=true"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert ps.stdout.strip() == "", f"Orphaned containers found: {ps.stdout}"
    finally:
        # Always clean up ephemeral built test images tagged dasv-live-weak-calc:*
        imgs = subprocess.run(
            ["docker", "images", "-q", "dasv-live-weak-calc"],
            capture_output=True,
            text=True,
            check=False,
        )
        for img_id in set(imgs.stdout.splitlines()):
            if img_id.strip():
                subprocess.run(["docker", "rmi", "-f", img_id.strip()], capture_output=True)
