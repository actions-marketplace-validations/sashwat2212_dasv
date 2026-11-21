"""Unit & real-HTTP tests for FEAT-16 CHERRL cross-family judge divergence (RULE-RUBR-003)."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from dasv.cli.app import app
from dasv.core.errors import (
    EXIT_AUDIT_FAIL,
    EXIT_BUDGET_EXCEEDED,
    EXIT_CONFIG_ERROR,
    EXIT_PASS,
    BudgetExceededError,
    ConfigError,
)
from dasv.core.schemas import Severity
from dasv.rubric import (
    CallableRubricScorer,
    JudgeSample,
    ProRubricAuditor,
    bootstrap_kendall_tau_ci,
    detect_reference_free_judge,
    kendall_tau_b,
    load_judge_samples_jsonl,
    mean_abs_delta,
    run_judge_divergence_audit,
    spearman_rho,
)

runner = CliRunner()


def test_kendall_tau_b_and_spearman_rho_exact_ties_and_degenerate() -> None:
    """Verify exact tie-aware Kendall tau-b, Spearman rho, MAD, and deterministic bootstrap CI."""
    # 1. Perfect monotone agreement & inversion
    assert kendall_tau_b([0.1, 0.3, 0.6, 0.9], [0.2, 0.4, 0.7, 1.0]) == pytest.approx(1.0)
    assert spearman_rho([0.1, 0.3, 0.6, 0.9], [0.2, 0.4, 0.7, 1.0]) == pytest.approx(1.0)
    assert kendall_tau_b([0.1, 0.3, 0.6, 0.9], [0.9, 0.6, 0.3, 0.1]) == pytest.approx(-1.0)
    assert spearman_rho([0.1, 0.3, 0.6, 0.9], [0.9, 0.6, 0.3, 0.1]) == pytest.approx(-1.0)

    # 2. Analytical tie-bearing fixture:
    # x = [1, 1, 2, 3], y = [1, 2, 2, 3]
    # n0 = 6 pairs:
    #   (0,1): tie in x only; (0,2): conc; (0,3): conc;
    #   (1,2): tie in y only; (1,3): conc; (2,3): conc.
    # C=4, D=0, ties_x=1, ties_y=1 => tau_b = 4 / sqrt(5 * 5) = 0.8
    assert kendall_tau_b([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0]) == pytest.approx(0.8)
    # Spearman midranks: rx = [1.5, 1.5, 3.0, 4.0], ry = [1.0, 2.5, 2.5, 4.0]
    # mean = 2.5; dx = [-1, -1, 0.5, 1.5], dy = [-1.5, 0, 0, 1.5]
    # cov = 1.5 + 0 + 0 + 2.25 = 3.75; var_x = 1+1+0.25+2.25 = 4.5; var_y = 2.25+0+0+2.25 = 4.5
    # rho = 3.75 / 4.5 = 5/6
    assert spearman_rho([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0]) == pytest.approx(5.0 / 6.0)
    assert mean_abs_delta([0.2, 0.8], [0.5, 0.4]) == pytest.approx(0.35)

    # 3. Degenerate inputs (n < 3, empty, constant) must never raise or return NaN
    for xs, ys in [([], []), ([0.5], [0.5]), ([0.5, 0.5, 0.5], [0.1, 0.5, 0.9])]:
        tau = kendall_tau_b(xs, ys)
        rho = spearman_rho(xs, ys)
        assert not math.isnan(tau) and tau == 0.0
        assert not math.isnan(rho) and rho == 0.0

    # 4. Fixed-seed bootstrap CI is deterministic and brackets valid bounds
    lo1, hi1 = bootstrap_kendall_tau_ci([0.1, 0.2, 0.5, 0.8, 0.9], [0.1, 0.3, 0.4, 0.85, 0.95])
    lo2, hi2 = bootstrap_kendall_tau_ci([0.1, 0.2, 0.5, 0.8, 0.9], [0.1, 0.3, 0.4, 0.85, 0.95])
    assert (lo1, hi1) == (lo2, hi2)
    assert -1.0 <= lo1 <= hi1 <= 1.0
    # n < 3 collapses CI cleanly to point estimate
    assert bootstrap_kendall_tau_ci([0.1, 0.9], [0.1, 0.9]) == (1.0, 1.0)


async def test_judge_divergence_severity_matrix_and_partial_budget_abort(tmp_path: Path) -> None:
    """Verify HIGH (ref-free + divergent), MEDIUM (gated + divergent), PASS, and budget abort."""
    samples = [
        JudgeSample(sample_id=f"s{i}", task_prompt="Explain merge sort", candidate_response=f"R{i}")
        for i in range(5)
    ]
    scores_a = {"R0": 0.1, "R1": 0.3, "R2": 0.5, "R3": 0.7, "R4": 0.95}
    scores_b_agree = {"R0": 0.12, "R1": 0.32, "R2": 0.48, "R3": 0.72, "R4": 0.92}
    scores_b_invert = {"R0": 0.95, "R1": 0.75, "R2": 0.5, "R3": 0.25, "R4": 0.05}

    ja = CallableRubricScorer(lambda _p, r: scores_a[r])
    jb_agree = CallableRubricScorer(lambda _p, r: scores_b_agree[r])
    jb_invert = CallableRubricScorer(lambda _p, r: scores_b_invert[r])

    # 1. Anchored/gated + agreeing judges -> PASS (0 findings)
    res_pass = await run_judge_divergence_audit(
        samples,
        [("judge-a", ja), ("judge-b", jb_agree)],
        is_reference_free=False,
    )
    assert res_pass.passed is True
    assert res_pass.findings == []
    assert res_pass.pairs[0].kendall_tau == pytest.approx(1.0)
    assert res_pass.pairs[0].spearman_rho == pytest.approx(1.0)

    # 2. Anchored/gated + diverging judges -> MEDIUM RULE-RUBR-003
    res_med = await run_judge_divergence_audit(
        samples,
        [("judge-a", ja), ("judge-b", jb_invert)],
        is_reference_free=False,
    )
    assert res_med.passed is False
    assert len(res_med.findings) == 1
    assert res_med.findings[0].rule_id == "RULE-RUBR-003"
    assert res_med.findings[0].severity == Severity.MEDIUM
    assert res_med.pairs[0].kendall_tau == pytest.approx(-1.0)

    # 3. Reference-free + diverging judges -> HIGH RULE-RUBR-003
    res_high = await run_judge_divergence_audit(
        samples,
        [("judge-a", ja), ("judge-b", jb_invert)],
        is_reference_free=True,
    )
    assert res_high.passed is False
    assert res_high.findings[0].rule_id == "RULE-RUBR-003"
    assert res_high.findings[0].severity == Severity.HIGH

    # 4. Budget abort mid-batch preserves aligned prefix & emits RULE-BUDG-001
    calls = {"n": 0}

    def exploding_b(_p: str, r: str) -> float:
        calls["n"] += 1
        if calls["n"] >= 4:
            raise BudgetExceededError("token cap exceeded on sample 4")
        return scores_b_agree[r]

    res_partial = await run_judge_divergence_audit(
        samples,
        [("judge-a", ja), ("judge-b", CallableRubricScorer(exploding_b))],
        is_reference_free=False,
    )
    assert res_partial.partial_budget_exceeded is True
    assert res_partial.sample_count == 3
    assert len(res_partial.scores_by_judge["judge-a"]) == 3
    assert len(res_partial.scores_by_judge["judge-b"]) == 3
    assert any(f.rule_id == "RULE-BUDG-001" for f in res_partial.findings)

    # 5. < 2 judges raises ConfigError
    with pytest.raises(ConfigError):
        await run_judge_divergence_audit(samples, [("only-one", ja)])

    # 6. Reference-free detection distinguishes anchored vs unanchored files
    ref_file = tmp_path / "anchored_judge.py"
    ref_file.write_text("def score(resp, reference_answer): return 1.0\n", encoding="utf-8")
    is_rf_anchored, _ = detect_reference_free_judge(ref_file, [])
    assert is_rf_anchored is False

    unanchored_file = tmp_path / "free_judge.py"
    unanchored_file.write_text("def score(resp): return 0.5\n", encoding="utf-8")
    is_rf_free, _ = detect_reference_free_judge(unanchored_file, [])
    assert is_rf_free is True


@contextmanager
def _mock_dual_judge_http_server(*, invert_second_judge: bool) -> Generator[str, None, None]:
    """Localhost OpenAI-compat server acting as two distinct model families by `model` field."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = cast(dict[str, Any], json.loads(self.rfile.read(length).decode("utf-8")))
            model = str(body.get("model", "judge-a"))
            msgs = cast(list[dict[str, Any]], body.get("messages", []))
            user_text = str(msgs[-1].get("content", "")) if msgs else ""
            # Deterministic length-based raw score in [0.1, 0.9]
            norm_score = min(0.9, max(0.1, (len(user_text) % 500) / 500.0))
            if invert_second_judge and model == "judge-b":
                norm_score = round(1.0 - norm_score, 3)
            payload = json.dumps(
                {
                    "id": f"cherrl-{model}",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"score": norm_score, "rationale": "ok"}),
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 12},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_cli_rubric_cherrl_real_http_agree_diverge_and_budget_abort(tmp_path: Path) -> None:
    """Verify `dasv rubric --judges` over real localhost HTTP + --samples + budget exit 3."""
    hardened_rubric = tmp_path / "gated_rubric.yaml"
    hardened_rubric.write_text(
        "name: gated\naggregation: conjunctive_gated\ncriteria:\n"
        "  - name: correctness\n    mandatory: true\n",
        encoding="utf-8",
    )
    samples_file = tmp_path / "samples.jsonl"
    samples_file.write_text(
        "\n".join(
            json.dumps({"prompt": "Sort array", "response": "x" * (40 + i * 70)}) for i in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(load_judge_samples_jsonl(samples_file)) == 4

    # 1. Agreeing judges on conjunctive-gated rubric -> EXIT_PASS (0)
    with _mock_dual_judge_http_server(invert_second_judge=False) as url:
        out_ok = tmp_path / "out_ok"
        res_ok = runner.invoke(
            app,
            [
                "rubric",
                str(hardened_rubric),
                "--judges",
                "judge-a,judge-b",
                "--judge-base-urls",
                url,
                "--samples",
                str(samples_file),
                "--out-dir",
                str(out_ok),
            ],
        )
        assert res_ok.exit_code == EXIT_PASS, res_ok.stdout + res_ok.stderr
        data = json.loads((out_ok / "rubric_audit.json").read_text(encoding="utf-8"))
        assert data["judge_divergence"]["sample_count"] == 4
        assert data["judge_divergence"]["pairs"][0]["kendall_tau"] == pytest.approx(1.0)

    # 2. Diverging judges on gated rubric -> EXIT_AUDIT_FAIL (1) with RULE-RUBR-003 MEDIUM
    with _mock_dual_judge_http_server(invert_second_judge=True) as url:
        out_div = tmp_path / "out_div"
        res_div = runner.invoke(
            app,
            [
                "rubric",
                str(hardened_rubric),
                "--judges",
                "judge-a,judge-b",
                "--judge-base-urls",
                f"{url},{url}",
                "--samples",
                str(samples_file),
                "--out-dir",
                str(out_div),
            ],
        )
        assert res_div.exit_code == EXIT_AUDIT_FAIL, res_div.stdout + res_div.stderr
        div_data = json.loads((out_div / "rubric_audit.json").read_text(encoding="utf-8"))
        assert any(f["rule_id"] == "RULE-RUBR-003" for f in div_data["findings"])

        # 3. Tight --max-tokens -> EXIT_BUDGET_EXCEEDED (3) with partial rubric_audit.json flushed
        out_budg = tmp_path / "out_budg"
        res_budg = runner.invoke(
            app,
            [
                "rubric",
                str(hardened_rubric),
                "--judges",
                "judge-a,judge-b",
                "--judge-base-urls",
                url,
                "--samples",
                str(samples_file),
                "--max-tokens",
                "10",
                "--out-dir",
                str(out_budg),
            ],
        )
        assert res_budg.exit_code == EXIT_BUDGET_EXCEEDED, res_budg.stdout + res_budg.stderr
        assert (out_budg / "rubric_audit.json").is_file()

    # 4. Invalid --judges (< 2 models) & mismatched --judge-base-urls -> EXIT_CONFIG_ERROR (2)
    assert runner.invoke(app, ["rubric", str(hardened_rubric), "--judges", "solo"]).exit_code == (
        EXIT_CONFIG_ERROR
    )
    assert (
        runner.invoke(
            app,
            [
                "rubric",
                str(hardened_rubric),
                "--judges",
                "a,b,c",
                "--judge-base-urls",
                "http://127.0.0.1:1/v1,http://127.0.0.1:2/v1",
            ],
        ).exit_code
        == EXIT_CONFIG_ERROR
    )
    _ = ProRubricAuditor
