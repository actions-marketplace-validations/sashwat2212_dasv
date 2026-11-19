"""Unit & security tests for the zero-CDN offline HTML Certification Report renderer (SDD-04)."""

from __future__ import annotations

import re
from pathlib import Path

from dasv.core.manifest import build_run_manifest
from dasv.core.schemas import (
    AuditFinding,
    AuditReport,
    CurriculumBucket,
    DeterminismMetrics,
    EnvSpec,
    EnvStandard,
    ExploitTrace,
    Severity,
    SignalProfile,
)
from dasv.reporting.html_renderer import render_html_report, write_html_report


def _make_sample_report(
    tmp_path: Path,
    *,
    emulated_arch: bool = False,
    with_xss: bool = False,
) -> AuditReport:
    xss = '<script>alert("xss")</script><img src=x onerror=alert(2)>'
    spec = EnvSpec(
        task_id=f"task-{xss}" if with_xss else "hardened-calculator",
        standard=EnvStandard.HARBOR,
        root_path=tmp_path,
        instruction_text=f"Solve {xss}" if with_xss else "Implement add(a, b) cleanly.",
        has_oracle=True,
        oracle_entrypoint="solution/solve.sh",
        verifier_paths=["tests/test.sh"],
    )
    manifest = build_run_manifest(
        tmp_path,
        cli_args=["audit", str(tmp_path), "--tiers", "0,1"],
        docker_server_arch="amd64",
        oci_image_digests={"python:3.12-slim": "sha256:0123456789abcdef"},
        models_used=["qwen2.5-coder:7b"],
        rng_seed=42,
        total_tokens=1200,
        total_cost_usd=0.0012,
        run_id="dasv-html-test",
    )
    findings = [
        AuditFinding(
            rule_id="RULE-BNDY-001",
            title=f"Bad title {xss}" if with_xss else "Tests copied into execution image",
            severity=Severity.CRITICAL,
            tier=0,
            source="dasv",
            file_path="environment/Dockerfile",
            line_number=4,
            evidence=f"COPY tests/ /app/tests/ {xss}" if with_xss else "COPY tests/ /app/tests/",
            remediation=f"Remove COPY {xss}" if with_xss else "Mount verifier read-only.",
        ),
        AuditFinding(
            rule_id="RULE-HERM-001",
            title="Floating base image tag",
            severity=Severity.HIGH,
            tier=0,
            source="dasv",
            file_path="environment/Dockerfile",
            line_number=1,
            evidence="FROM python:3.12-slim",
            remediation="Pin image by @sha256 digest.",
        ),
    ]
    det = DeterminismMetrics(
        oracle_trials=10,
        oracle_pass_rate=0.9,
        noop_trials=3,
        noop_pass_rate=0.0,
        replay_variance=0.09,
        pass_at_1=0.9,
        pass_pow_k={1: 0.9, 3: 0.729, 5: 0.59049, 10: 0.348678},
        latency_p50_ms=420.0,
        latency_p95_ms=890.0,
        peak_memory_mb=128.5,
        emulated_arch_warning=emulated_arch,
    )
    exploits = [
        ExploitTrace(
            exploit_id="exp-01",
            task_id=spec.task_id,
            hack_class="test_tampering",
            reward_obtained=1.0,
            steps=[{"turn": 1, "cmd": "rm -rf tests/"}],
            proposed_fix_diff=f"--- a/Dockerfile\n+++ b/Dockerfile\n-{xss}"
            if with_xss
            else "--- a/Dockerfile\n+++ b/Dockerfile\n-COPY tests/ /app/tests/",
            gold_sanity_status="verified_fix",
        )
    ]
    signal = SignalProfile(
        task_id=spec.task_id,
        base_model="qwen2.5:3b",
        rollouts_g=8,
        pass_at_1=0.25,
        pass_at_k={1: 0.25, 4: 0.68, 8: 0.90},
        pass_pow_k={1: 0.25, 3: 0.0156},
        advantage_variance=0.1875,
        zero_grad_risk=False,
        bucket=CurriculumBucket.GOLDILOCKS,
    )
    return AuditReport(
        manifest=manifest,
        env_spec=spec,
        grade="F",
        hackability_score=65.0,
        findings=findings,
        determinism=det,
        exploits=exploits,
        signal_profile=signal,
    )


def test_html_report_zero_cdn_and_inline_svgs(tmp_path: Path) -> None:
    report = _make_sample_report(tmp_path, emulated_arch=True, with_xss=False)
    out_file = tmp_path / "cert" / "report.html"
    written = write_html_report(report, out_file)

    assert written == out_file
    html = out_file.read_text(encoding="utf-8")
    assert len(html.encode("utf-8")) < 250_000

    # Zero external CDN / script / link references
    assert not re.search(r"https?://", html, re.IGNORECASE)
    assert "<script src=" not in html.lower()
    assert "<link " not in html.lower()

    # Executive SVG components, pass^k decay chart, banner, and sections present
    assert html.count("<svg") >= 3
    assert "pass^k RELIABILITY DECAY" in html
    assert "Emulated Architecture Warning:" in html
    assert "RULE-BNDY-001" in html
    assert "verified_fix" in html
    assert "goldilocks" in html
    assert "dasv-html-test" in html


def test_html_report_omits_emulated_banner_when_native(tmp_path: Path) -> None:
    report = _make_sample_report(tmp_path, emulated_arch=False, with_xss=False)
    html = render_html_report(report)
    assert "Emulated Architecture Warning:" not in html


def test_html_report_escapes_xss_payloads(tmp_path: Path) -> None:
    report = _make_sample_report(tmp_path, emulated_arch=False, with_xss=True)
    html = render_html_report(report)

    assert '<script>alert("xss")</script>' not in html
    assert "<img src=x onerror=alert(2)>" not in html
    assert "&lt;script&gt;alert(&#34;xss&#34;)&lt;/script&gt;" in html
    assert "&lt;img src=x onerror=alert(2)&gt;" in html
