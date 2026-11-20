"""CLI smoke tests for dasv --version, doctor, inspect, audit --tiers 0, report, and stubs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dasv import __version__
from dasv.cli.app import app
from dasv.core.errors import EXIT_AUDIT_FAIL, EXIT_CONFIG_ERROR, EXIT_PASS, EXIT_RUNTIME_ERROR
from dasv.core.schemas import AuditReport, EnvSpec, EnvStandard

runner = CliRunner()


def test_cli_version_and_doctor() -> None:
    ver_res = runner.invoke(app, ["--version"])
    assert ver_res.exit_code == EXIT_PASS
    assert f"dasv {__version__}" in ver_res.stdout

    doc_res = runner.invoke(app, ["doctor"])
    assert doc_res.exit_code == EXIT_PASS
    assert "Python >= 3.11" in doc_res.stdout
    assert "Docker Runtime" in doc_res.stdout


def test_cli_inspect_harbor_openenv_and_verifiers(
    hardened_harbor_env: Path,
    minimal_openenv: Path,
    minimal_verifiers_v1: Path,
) -> None:
    for path, expected in (
        (hardened_harbor_env, EnvStandard.HARBOR),
        (minimal_openenv, EnvStandard.OPENENV),
        (minimal_verifiers_v1, EnvStandard.VERIFIERS_V1),
    ):
        res = runner.invoke(app, ["inspect", str(path), "--json"])
        assert res.exit_code == EXIT_PASS, res.stderr
        spec = EnvSpec.model_validate_json(res.stdout)
        assert spec.standard == expected


def test_cli_audit_tier0_pass_and_fail_with_sarif_and_report(
    hardened_harbor_env: Path,
    vulnerable_harbor_env: Path,
    legacy_verifiers_v0: Path,
    tmp_path: Path,
) -> None:
    # 1. Hardened Harbor passes with Grade A, JSON, SARIF, and zero-CDN HTML (exit 0)
    clean_out = tmp_path / "clean_run"
    clean_sarif = clean_out / "report.sarif"
    clean_html = clean_out / "report.html"
    res_ok = runner.invoke(
        app,
        [
            "audit",
            str(hardened_harbor_env),
            "--tiers",
            "0",
            "--no-native",
            "--out-dir",
            str(clean_out),
            "--sarif",
            str(clean_sarif),
            "--html",
            str(clean_html),
        ],
    )
    assert res_ok.exit_code == EXIT_PASS, res_ok.stdout + res_ok.stderr
    clean_report = AuditReport.model_validate_json(
        (clean_out / "report.json").read_text(encoding="utf-8")
    )
    assert clean_report.grade == "A"
    assert clean_report.hackability_score == 0.0
    assert (clean_out / "manifest.json").is_file()
    assert json.loads(clean_sarif.read_text(encoding="utf-8"))["version"] == "2.1.0"
    html_text = clean_html.read_text(encoding="utf-8")
    assert "hardened-calculator" in html_text
    assert "http://" not in html_text and "https://" not in html_text

    # 2. Vulnerable Harbor fails with Grade F and exit 1
    vuln_out = tmp_path / "vuln_run"
    vuln_sarif = vuln_out / "report.sarif"
    res_fail = runner.invoke(
        app,
        [
            "audit",
            str(vulnerable_harbor_env),
            "--tiers",
            "0",
            "--no-native",
            "--out-dir",
            str(vuln_out),
            "--sarif",
            str(vuln_sarif),
        ],
    )
    assert res_fail.exit_code == EXIT_AUDIT_FAIL
    vuln_report = AuditReport.model_validate_json(
        (vuln_out / "report.json").read_text(encoding="utf-8")
    )
    assert vuln_report.grade == "F"
    sarif_rules = {
        r["ruleId"]
        for r in json.loads(vuln_sarif.read_text(encoding="utf-8"))["runs"][0]["results"]
    }
    assert {"RULE-BNDY-001", "RULE-BNDY-002", "RULE-BNDY-003", "RULE-BNDY-004"}.issubset(
        sarif_rules
    )

    # 3. `dasv report` re-renders SARIF and offline HTML cleanly from saved report.json
    rerendered_sarif = tmp_path / "rerendered.sarif"
    rerendered_html = tmp_path / "rerendered.html"
    rep_res = runner.invoke(
        app,
        [
            "report",
            str(vuln_out / "report.json"),
            "--sarif",
            str(rerendered_sarif),
            "--html",
            str(rerendered_html),
        ],
    )
    assert rep_res.exit_code == EXIT_PASS
    assert json.loads(rerendered_sarif.read_text(encoding="utf-8"))["version"] == "2.1.0"
    assert "RULE-BNDY-001" in rerendered_html.read_text(encoding="utf-8")

    # 4. Legacy verifiers v0 fails Tier 0 with RULE-ADPT-001
    v0_res = runner.invoke(app, ["audit", str(legacy_verifiers_v0), "--tiers", "0", "--no-native"])
    assert v0_res.exit_code == EXIT_AUDIT_FAIL
    assert "RULE-ADPT-001" in v0_res.stdout


def test_cli_audit_honors_config_default_tiers_and_runtime_error_when_docker_absent(
    hardened_harbor_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    # 1. Config specifying default_tiers = [0] runs Tier 0 when --tiers is omitted
    cfg_file = tmp_path / "dasv.toml"
    cfg_file.write_text("[project]\ndefault_tiers = [0]\n", encoding="utf-8")
    res_cfg = runner.invoke(
        app,
        ["audit", str(hardened_harbor_env), "--config", str(cfg_file), "--no-native"],
    )
    assert res_cfg.exit_code == EXIT_PASS, res_cfg.stdout + res_cfg.stderr

    # 2. Requesting Tier 1 when Docker is unavailable exits with EXIT_RUNTIME_ERROR (4)
    async def missing_docker(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_docker)
    res_no_docker = runner.invoke(
        app,
        ["audit", str(hardened_harbor_env), "--tiers", "0,1", "--replays", "2", "--no-native"],
    )
    assert res_no_docker.exit_code == EXIT_RUNTIME_ERROR
    assert "Grade A" not in res_no_docker.stdout


def test_cli_mutate_dry_run_and_remaining_stage6_stubs(
    hardened_harbor_env: Path,
    tmp_path: Path,
) -> None:
    # Populate a small Python source & test file inside the fixture so all mutators fire
    (hardened_harbor_env / "src").mkdir(exist_ok=True)
    (hardened_harbor_env / "src" / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    if a < 0:\n"
        "        raise ValueError('neg')\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    (hardened_harbor_env / "tests" / "test_calc.py").write_text(
        "def test_add() -> None:\n    assert 2 + 3 == 5\n",
        encoding="utf-8",
    )

    mut_out = tmp_path / "mut_artifacts"
    mut_res = runner.invoke(
        app,
        ["mutate", str(hardened_harbor_env), "--dry-run", "--out-dir", str(mut_out)],
    )
    assert mut_res.exit_code == EXIT_PASS, mut_res.stdout + mut_res.stderr
    assert (mut_out / "mutations.json").is_file()
    mut_data = json.loads((mut_out / "mutations.json").read_text(encoding="utf-8"))
    assert mut_data["anti_patches_generated"] == 5
    assert mut_data["one_off_mutations_generated"] == 1
    assert len(mut_data["conflicting_variants"]) == 1

    for cmd in ("rubric", "profile"):
        res = runner.invoke(app, [cmd, str(hardened_harbor_env)])
        assert res.exit_code == EXIT_CONFIG_ERROR
        assert "Stage 6" in res.stderr


def test_cli_redteam_config_and_runtime_error_contracts(
    hardened_harbor_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    # 1. Invalid hack class -> EXIT_CONFIG_ERROR (2)
    bad_hc = runner.invoke(
        app,
        ["redteam", str(hardened_harbor_env), "--hack-classes", "not_a_real_class"],
    )
    assert bad_hc.exit_code == EXIT_CONFIG_ERROR
    assert "Invalid hack class" in bad_hc.stderr

    # 2. Missing Docker daemon -> EXIT_RUNTIME_ERROR (4)
    async def missing_docker(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_docker)
    no_docker = runner.invoke(
        app,
        ["redteam", str(hardened_harbor_env), "--hack-classes", "state_bypass"],
    )
    assert no_docker.exit_code == EXIT_RUNTIME_ERROR
    assert "Docker runtime unavailable" in no_docker.stderr
