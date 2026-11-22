"""Unit tests for JSON, RunManifest, and SARIF 2.1.0 report writers."""

from __future__ import annotations

import json
from pathlib import Path

from dasv import __version__
from dasv.core.manifest import build_run_manifest
from dasv.core.schemas import AuditFinding, AuditReport, EnvSpec, EnvStandard, Severity
from dasv.reporting.json_writer import write_manifest_json, write_report_json
from dasv.reporting.sarif_writer import SARIF_SCHEMA_URI, build_sarif_dict, write_sarif


def test_json_and_sarif_writers(hardened_harbor_env: Path, tmp_path: Path) -> None:
    manifest = build_run_manifest(hardened_harbor_env)
    spec = EnvSpec(
        task_id="sample-task",
        standard=EnvStandard.HARBOR,
        root_path=hardened_harbor_env,
        instruction_text="Do the task.",
    )
    findings = [
        AuditFinding(
            rule_id="RULE-BNDY-001",
            title="Tests copied into agent image",
            severity=Severity.CRITICAL,
            tier=0,
            source="dasv",
            file_path="environment/Dockerfile",
            line_number=4,
            evidence="COPY tests/ /app/tests/",
            remediation="Mount tests read-only.",
        ),
        AuditFinding(
            rule_id="RULE-HERM-004",
            title="Emulated arch warning",
            severity=Severity.MEDIUM,
            tier=0,
            source="dasv",
            file_path=None,
            line_number=None,
            evidence="Running amd64 on arm64 host.",
            remediation="Run CI on ubuntu-latest.",
        ),
        AuditFinding(
            rule_id="RULE-NATIVE-001",
            title="Optional native CLI absent",
            severity=Severity.INFO,
            tier=0,
            source="native_delegate",
            evidence="harbor not installed",
            remediation="Install harbor",
        ),
    ]
    report = AuditReport(
        manifest=manifest,
        env_spec=spec,
        grade="F",
        hackability_score=20.0,
        findings=findings,
    )

    rep_path = write_report_json(report, tmp_path / "out" / "report.json")
    man_path = write_manifest_json(manifest, tmp_path / "out" / "manifest.json")
    sarif_path = write_sarif(report, tmp_path / "out" / "report.sarif")

    assert AuditReport.model_validate_json(rep_path.read_text(encoding="utf-8")) == report
    assert json.loads(man_path.read_text(encoding="utf-8"))["run_id"] == manifest.run_id

    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["$schema"] == SARIF_SCHEMA_URI
    assert sarif["version"] == "2.1.0"
    assert len(sarif["runs"]) == 1

    run = sarif["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "dasv"
    assert driver["semanticVersion"] == __version__
    assert [r["id"] for r in driver["rules"]] == [
        "RULE-BNDY-001",
        "RULE-HERM-004",
        "RULE-NATIVE-001",
    ]

    results = run["results"]
    assert len(results) == 3
    assert [r["level"] for r in results] == ["error", "warning", "note"]
    loc0 = results[0]["locations"][0]["physicalLocation"]
    assert loc0["artifactLocation"]["uri"] == "environment/Dockerfile"
    assert loc0["region"]["startLine"] == 4
    assert results[1]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "."


def test_sarif_preserves_dotfiles_and_normalizes_prefixes(hardened_harbor_env: Path) -> None:
    manifest = build_run_manifest(hardened_harbor_env)
    spec = EnvSpec(
        task_id="dotfile-task",
        standard=EnvStandard.HARBOR,
        root_path=hardened_harbor_env,
        instruction_text="Check dotfile URIs.",
    )
    cases = [
        (".env", ".env"),
        ("./.env", ".env"),
        (".github/workflows/ci.yml", ".github/workflows/ci.yml"),
        ("./environment/Dockerfile", "environment/Dockerfile"),
        ("/etc/passwd", "etc/passwd"),
        ("./", "."),
        ("", "."),
        (None, "."),
    ]
    findings = [
        AuditFinding(
            rule_id=f"RULE-URI-{idx:03d}",
            title="URI normalization check",
            severity=Severity.LOW,
            tier=0,
            source="dasv",
            file_path=raw_path,
            line_number=1 if raw_path else None,
            evidence="synthetic",
            remediation="none",
        )
        for idx, (raw_path, _expected) in enumerate(cases)
    ]
    report = AuditReport(
        manifest=manifest,
        env_spec=spec,
        grade="B",
        hackability_score=0.0,
        findings=findings,
    )

    sarif = build_sarif_dict(report)
    actual_uris = [
        r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        for r in sarif["runs"][0]["results"]
    ]
    assert actual_uris == [expected for _raw, expected in cases]


def test_sarif_emits_repo_relative_uris_inside_git_checkout(tmp_path: Path) -> None:
    """When auditing a subdirectory inside a git repo, SARIF URIs must be repo-relative."""
    (tmp_path / ".git").mkdir()
    env_dir = tmp_path / "envs" / "my_task"
    (env_dir / "environment").mkdir(parents=True)
    (env_dir / "environment" / "Dockerfile").write_text("FROM python:3.12\n", encoding="utf-8")

    manifest = build_run_manifest(env_dir)
    spec = EnvSpec(
        task_id="my-task",
        standard=EnvStandard.HARBOR,
        root_path=env_dir,
        instruction_text="Solve it.",
    )
    report = AuditReport(
        manifest=manifest,
        env_spec=spec,
        grade="F",
        hackability_score=20.0,
        findings=[
            AuditFinding(
                rule_id="RULE-BNDY-001",
                title="Tests copied into agent image",
                severity=Severity.CRITICAL,
                tier=0,
                source="dasv",
                file_path="environment/Dockerfile",
                line_number=4,
                evidence="COPY tests/test_outputs.py /app",
                remediation="Mount tests read-only.",
            ),
            AuditFinding(
                rule_id="RULE-BNDY-002",
                title="Network not disabled",
                severity=Severity.HIGH,
                tier=0,
                source="dasv",
                file_path=None,
                line_number=None,
                evidence="allow_internet = true",
                remediation="Set allow_internet = false.",
            ),
        ],
    )
    sarif = build_sarif_dict(report)
    driver = sarif["runs"][0]["tool"]["driver"]
    assert driver["informationUri"] == "https://github.com/sashwat2212/dasv"
    uris = [
        r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        for r in sarif["runs"][0]["results"]
    ]
    assert uris == ["envs/my_task/environment/Dockerfile", "envs/my_task"]
