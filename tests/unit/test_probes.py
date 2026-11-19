"""Unit tests for Tier 0 static probes (RULE-BNDY-001..004, RULE-HERM-001..002)."""

from __future__ import annotations

from pathlib import Path

from dasv.adapters.harbor import HarborAdapter
from dasv.core.schemas import Severity
from dasv.sandbox.probes import run_tier0_static_probes


def test_hardened_fixture_has_zero_tier0_findings(hardened_harbor_env: Path) -> None:
    spec = HarborAdapter().load(hardened_harbor_env)
    findings = run_tier0_static_probes(spec)
    assert findings == []


def test_vulnerable_fixture_triggers_all_six_tier0_rules(vulnerable_harbor_env: Path) -> None:
    spec = HarborAdapter().load(vulnerable_harbor_env)
    findings = run_tier0_static_probes(spec)
    by_rule = {f.rule_id: f for f in findings}

    expected_critical = {
        "RULE-BNDY-001",
        "RULE-BNDY-002",
        "RULE-BNDY-003",
        "RULE-BNDY-004",
    }
    expected_high = {"RULE-HERM-001", "RULE-HERM-002"}

    assert expected_critical.issubset(by_rule.keys())
    assert expected_high.issubset(by_rule.keys())

    for rule_id in expected_critical:
        assert by_rule[rule_id].severity == Severity.CRITICAL
    for rule_id in expected_high:
        assert by_rule[rule_id].severity == Severity.HIGH

    # Confirm line numbers and relative paths are populated accurately
    assert by_rule["RULE-BNDY-001"].file_path == "environment/Dockerfile"
    assert by_rule["RULE-BNDY-001"].line_number == 3
    assert "solution/solve.sh" in by_rule["RULE-BNDY-004"].evidence

    # Both Dockerfile ENV secret and task.toml HF token should be caught
    secret_findings = [f for f in findings if f.rule_id == "RULE-BNDY-003"]
    assert len(secret_findings) == 2

    # Both Dockerfile CMD pip install and entrypoint.sh apt-get install caught
    herm2_findings = [f for f in findings if f.rule_id == "RULE-HERM-002"]
    assert len(herm2_findings) == 2
