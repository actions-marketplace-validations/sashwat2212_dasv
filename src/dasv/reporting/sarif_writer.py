"""SARIF 2.1.0 report builder and writer for GitHub Code Scanning (FEAT-19, SDD-04)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dasv import __version__
from dasv.core.schemas import AuditReport, Severity

SARIF_SCHEMA_URI = "https://json.schemastore.org/sarif-2.1.0.json"

_SEVERITY_TO_SARIF_LEVEL: dict[Severity, str] = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
    Severity.INFO: "note",
}


def _resolve_repo_relative_prefix(env_root: Path) -> str:
    """Return POSIX path of `env_root` relative to its enclosing git repo root, if any."""
    try:
        resolved = env_root.resolve()
    except OSError:
        return ""
    search_dir = resolved if resolved.is_dir() else resolved.parent
    for candidate in (search_dir, *search_dir.parents):
        if (candidate / ".git").exists():
            try:
                rel = resolved.relative_to(candidate).as_posix()
            except ValueError:
                return ""
            return "" if rel in {"", "."} else rel
    return ""


def _normalize_sarif_uri(file_path: str | None, *, repo_prefix: str = "") -> str:
    """Normalize finding file paths to repo-relative SARIF artifact URIs preserving dotfiles."""
    if not file_path:
        return repo_prefix or "."
    raw_posix = file_path.replace("\\", "/")
    is_abs = raw_posix.startswith("/")
    normalized = raw_posix
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    if not normalized or normalized == ".":
        return repo_prefix or "."
    if is_abs or not repo_prefix:
        return normalized
    if normalized == repo_prefix or normalized.startswith(f"{repo_prefix}/"):
        return normalized
    return f"{repo_prefix}/{normalized}"


def build_sarif_dict(report: AuditReport) -> dict[str, Any]:
    """Convert an AuditReport into a SARIF 2.1.0-compliant dictionary."""
    rules_by_id: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    repo_prefix = _resolve_repo_relative_prefix(report.env_spec.root_path)

    for finding in report.findings:
        level = _SEVERITY_TO_SARIF_LEVEL[finding.severity]

        if finding.rule_id not in rules_by_id:
            rules_by_id[finding.rule_id] = {
                "id": finding.rule_id,
                "name": finding.rule_id,
                "shortDescription": {"text": finding.title},
                "help": {
                    "text": finding.remediation,
                    "markdown": f"**Remediation:** {finding.remediation}",
                },
                "defaultConfiguration": {"level": level},
                "properties": {
                    "dasvSeverity": finding.severity.value,
                    "dasvTier": finding.tier,
                    "dasvSource": finding.source,
                },
            }

        uri = _normalize_sarif_uri(finding.file_path, repo_prefix=repo_prefix)
        physical_location: dict[str, Any] = {
            "artifactLocation": {"uri": uri},
        }
        if finding.line_number is not None and finding.line_number >= 1:
            physical_location["region"] = {"startLine": finding.line_number}

        results.append(
            {
                "ruleId": finding.rule_id,
                "level": level,
                "message": {"text": f"{finding.title}: {finding.evidence}"},
                "locations": [{"physicalLocation": physical_location}],
                "properties": {
                    "dasvSeverity": finding.severity.value,
                    "dasvTier": finding.tier,
                    "dasvSource": finding.source,
                    "remediation": finding.remediation,
                },
            }
        )

    return {
        "$schema": SARIF_SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "dasv",
                        "semanticVersion": __version__,
                        "informationUri": "https://github.com/sashwat2212/dasv",
                        "rules": list(rules_by_id.values()),
                    }
                },
                "results": results,
                "properties": {
                    "runId": report.manifest.run_id,
                    "taskId": report.env_spec.task_id,
                    "grade": report.grade,
                    "hackabilityScore": report.hackability_score,
                },
            }
        ],
    }


def write_sarif(report: AuditReport, path: Path) -> Path:
    """Write a SARIF 2.1.0 JSON file to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_sarif_dict(report)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
