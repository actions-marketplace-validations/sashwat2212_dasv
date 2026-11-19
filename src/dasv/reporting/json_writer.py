"""Deterministic JSON writers for AuditReport and RunManifest."""

from __future__ import annotations

from pathlib import Path

from dasv.core.schemas import AuditReport, RunManifest


def write_report_json(report: AuditReport, path: Path) -> Path:
    """Serialize AuditReport to UTF-8 pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def write_manifest_json(manifest: RunManifest, path: Path) -> Path:
    """Serialize RunManifest to UTF-8 pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
