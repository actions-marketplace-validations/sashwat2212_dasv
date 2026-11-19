"""JSON, manifest, SARIF 2.1.0, and zero-CDN offline HTML report renderers (SDD-04)."""

from __future__ import annotations

from dasv.reporting.html_renderer import render_html_report, write_html_report
from dasv.reporting.json_writer import write_manifest_json, write_report_json
from dasv.reporting.sarif_writer import build_sarif_dict, write_sarif

__all__ = [
    "build_sarif_dict",
    "render_html_report",
    "write_html_report",
    "write_manifest_json",
    "write_report_json",
    "write_sarif",
]
