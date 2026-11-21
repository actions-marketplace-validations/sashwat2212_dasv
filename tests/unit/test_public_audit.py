"""Unit & integrity tests for the KPI-RES-02 public environment audit artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_DIR = _REPO_ROOT / "results" / "public_audit"
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_EXTERNAL_URL_RE = re.compile(r"https?://", re.IGNORECASE)


def test_kpi_res_02_public_audit_artifacts_and_metrics() -> None:
    """Verify reproducible KPI-RES-02 dataset integrity across >=30 real public environments."""
    summary_path = _AUDIT_DIR / "summary.json"
    sources_path = _AUDIT_DIR / "sources.json"
    report_md_path = _AUDIT_DIR / "AUDIT_REPORT.md"

    assert summary_path.is_file()
    assert sources_path.is_file()
    assert report_md_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    report_md = report_md_path.read_text(encoding="utf-8")

    # KPI-RES-02 requires 30-50 real public tasks across Harbor, verifiers, and OpenEnv
    total = int(summary["total_tasks_audited"])
    assert 30 <= total <= 50
    assert len(summary["tasks"]) == total
    assert summary["api_cost_usd"] == 0.0

    # All 3 pinned upstream sources must record valid 40-char commit SHAs and licenses
    assert set(sources.keys()) == {"terminal_bench_2", "prime_verifiers", "meta_openenv"}
    for src in sources.values():
        assert _HEX40_RE.match(src["pinned_sha"])
        assert src["license"] in {"Apache-2.0", "MIT", "BSD-3-Clause"}

    ecosystems = set(summary["ecosystem_breakdown"].keys())
    assert ecosystems == {"harbor", "verifiers_v1", "openenv"}

    # Unique-task counts must never exceed total tasks or total finding occurrences
    rule_task_freq: dict[str, int] = summary["rule_task_frequency"]
    rule_freq: dict[str, int] = summary["rule_frequency"]
    assert "RULE-HERM-001" in rule_task_freq
    assert "RULE-BNDY-002" in rule_task_freq
    assert "RULE-BNDY-001" in rule_task_freq
    for rule_id, unique_cnt in rule_task_freq.items():
        assert 0 < unique_cnt <= total
        assert unique_cnt <= rule_freq[rule_id]

    # Concrete Terminal-Bench 2.0 boundary leaks must be documented
    leaks = summary["notable_boundary_leaks"]
    assert any(
        lk["task_id"] == "break-filter-js-from-html" and lk["rule_id"] == "RULE-BNDY-001"
        for lk in leaks
    )
    assert "break-filter-js-from-html" in report_md
    assert "allow_internet = true" in report_md

    # Live Tier 2 CHERRL cross-family results must include non-degenerate fluff-hack divergence
    cherrl_rows = summary["cherrl_live_ollama_audit"]
    assert len(cherrl_rows) >= 5
    color_row = next(r for r in cherrl_rows if r["task_id"] == "color-codeword")
    assert color_row["divergent"] is True
    assert color_row["degenerate_reason"] is None
    assert len(color_row["scores_by_judge"]) == 2
    assert color_row["mean_abs_delta"] > 0.15


def test_public_audit_offline_html_cards_have_zero_cdn_urls() -> None:
    """Archived HTML Certification Cards must be strictly offline (zero external HTTP/HTTPS)."""
    cards_dir = _AUDIT_DIR / "cards"
    for filename in ("card_cleanest.html", "card_most_vulnerable.html"):
        card = cards_dir / filename
        assert card.is_file()
        html = card.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "dasv Certification Report" in html
        assert not _EXTERNAL_URL_RE.search(html), f"External URL leaked into {filename}"
