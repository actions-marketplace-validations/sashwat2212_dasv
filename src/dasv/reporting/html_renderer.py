"""Zero-CDN offline interactive HTML Certification Report renderer (SDD-04, FEAT-19)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from dasv.core.schemas import AuditReport, Severity

_GRADE_COLORS: dict[str, tuple[str, str]] = {
    "A": ("#22c55e", "FRONTIER-READY"),
    "B": ("#84cc16", "HARDENED"),
    "C": ("#f59e0b", "NEEDS HARDENING"),
    "D": ("#f97316", "HIGH RISK"),
    "F": ("#ef4444", "REJECTED"),
}

_GAUGE_CIRCUMFERENCE = 251.33  # 2 * pi * 40


@dataclass(frozen=True)
class _PassPowBar:
    k: int
    value: float
    pct_label: str
    x: int
    y: int
    height: int
    color: str


@dataclass(frozen=True)
class _SeverityCount:
    name: str
    count: int


def _build_jinja_env() -> Environment:
    return Environment(
        loader=PackageLoader("dasv.reporting", "templates"),
        autoescape=select_autoescape(enabled_extensions=("html", "j2", "xml"), default=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _hackability_gauge_color(score: float) -> str:
    if score <= 0.0:
        return "#22c55e"
    if score <= 10.0:
        return "#84cc16"
    if score <= 25.0:
        return "#f59e0b"
    if score <= 50.0:
        return "#f97316"
    return "#ef4444"


def _build_pass_pow_bars(pass_pow_k: dict[int, float]) -> tuple[list[_PassPowBar], int]:
    bars: list[_PassPowBar] = []
    max_bar_height = 84
    base_y = 118
    for idx, k in enumerate(sorted(pass_pow_k)):
        val = max(0.0, min(1.0, float(pass_pow_k[k])))
        bar_h = max(2, int(round(val * max_bar_height)))
        color = "#22c55e" if val >= 0.999 else ("#f59e0b" if val >= 0.5 else "#ef4444")
        bars.append(
            _PassPowBar(
                k=k,
                value=val,
                pct_label=f"{val * 100.0:.0f}%",
                x=20 + idx * 72,
                y=base_y - bar_h,
                height=bar_h,
                color=color,
            )
        )
    width = max(220, 32 + len(bars) * 72)
    return bars, width


def render_html_report(report: AuditReport) -> str:
    """Render an offline, zero-CDN, XSS-safe HTML Certification Report string."""
    env = _build_jinja_env()
    template = env.get_template("report.html.j2")

    grade_color, grade_label = _GRADE_COLORS.get(report.grade, ("#93a4bd", "UNKNOWN"))
    hack_score = 0.0 if report.hackability_score is None else float(report.hackability_score)
    clamped_hack = max(0.0, min(100.0, hack_score))
    hack_dash = round((clamped_hack / 100.0) * _GAUGE_CIRCUMFERENCE, 2)

    pass_pow_bars: list[_PassPowBar] = []
    pass_pow_svg_width = 220
    if report.determinism is not None and report.determinism.pass_pow_k:
        pass_pow_bars, pass_pow_svg_width = _build_pass_pow_bars(report.determinism.pass_pow_k)

    severity_counts: list[_SeverityCount] = []
    for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO):
        cnt = sum(1 for f in report.findings if f.severity == sev)
        if cnt > 0:
            severity_counts.append(_SeverityCount(name=sev.value, count=cnt))

    return template.render(
        report=report,
        grade_color=grade_color,
        grade_label=grade_label,
        hack_score_display=f"{clamped_hack:.1f}",
        hack_gauge_dash=hack_dash,
        hack_gauge_color=_hackability_gauge_color(clamped_hack),
        pass_pow_bars=pass_pow_bars,
        pass_pow_svg_width=pass_pow_svg_width,
        severity_counts=severity_counts,
    )


def write_html_report(report: AuditReport, path: Path) -> Path:
    """Render and write the offline HTML Certification Report to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(report), encoding="utf-8")
    return path

