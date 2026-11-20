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


@dataclass(frozen=True)
class _SignalPoint:
    k: int
    x: int
    pass_at_k_y: int
    pass_pow_k_y: int
    pass_at_k_label: str
    pass_pow_k_label: str


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


def _build_signal_curve(
    pass_at_k: dict[int, float],
    pass_pow_k: dict[int, float],
) -> tuple[list[_SignalPoint], str, str, int]:
    ks = sorted(set(pass_at_k.keys()) | set(pass_pow_k.keys()))
    if not ks:
        return [], "", "", 260
    top_y = 28
    bottom_y = 116
    span_y = bottom_y - top_y
    points: list[_SignalPoint] = []
    pak_coords: list[str] = []
    ppk_coords: list[str] = []
    for idx, k in enumerate(ks):
        x = 36 + idx * 72
        pak_val = max(0.0, min(1.0, float(pass_at_k.get(k, 0.0))))
        ppk_val = max(0.0, min(1.0, float(pass_pow_k.get(k, 0.0))))
        pak_y = int(round(bottom_y - pak_val * span_y))
        ppk_y = int(round(bottom_y - ppk_val * span_y))
        points.append(
            _SignalPoint(
                k=k,
                x=x,
                pass_at_k_y=pak_y,
                pass_pow_k_y=ppk_y,
                pass_at_k_label=f"{pak_val * 100.0:.0f}%",
                pass_pow_k_label=f"{ppk_val * 100.0:.0f}%",
            )
        )
        pak_coords.append(f"{x},{pak_y}")
        ppk_coords.append(f"{x},{ppk_y}")
    width = max(260, 72 + len(ks) * 72)
    return points, " ".join(pak_coords), " ".join(ppk_coords), width


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

    signal_points: list[_SignalPoint] = []
    signal_pak_poly = ""
    signal_ppk_poly = ""
    signal_svg_width = 260
    if report.signal_profile is not None:
        (
            signal_points,
            signal_pak_poly,
            signal_ppk_poly,
            signal_svg_width,
        ) = _build_signal_curve(
            report.signal_profile.pass_at_k,
            report.signal_profile.pass_pow_k,
        )

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
        signal_points=signal_points,
        signal_pak_poly=signal_pak_poly,
        signal_ppk_poly=signal_ppk_poly,
        signal_svg_width=signal_svg_width,
        severity_counts=severity_counts,
    )


def write_html_report(report: AuditReport, path: Path) -> Path:
    """Render and write the offline HTML Certification Report to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(report), encoding="utf-8")
    return path
