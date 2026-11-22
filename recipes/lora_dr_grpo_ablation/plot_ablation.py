"""Zero-dependency SVG, HTML & stats ablation plotter for Unhardened vs. dasv-Hardened RLVR."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from recipes.lora_dr_grpo_ablation.train_dr_grpo import (  # noqa: E402
    mcnemar_exact_p_value,
    paired_bootstrap_test,
    wilson_ci,
)


def load_arm_metrics(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed: Any = json.loads(stripped)
        if isinstance(parsed, dict):
            rows.append(cast(dict[str, Any], parsed))
    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows


def _polyline(rows: list[dict[str, Any]], key: str, max_step: int, max_val: float) -> str:
    if not rows:
        return ""
    pts: list[str] = []
    for row in rows:
        if key not in row:
            continue
        step = int(row.get("step", 0))
        val = float(row.get(key, 0.0))
        x = 56.0 + (step / max(1, max_step)) * 460.0
        y = 195.0 - (min(max_val, max(0.0, val)) / max(1e-9, max_val)) * 150.0
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _y_coord(val: float, max_val: float) -> float:
    return 195.0 - (min(max_val, max(0.0, val)) / max(1e-9, max_val)) * 150.0


def build_ablation_svgs(
    unhardened_rows: list[dict[str, Any]],
    hardened_rows: list[dict[str, Any]],
    *,
    arm0_pass_at_1: float | None = None,
    standalone: bool = False,
) -> dict[str, str]:
    """Build 3 publication-ready zero-dependency SVG figures for README/report embedding."""
    ns_attr = ' xmlns="http://www.w3.org/2000/svg"' if standalone else ""
    all_steps = [int(r.get("step", 0)) for r in (*unhardened_rows, *hardened_rows)]
    max_step = max(all_steps, default=1)

    def _with_step0(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if arm0_pass_at_1 is None:
            return rows
        if any(int(r.get("step", -1)) == 0 and "clean_heldout_pass_at_1" in r for r in rows):
            return rows
        return [{"step": 0, "clean_heldout_pass_at_1": arm0_pass_at_1}, *rows]

    u_train = _polyline(unhardened_rows, "train_reward", max_step, 1.0)
    u_held = _polyline(_with_step0(unhardened_rows), "clean_heldout_pass_at_1", max_step, 1.0)
    h_train = _polyline(hardened_rows, "train_reward", max_step, 1.0)
    h_held = _polyline(_with_step0(hardened_rows), "clean_heldout_pass_at_1", max_step, 1.0)

    arm0_line = ""
    if arm0_pass_at_1 is not None:
        y0 = _y_coord(arm0_pass_at_1, 1.0)
        arm0_line = (
            f'<line x1="56" y1="{y0:.1f}" x2="516" y2="{y0:.1f}" '
            f'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3,3" />'
            f'<text x="520" y="{y0 + 4:.1f}" fill="#94a3b8" font-size="10" '
            f'font-family="monospace">Arm0 {arm0_pass_at_1:.2f}</text>'
        )

    reward_svg = (
        f'<svg{ns_attr} width="580" height="230" viewBox="0 0 580 230" '
        'role="img" aria-label="Reward and clean held-out pass@1 curves">'
        '<rect width="580" height="230" rx="8" fill="#0f172a" />'
        '<line x1="56" y1="195" x2="516" y2="195" stroke="#334155" />'
        '<line x1="56" y1="45" x2="56" y2="195" stroke="#334155" />'
        f"{arm0_line}"
        f'<polyline fill="none" stroke="#f97316" stroke-width="2" stroke-dasharray="5,4" '
        f'points="{u_train}" />'
        f'<polyline fill="none" stroke="#ef4444" stroke-width="2.2" points="{u_held}" />'
        f'<polyline fill="none" stroke="#38bdf8" stroke-width="2" stroke-dasharray="5,4" '
        f'points="{h_train}" />'
        f'<polyline fill="none" stroke="#22c55e" stroke-width="2.6" points="{h_held}" />'
        '<text x="56" y="24" fill="#cbd5e1" font-size="11" font-family="monospace">'
        "Orange=Arm1 Train | Red=Arm1 HeldOut | Cyan=Arm2 Train | Green=Arm2 HeldOut"
        "</text>"
        '<text x="16" y="52" fill="#64748b" font-size="10" font-family="monospace">1.0</text>'
        '<text x="16" y="198" fill="#64748b" font-size="10" font-family="monospace">0.0</text>'
        f'<text x="468" y="216" fill="#64748b" font-size="10" font-family="monospace">'
        f"step {max_step}</text>"
        "</svg>"
    )

    u_tamper = _polyline(unhardened_rows, "tamper_rate", max_step, 1.0)
    h_tamper = _polyline(hardened_rows, "tamper_rate", max_step, 1.0)
    u_gap = _polyline(unhardened_rows, "hack_divergence_gap", max_step, 1.0)
    h_gap = _polyline(hardened_rows, "hack_divergence_gap", max_step, 1.0)

    hacking_svg = (
        f'<svg{ns_attr} width="580" height="230" viewBox="0 0 580 230" '
        'role="img" aria-label="Hacking divergence gap and tamper rate curves">'
        '<rect width="580" height="230" rx="8" fill="#0f172a" />'
        '<line x1="56" y1="195" x2="516" y2="195" stroke="#334155" />'
        '<line x1="56" y1="45" x2="56" y2="195" stroke="#334155" />'
        f'<polyline fill="none" stroke="#ef4444" stroke-width="2.2" points="{u_gap}" />'
        f'<polyline fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4,3" '
        f'points="{u_tamper}" />'
        f'<polyline fill="none" stroke="#22c55e" stroke-width="2.2" points="{h_gap}" />'
        f'<polyline fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="4,3" '
        f'points="{h_tamper}" />'
        '<text x="56" y="24" fill="#cbd5e1" font-size="11" font-family="monospace">'
        "Red=Arm1 HackGap | Amber=Arm1 TamperRate | Green=Arm2 Gap | Cyan=Arm2 Tamper"
        "</text>"
        '<text x="16" y="52" fill="#64748b" font-size="10" font-family="monospace">1.0</text>'
        '<text x="16" y="198" fill="#64748b" font-size="10" font-family="monospace">0.0</text>'
        "</svg>"
    )

    max_drift = max(
        [
            float(r.get("p95_abs_delta", 0.0))
            for r in (*unhardened_rows, *hardened_rows)
            if "p95_abs_delta" in r
        ]
        + [0.5]
    )
    u_drift = _polyline(unhardened_rows, "p95_abs_delta", max_step, max_drift)
    h_drift = _polyline(hardened_rows, "p95_abs_delta", max_step, max_drift)
    u_ice = _polyline(unhardened_rows, "icepop_masked_frac", max_step, 1.0)
    h_ice = _polyline(hardened_rows, "icepop_masked_frac", max_step, 1.0)

    drift_svg = (
        f'<svg{ns_attr} width="580" height="230" viewBox="0 0 580 230" '
        'role="img" aria-label="Train-inference logprob drift curves">'
        '<rect width="580" height="230" rx="8" fill="#0f172a" />'
        '<line x1="56" y1="195" x2="516" y2="195" stroke="#334155" />'
        '<line x1="56" y1="45" x2="56" y2="195" stroke="#334155" />'
        f'<polyline fill="none" stroke="#ef4444" stroke-width="2.2" points="{u_drift}" />'
        f'<polyline fill="none" stroke="#22c55e" stroke-width="2.2" points="{h_drift}" />'
        f'<polyline fill="none" stroke="#f97316" stroke-width="1.6" stroke-dasharray="4,3" '
        f'points="{u_ice}" />'
        f'<polyline fill="none" stroke="#38bdf8" stroke-width="1.6" stroke-dasharray="4,3" '
        f'points="{h_ice}" />'
        '<text x="56" y="24" fill="#cbd5e1" font-size="11" font-family="monospace">'
        "Red=Arm1 p95|delta| | Green=Arm2 p95|delta| | Dashed=IcePop Masked Frac"
        "</text>"
        f'<text x="10" y="52" fill="#64748b" font-size="10" font-family="monospace">'
        f"{max_drift:.2f}</text>"
        '<text x="16" y="198" fill="#64748b" font-size="10" font-family="monospace">0.0</text>'
        "</svg>"
    )

    return {
        "ablation_reward_and_heldout.svg": reward_svg,
        "ablation_hacking_and_tamper.svg": hacking_svg,
        "ablation_logprob_drift.svg": drift_svg,
    }


def compute_ablation_significance(
    arm1_eval: dict[str, Any],
    arm2_eval: dict[str, Any],
) -> dict[str, Any]:
    """Compute paired bootstrap + exact McNemar p-values from final heldout_eval records."""
    s1: dict[str, int] = arm1_eval.get("per_task_successes", {})
    s2: dict[str, int] = arm2_eval.get("per_task_successes", {})
    k1 = max(1, int(arm1_eval.get("k", 1)))
    k2 = max(1, int(arm2_eval.get("k", 1)))
    shared_ids = sorted(set(s1.keys()) & set(s2.keys()))
    if not shared_ids:
        return {"paired_tasks": 0}

    rates_1 = [s1[tid] / k1 for tid in shared_ids]
    rates_2 = [s2[tid] / k2 for tid in shared_ids]
    boot = paired_bootstrap_test(rates_1, rates_2)

    p1_bool = [bool(x) for x in arm1_eval.get("per_task_first_passed", [])]
    p2_bool = [bool(x) for x in arm2_eval.get("per_task_first_passed", [])]
    mcnemar = (
        mcnemar_exact_p_value(p1_bool, p2_bool) if p1_bool and len(p1_bool) == len(p2_bool) else {}
    )
    ci1 = wilson_ci(sum(1 for x in p1_bool if x), len(p1_bool)) if p1_bool else (0.0, 0.0)
    ci2 = wilson_ci(sum(1 for x in p2_bool if x), len(p2_bool)) if p2_bool else (0.0, 0.0)

    return {
        "paired_tasks": len(shared_ids),
        "arm1_wilson_95_ci": list(ci1),
        "arm2_wilson_95_ci": list(ci2),
        "paired_bootstrap": boot,
        "mcnemar_exact": mcnemar,
    }


def render_ablation_comparison_html(
    unhardened_rows: list[dict[str, Any]],
    hardened_rows: list[dict[str, Any]],
    *,
    arm0_pass_at_1: float | None = None,
    significance: dict[str, Any] | None = None,
) -> str:
    """Render an offline zero-CDN HTML/SVG report comparing Arm 0, Arm 1, and Arm 2."""
    all_steps = [int(r.get("step", 0)) for r in (*unhardened_rows, *hardened_rows)]
    max_step = max(all_steps, default=1)
    svgs = build_ablation_svgs(
        unhardened_rows,
        hardened_rows,
        arm0_pass_at_1=arm0_pass_at_1,
    )

    def _final(rows: list[dict[str, Any]], key: str) -> float:
        for row in reversed(rows):
            if key in row and row[key] is not None:
                return float(row[key])
        return 0.0

    u_tr_final = _final(unhardened_rows, "train_reward")
    u_he_final = _final(unhardened_rows, "clean_heldout_pass_at_1")
    u_dr_final = _final(unhardened_rows, "p95_abs_delta")
    u_tp_final = _final(unhardened_rows, "tamper_rate")
    h_tr_final = _final(hardened_rows, "train_reward")
    h_he_final = _final(hardened_rows, "clean_heldout_pass_at_1")
    h_dr_final = _final(hardened_rows, "p95_abs_delta")
    h_tp_final = _final(hardened_rows, "tamper_rate")
    u_gap = u_tr_final - u_he_final
    h_gap = h_tr_final - h_he_final

    arm0_row_html = ""
    if arm0_pass_at_1 is not None:
        arm0_row_html = (
            f"<tr><td>Arm 0 (Base Model)</td><td>-</td><td>{arm0_pass_at_1:.3f}</td>"
            "<td>-</td><td>0.000</td><td>-</td></tr>"
        )

    sig_html = ""
    if significance and significance.get("paired_bootstrap"):
        boot = significance["paired_bootstrap"]
        mcn = significance.get("mcnemar_exact", {})
        sig_html = (
            '<div class="card"><h2>Statistical Significance (Arm 2 vs. Arm 1)</h2>'
            f"<p>Paired tasks: <b>{significance['paired_tasks']}</b> | "
            f"Bootstrap Delta = <b>{boot['delta_mean']:+.4f}</b> "
            f"95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
            f"(two-sided p = <b>{boot['p_value_two_sided']:.4f}</b>)"
            + (
                f" | McNemar exact p = <b>{mcn['p_value_two_sided']:.4f}</b>"
                if "p_value_two_sided" in mcn
                else ""
            )
            + "</p></div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>dasv LoRA-RLVR Dr. GRPO Ablation Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0b1120; color: #e2e8f0; padding: 24px; }}
  .card {{
    background: #111827; border: 1px solid #1f2937;
    border-radius: 10px; padding: 18px; margin-bottom: 18px;
  }}
  table {{ width: 100%; border-collapse: collapse; font-family: monospace; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #1f2937; }}
</style>
</head>
<body>
  <h1>dasv Causal RLVR Ablation: Unhardened (Arm 1) vs. dasv-Hardened (Arm 2)</h1>
  <div class="card">
    <h2>Final Step Summary (step={html.escape(str(max_step))})</h2>
    <table>
      <tr>
        <th>Arm</th><th>Train Reward</th><th>Clean Held-Out pass@1</th>
        <th>Hacking Divergence Gap</th><th>Tamper Rate</th><th>Final p95 |logprob delta|</th>
      </tr>
      {arm0_row_html}
      <tr>
        <td>Arm 1 (Unhardened)</td><td>{u_tr_final:.3f}</td><td>{u_he_final:.3f}</td>
        <td>{u_gap:+.3f}</td><td>{u_tp_final:.3f}</td><td>{u_dr_final:.4f}</td>
      </tr>
      <tr>
        <td>Arm 2 (dasv-Hardened)</td><td>{h_tr_final:.3f}</td><td>{h_he_final:.3f}</td>
        <td>{h_gap:+.3f}</td><td>{h_tp_final:.3f}</td><td>{h_dr_final:.4f}</td>
      </tr>
    </table>
  </div>
  {sig_html}
  <div class="card">
    <h2>Reward &amp; Held-Out pass@1 Curves</h2>
    {svgs["ablation_reward_and_heldout.svg"]}
  </div>
  <div class="card">
    <h2>Hacking Divergence Gap &amp; Tamper Rate</h2>
    {svgs["ablation_hacking_and_tamper.svg"]}
  </div>
  <div class="card">
    <h2>Train-Inference Logprob Drift (p95 |delta_t| &amp; IcePop Masked Fraction)</h2>
    {svgs["ablation_logprob_drift.svg"]}
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Arm 0/1/2 RLVR ablation metrics & SVGs.")
    parser.add_argument("--arm0", type=Path, default=None, help="Optional Arm 0 manifest.json")
    parser.add_argument("--arm1", type=Path, required=True, help="Arm 1 (unhardened) metrics JSONL")
    parser.add_argument("--arm2", type=Path, required=True, help="Arm 2 (hardened) metrics JSONL")
    parser.add_argument("--arm1-eval", type=Path, default=None, help="Arm 1 heldout_eval.jsonl")
    parser.add_argument("--arm2-eval", type=Path, default=None, help="Arm 2 heldout_eval.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output zero-CDN HTML report path")
    parser.add_argument("--svg-dir", type=Path, default=None, help="Optional standalone SVG dir")
    args = parser.parse_args()

    arm0_p1: float | None = None
    if args.arm0 and args.arm0.is_file():
        m0 = json.loads(args.arm0.read_text(encoding="utf-8"))
        arm0_p1 = float(m0.get("arm0_eval", {}).get("clean_heldout_pass_at_1", 0.0))

    sig: dict[str, Any] | None = None
    if args.arm1_eval and args.arm2_eval and args.arm1_eval.is_file() and args.arm2_eval.is_file():
        e1 = load_arm_metrics(args.arm1_eval)
        e2 = load_arm_metrics(args.arm2_eval)
        if e1 and e2:
            sig = compute_ablation_significance(e1[-1], e2[-1])

    u_rows = load_arm_metrics(args.arm1)
    h_rows = load_arm_metrics(args.arm2)
    html_text = render_ablation_comparison_html(
        u_rows,
        h_rows,
        arm0_pass_at_1=arm0_p1,
        significance=sig,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")

    if args.svg_dir is not None:
        args.svg_dir.mkdir(parents=True, exist_ok=True)
        for fname, content in build_ablation_svgs(
            u_rows, h_rows, arm0_pass_at_1=arm0_p1, standalone=True
        ).items():
            (args.svg_dir / fname).write_text(content, encoding="utf-8")
        if sig is not None:
            (args.svg_dir / "significance.json").write_text(
                json.dumps(sig, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
