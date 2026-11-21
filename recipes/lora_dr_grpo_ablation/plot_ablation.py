"""Zero-dependency SVG & JSON ablation plotter comparing Unhardened vs. dasv-Hardened RLVR arms."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, cast


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
        step = int(row.get("step", 0))
        val = float(row.get(key, 0.0))
        x = 48.0 + (step / max(1, max_step)) * 420.0
        y = 180.0 - (min(max_val, max(0.0, val)) / max(1e-9, max_val)) * 140.0
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def render_ablation_comparison_html(
    unhardened_rows: list[dict[str, Any]],
    hardened_rows: list[dict[str, Any]],
) -> str:
    """Render an offline zero-CDN HTML/SVG report comparing Arm 1 vs. Arm 2 trajectories."""
    all_steps = [int(r.get("step", 0)) for r in (*unhardened_rows, *hardened_rows)]
    max_step = max(all_steps, default=1)

    u_train = _polyline(unhardened_rows, "train_reward", max_step, 1.0)
    u_held = _polyline(unhardened_rows, "clean_heldout_pass_at_1", max_step, 1.0)
    h_train = _polyline(hardened_rows, "train_reward", max_step, 1.0)
    h_held = _polyline(hardened_rows, "clean_heldout_pass_at_1", max_step, 1.0)
    u_drift = _polyline(unhardened_rows, "p95_abs_delta", max_step, 1.0)
    h_drift = _polyline(hardened_rows, "p95_abs_delta", max_step, 1.0)

    def _final(rows: list[dict[str, Any]], key: str) -> float:
        return float(rows[-1].get(key, 0.0)) if rows else 0.0

    u_tr_final = _final(unhardened_rows, "train_reward")
    u_he_final = _final(unhardened_rows, "clean_heldout_pass_at_1")
    u_dr_final = _final(unhardened_rows, "p95_abs_delta")
    h_tr_final = _final(hardened_rows, "train_reward")
    h_he_final = _final(hardened_rows, "clean_heldout_pass_at_1")
    h_dr_final = _final(hardened_rows, "p95_abs_delta")
    u_gap = u_tr_final - u_he_final
    h_gap = h_tr_final - h_he_final

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
        <th>Hacking Divergence Gap</th><th>Final p95 |logprob delta|</th>
      </tr>
      <tr>
        <td>Arm 1 (Unhardened)</td><td>{u_tr_final:.3f}</td><td>{u_he_final:.3f}</td>
        <td>{u_gap:+.3f}</td><td>{u_dr_final:.4f}</td>
      </tr>
      <tr>
        <td>Arm 2 (dasv-Hardened)</td><td>{h_tr_final:.3f}</td><td>{h_he_final:.3f}</td>
        <td>{h_gap:+.3f}</td><td>{h_dr_final:.4f}</td>
      </tr>
    </table>
  </div>
  <div class="card">
    <h2>Reward &amp; Held-Out pass@1 Curves</h2>
    <svg width="520" height="210" viewBox="0 0 520 210" role="img"
         aria-label="Reward and held-out pass@1 curves">
      <line x1="48" y1="180" x2="468" y2="180" stroke="#334155" />
      <line x1="48" y1="40" x2="48" y2="180" stroke="#334155" />
      <polyline fill="none" stroke="#f97316" stroke-width="2"
                stroke-dasharray="5,4" points="{u_train}" />
      <polyline fill="none" stroke="#ef4444" stroke-width="2" points="{u_held}" />
      <polyline fill="none" stroke="#38bdf8" stroke-width="2"
                stroke-dasharray="5,4" points="{h_train}" />
      <polyline fill="none" stroke="#22c55e" stroke-width="2.5" points="{h_held}" />
      <text x="54" y="22" fill="#94a3b8" font-size="11" font-family="monospace">
        Orange=Arm1 Train | Red=Arm1 HeldOut | Cyan=Arm2 Train | Green=Arm2 HeldOut
      </text>
    </svg>
  </div>
  <div class="card">
    <h2>Train-Inference Logprob Drift (p95 |delta_t|)</h2>
    <svg width="520" height="210" viewBox="0 0 520 210" role="img"
         aria-label="Train-inference logprob drift curves">
      <line x1="48" y1="180" x2="468" y2="180" stroke="#334155" />
      <line x1="48" y1="40" x2="48" y2="180" stroke="#334155" />
      <polyline fill="none" stroke="#ef4444" stroke-width="2" points="{u_drift}" />
      <polyline fill="none" stroke="#22c55e" stroke-width="2" points="{h_drift}" />
      <text x="54" y="22" fill="#94a3b8" font-size="11" font-family="monospace">
        Red=Arm1 p95 |delta_t| | Green=Arm2 p95 |delta_t|
      </text>
    </svg>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Arm 1 vs Arm 2 RLVR ablation metrics.")
    parser.add_argument("--arm1", type=Path, required=True, help="Arm 1 (unhardened) metrics JSONL")
    parser.add_argument("--arm2", type=Path, required=True, help="Arm 2 (hardened) metrics JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Output zero-CDN HTML report path")
    args = parser.parse_args()
    html_text = render_ablation_comparison_html(
        load_arm_metrics(args.arm1),
        load_arm_metrics(args.arm2),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
