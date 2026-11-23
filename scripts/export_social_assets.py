"""Generate GitHub Social Preview & high-DPI X/LinkedIn PNG charts from 2-seed SVGs."""
# ruff: noqa: E501

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "v1_ablation"
PNG_OUT_DIR = RESULTS_DIR / "pngs"
SOCIAL_DIR = REPO_ROOT / "assets" / "social"

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
)


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if Path(path).is_file():
            return path
    for name in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("No Chromium/Chrome binary found for headless SVG->PNG rendering.")


def render_html_to_png(
    chrome_bin: str,
    html_content: str,
    out_png: Path,
    *,
    width: int,
    height: int,
    scale: int = 2,
) -> None:
    """Render an offline HTML/SVG document to a crisp high-DPI PNG via headless Chrome."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dasv_render_") as tmp:
        html_path = Path(tmp) / "page.html"
        html_path.write_text(html_content, encoding="utf-8")
        cmd = [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            f"--force-device-scale-factor={scale}",
            f"--screenshot={out_png.resolve()}",
            html_path.resolve().as_uri(),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out_png.is_file():
            raise RuntimeError(
                f"Chrome render failed for {out_png.name} (rc={proc.returncode}): {proc.stderr}"
            )


def _wrap_svg_card(svg_markup: str, title: str, subtitle: str, badge: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100vw; height: 100vh; overflow: hidden;
    background: radial-gradient(1200px 700px at 85% 0%, #172554 0%, #0b1120 62%);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
    color: #f8fafc;
    display: flex; flex-direction: column; justify-content: space-between;
    padding: 32px 62px 28px 46px;
  }}
  .top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; }}
  .title {{ font-size: 27px; font-weight: 700; letter-spacing: -0.02em; color: #f8fafc; }}
  .sub {{ font-size: 14.5px; color: #94a3b8; margin-top: 5px; }}
  .badge {{
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: 13px; font-weight: 600; color: #38bdf8;
    background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 7px 14px; border-radius: 999px; white-space: nowrap;
  }}
  .frame {{
    flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center;
    background: #0f172a; border: 1px solid #1e293b; border-radius: 16px;
    padding: 14px 20px; margin: 16px 0 14px 0;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
  }}
  .frame svg {{ width: auto; height: 100%; max-width: 100%; max-height: 410px; display: block; }}
  .footer {{
    display: flex; justify-content: space-between; align-items: center;
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: 12.5px; color: #64748b;
  }}
  .footer strong {{ color: #e2e8f0; }}
</style>
</head>
<body>
  <div class="top">
    <div>
      <div class="title">{title}</div>
      <div class="sub">{subtitle}</div>
    </div>
    <div class="badge">{badge}</div>
  </div>
  <div class="frame">{svg_markup}</div>
  <div class="footer">
    <span><strong>dasv v1.0.0</strong> · Qwen2.5-Coder-3B-Instruct · LoRA-Dr.GRPO · 1× NVIDIA L4</span>
    <span>github.com/sashwat2212/dasv · uvx dasv</span>
  </div>
</body>
</html>
"""


def build_github_social_preview_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 1280px; height: 640px; overflow: hidden;
    background:
      radial-gradient(900px 520px at 88% 12%, rgba(34, 197, 94, 0.14), transparent 60%),
      radial-gradient(860px 520px at 14% 92%, rgba(56, 189, 248, 0.16), transparent 60%),
      #0b1120;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
    color: #f8fafc;
    position: relative;
  }
  .grid-bg {
    position: absolute; inset: 0;
    background-image:
      linear-gradient(rgba(148, 163, 184, 0.05) 1px, transparent 1px),
      linear-gradient(90deg, rgba(148, 163, 184, 0.05) 1px, transparent 1px);
    background-size: 40px 40px;
  }
  .wrap {
    position: relative; z-index: 1;
    width: 1280px; height: 640px;
    padding: 58px 66px;
    display: grid; grid-template-columns: 1.25fr 0.95fr; gap: 46px; align-items: center;
  }
  .brand-row { display: flex; align-items: center; gap: 16px; margin-bottom: 24px; }
  .shield {
    width: 54px; height: 54px; border-radius: 14px;
    background: linear-gradient(135deg, #0ea5e9, #22c55e);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 10px 28px rgba(14, 165, 233, 0.35);
  }
  .brand-name {
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: 42px; font-weight: 800; letter-spacing: -0.03em; color: #ffffff;
  }
  .ver-pill {
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: 15px; font-weight: 700; color: #4ade80;
    background: rgba(34, 197, 94, 0.14); border: 1px solid rgba(34, 197, 94, 0.4);
    padding: 5px 13px; border-radius: 999px;
  }
  h1 {
    font-size: 42px; line-height: 1.12; font-weight: 800; letter-spacing: -0.03em;
    color: #f8fafc; margin-bottom: 18px;
  }
  h1 span { color: #38bdf8; }
  .desc {
    font-size: 19px; line-height: 1.48; color: #cbd5e1; margin-bottom: 26px; max-width: 640px;
  }
  .ecosystems { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 28px; }
  .eco {
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: 13.5px; color: #e2e8f0;
    background: rgba(30, 41, 59, 0.85); border: 1px solid #334155;
    padding: 6px 13px; border-radius: 8px;
  }
  .cmd {
    display: inline-flex; align-items: center; gap: 12px;
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: 17px; color: #f8fafc;
    background: #020617; border: 1px solid #334155; border-radius: 12px;
    padding: 14px 20px; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
  }
  .cmd .prompt { color: #22c55e; font-weight: 700; }
  .cards { display: flex; flex-direction: column; gap: 18px; }
  .card {
    background: rgba(15, 23, 42, 0.92); border: 1px solid #1e293b;
    border-radius: 16px; padding: 20px 24px;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.35);
  }
  .card.green { border-left: 4px solid #22c55e; }
  .card.blue { border-left: 4px solid #38bdf8; }
  .card.amber { border-left: 4px solid #f59e0b; }
  .kicker {
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
    color: #94a3b8; margin-bottom: 6px;
  }
  .stat { font-size: 23.5px; font-weight: 800; letter-spacing: -0.02em; color: #ffffff; }
  .stat .hi-g { color: #4ade80; }
  .stat .hi-b { color: #38bdf8; }
  .stat .hi-a { color: #fbbf24; }
  .detail { font-size: 14.5px; color: #94a3b8; margin-top: 5px; line-height: 1.38; }
</style>
</head>
<body>
  <div class="grid-bg"></div>
  <div class="wrap">
    <div>
      <div class="brand-row">
        <div class="shield">
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#020617" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <path d="m9 12 2 2 4-4"/>
          </svg>
        </div>
        <div class="brand-name">dasv</div>
        <div class="ver-pill">v1.0.0</div>
      </div>
      <h1>Pre-Flight Security &amp; Signal Auditing for <span>RLVR Environments</span></h1>
      <p class="desc">
        Catch verifier leaks, weak tests, rubric fluff hacks, and zero-gradient GRPO/DAPO waste
        before burning GPU budgets.
      </p>
      <div class="ecosystems">
        <span class="eco">Harbor</span>
        <span class="eco">OpenEnv</span>
        <span class="eco">verifiers v1</span>
        <span class="eco">HUD</span>
        <span class="eco">SARIF 2.1.0 CI</span>
      </div>
      <div class="cmd"><span class="prompt">$</span> uvx dasv audit ./env --tiers 0,1 --with-mutations</div>
    </div>
    <div class="cards">
      <div class="card green">
        <div class="kicker">Controlled 2-Seed LoRA-Dr.GRPO Ablation</div>
        <div class="stat"><span class="hi-g">+7.03 pp pass^4</span> (p = 0.0072)</div>
        <div class="detail">0/320 hack steps vs. 47/320 unhardened · +3.71 pp pass@1 (p = 0.0456) · 2.2× lower variance</div>
      </div>
      <div class="card blue">
        <div class="kicker">45 Real Public RL Environments ($0 API Spend)</div>
        <div class="stat"><span class="hi-b">100%</span> Unpinned · <span class="hi-b">2</span> COPY tests/ Leaks</div>
        <div class="detail">Audited Terminal-Bench 2.0, OpenEnv &amp; Prime verifiers + live CHERRL judge divergence (τ = 0.177)</div>
      </div>
      <div class="card amber">
        <div class="kicker">Planted Vulnerability Corpus · Apache-2.0</div>
        <div class="stat"><span class="hi-a">100% Recall</span> · 0% False Alarms</div>
        <div class="detail">4-Plane isolation · Clean-image grading · TRL / veRL / prime-rl / OpenEnv exporters</div>
      </div>
    </div>
  </div>
</body>
</html>
"""


def build_linkedin_2seed_hero_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 100vw; height: 100vh; overflow: hidden;
    background: radial-gradient(1100px 620px at 90% 0%, #172554 0%, #0b1120 65%);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
    color: #f8fafc;
    padding: 32px 62px 28px 46px;
    display: flex; flex-direction: column; justify-content: space-between;
  }
  .header { display: flex; justify-content: space-between; align-items: flex-start; }
  .tag {
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: 12px; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
    color: #38bdf8; margin-bottom: 6px;
  }
  h1 { font-size: 31px; font-weight: 800; letter-spacing: -0.025em; line-height: 1.15; }
  .subtitle { font-size: 15px; color: #94a3b8; margin-top: 6px; }
  .pill {
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: 13px; font-weight: 700; color: #4ade80;
    background: rgba(34, 197, 94, 0.14); border: 1px solid rgba(34, 197, 94, 0.35);
    padding: 8px 14px; border-radius: 999px; white-space: nowrap;
  }
  .main { display: grid; grid-template-columns: 1.38fr 0.92fr; gap: 24px; margin: 18px 0 14px 0; flex: 1; }
  .panel {
    background: rgba(15, 23, 42, 0.92); border: 1px solid #1e293b;
    border-radius: 16px; padding: 22px 24px;
    display: flex; flex-direction: column; justify-content: space-between;
  }
  .panel-title { font-size: 15px; font-weight: 700; color: #e2e8f0; margin-bottom: 14px; display: flex; justify-content: space-between; }
  .legend { display: flex; gap: 14px; font-size: 12px; color: #cbd5e1; font-weight: 500; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
  .metric-row { margin-bottom: 15px; }
  .metric-row:last-child { margin-bottom: 0; }
  .m-head { display: flex; justify-content: space-between; font-size: 13.5px; font-weight: 700; color: #e2e8f0; margin-bottom: 6px; }
  .m-note { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 11.5px; color: #94a3b8; font-weight: 500; }
  .bar-group { display: flex; flex-direction: column; gap: 5px; }
  .bar-line { display: grid; grid-template-columns: 110px 1fr 125px; align-items: center; gap: 10px; }
  .b-lbl { font-size: 12px; color: #94a3b8; font-family: "SF Mono", ui-monospace, Menlo, monospace; }
  .track { height: 16px; background: #1e293b; border-radius: 5px; overflow: hidden; }
  .fill { height: 100%; border-radius: 5px; }
  .f-base { background: #64748b; }
  .f-arm1 { background: linear-gradient(90deg, #f97316, #ef4444); }
  .f-arm2 { background: linear-gradient(90deg, #0ea5e9, #22c55e); }
  .b-val { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 12px; font-weight: 700; text-align: right; }
  .sig { color: #4ade80; }
  .ns { color: #fb923c; }
  .muted { color: #94a3b8; }
  .right-col { display: flex; flex-direction: column; gap: 14px; }
  .kpi-box {
    background: rgba(15, 23, 42, 0.92); border: 1px solid #1e293b; border-radius: 14px; padding: 16px 20px;
  }
  .kpi-box.red { border-left: 4px solid #ef4444; }
  .kpi-box.green { border-left: 4px solid #22c55e; }
  .kpi-box.cyan { border-left: 4px solid #38bdf8; }
  .k-label { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; }
  .k-big { font-size: 24px; font-weight: 800; margin: 4px 0; }
  .k-sub { font-size: 13px; color: #cbd5e1; line-height: 1.38; }
  .seed-tbl { width: 100%; border-collapse: collapse; font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 12px; margin-top: 8px; }
  .seed-tbl th { text-align: left; color: #64748b; padding: 5px 4px; border-bottom: 1px solid #1e293b; }
  .seed-tbl td { padding: 6px 4px; border-bottom: 1px solid rgba(30, 41, 59, 0.7); }
  .footer {
    display: flex; justify-content: space-between; font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: 12.5px; color: #64748b;
  }
  .footer strong { color: #e2e8f0; }
</style>
</head>
<body>
  <div class="header">
    <div>
      <div class="tag">dasv v1.0.0 · Controlled 2-Seed LoRA-Dr.GRPO Ablation (KPI-RES-03)</div>
      <h1>The Reward-Hacking Paradox in RLVR Post-Training</h1>
      <div class="subtitle">Qwen2.5-Coder-3B-Instruct · 160 steps/seed × 2 seeds (42, 1337) · 10,240 train &amp; 512 clean held-out rollouts/arm</div>
    </div>
    <div class="pill">Arm 2 wins 2 / 2 Seeds</div>
  </div>

  <div class="main">
    <div class="panel">
      <div class="panel-title">
        <span>Pooled 2-Seed Performance (N = 64 Unseen Tasks)</span>
        <span class="legend">
          <span><i class="dot" style="background:#64748b"></i>Base</span>
          <span><i class="dot" style="background:#ef4444"></i>Arm 1 Unhardened</span>
          <span><i class="dot" style="background:#22c55e"></i>Arm 2 dasv-Hardened</span>
        </span>
      </div>

      <div class="metric-row">
        <div class="m-head"><span>Mean Training Reward (Misleading Proxy)</span><span class="m-note">Arm 1 looks HIGHER due to shortcuts</span></div>
        <div class="bar-group">
          <div class="bar-line"><span class="b-lbl">Arm 1</span><div class="track"><div class="fill f-arm1" style="width:70%"></div></div><span class="b-val ns">48.90% (+1.61)</span></div>
          <div class="bar-line"><span class="b-lbl">Arm 2 (dasv)</span><div class="track"><div class="fill f-arm2" style="width:67.5%"></div></div><span class="b-val sig">47.29% (0 hacks)</span></div>
        </div>
      </div>

      <div class="metric-row">
        <div class="m-head"><span>Clean Unseen Held-Out pass@1 (N = 64)</span><span class="m-note">Only dasv beats Base significantly</span></div>
        <div class="bar-group">
          <div class="bar-line"><span class="b-lbl">Arm 0 Base</span><div class="track"><div class="fill f-base" style="width:55.2%"></div></div><span class="b-val muted">38.67%</span></div>
          <div class="bar-line"><span class="b-lbl">Arm 1</span><div class="track"><div class="fill f-arm1" style="width:58.6%"></div></div><span class="b-val ns">41.02% (n.s.)</span></div>
          <div class="bar-line"><span class="b-lbl">Arm 2 (dasv)</span><div class="track"><div class="fill f-arm2" style="width:60.5%"></div></div><span class="b-val sig">42.38% p=0.046</span></div>
        </div>
      </div>

      <div class="metric-row">
        <div class="m-head"><span>Clean Unseen pass^4 Consistency (4-for-4 Reliable)</span><span class="m-note">p = 0.0072 · 2.2× lower variance</span></div>
        <div class="bar-group">
          <div class="bar-line"><span class="b-lbl">Arm 0 Base</span><div class="track"><div class="fill f-base" style="width:39.1%"></div></div><span class="b-val muted">27.35%</span></div>
          <div class="bar-line"><span class="b-lbl">Arm 1</span><div class="track"><div class="fill f-arm1" style="width:46.4%"></div></div><span class="b-val ns">32.49% (n.s.)</span></div>
          <div class="bar-line"><span class="b-lbl">Arm 2 (dasv)</span><div class="track"><div class="fill f-arm2" style="width:49.1%"></div></div><span class="b-val sig">34.38% p=0.007</span></div>
        </div>
      </div>

      <div class="metric-row">
        <div class="m-head"><span>Active MBPP+ Subset pass^4 (N = 39 Non-Starved)</span><span class="m-note">+10.75 pp lift · p = 0.0088</span></div>
        <div class="bar-group">
          <div class="bar-line"><span class="b-lbl">Arm 0 Base</span><div class="track"><div class="fill f-base" style="width:64.1%"></div></div><span class="b-val muted">44.86%</span></div>
          <div class="bar-line"><span class="b-lbl">Arm 1</span><div class="track"><div class="fill f-arm1" style="width:75.5%"></div></div><span class="b-val ns">52.83% (n.s.)</span></div>
          <div class="bar-line"><span class="b-lbl">Arm 2 (dasv)</span><div class="track"><div class="fill f-arm2" style="width:79.4%"></div></div><span class="b-val sig">55.61% p=0.009</span></div>
        </div>
      </div>
    </div>

    <div class="right-col">
      <div class="kpi-box red">
        <div class="k-label">Unhardened Reward Hacking (Arm 1)</div>
        <div class="k-big" style="color:#f87171">47 / 320 Steps Hacked</div>
        <div class="k-sub">Off-policy shadow gap averaged <strong>+2.03 pp</strong> (peaking at <strong>+31.25 pp</strong>). Seed 1337 crashed −15.6 pp mid-run (50.0% → 34.4%).</div>
      </div>

      <div class="kpi-box green">
        <div class="k-label">dasv-Hardened Training (Arm 2)</div>
        <div class="k-big" style="color:#4ade80">0 / 320 Hack Steps</div>
        <div class="k-sub">Zero hacking divergence across both seeds · <strong>2.2× lower</strong> cross-seed pass^4 variance · Seed 1337 finished <strong>31% faster</strong> with 25% fewer tokens.</div>
      </div>

      <div class="kpi-box cyan">
        <div class="k-label">Per-Seed Held-Out Replication</div>
        <table class="seed-tbl">
          <tr><th>Seed</th><th>Arm 1 p@1 / p^4</th><th>Arm 2 p@1 / p^4</th></tr>
          <tr><td>42</td><td style="color:#fb923c">41.4% / 34.1%</td><td style="color:#4ade80"><strong>43.8% / 35.1%</strong></td></tr>
          <tr><td>1337</td><td style="color:#fb923c">40.6% / 30.9%</td><td style="color:#4ade80"><strong>41.0% / 33.7%</strong></td></tr>
        </table>
      </div>
    </div>
  </div>

  <div class="footer">
    <span><strong>Engine Control:</strong> mean p95 |logprob Δ| ≤ 0.052 across all 640 steps (no drift confounder)</span>
    <span><strong>pip install dasv</strong> · github.com/sashwat2212/dasv</span>
  </div>
</body>
</html>
"""


def build_linkedin_public_audit_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 100vw; height: 100vh; overflow: hidden;
    background: radial-gradient(1100px 620px at 10% 0%, #1e1b4b 0%, #0b1120 65%);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
    color: #f8fafc;
    padding: 32px 62px 28px 46px;
    display: flex; flex-direction: column; justify-content: space-between;
  }
  .header { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; }
  .tag { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 12px; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: #38bdf8; margin-bottom: 6px; }
  h1 { font-size: 27px; font-weight: 800; letter-spacing: -0.025em; }
  .subtitle { font-size: 14.5px; color: #94a3b8; margin-top: 6px; }
  .pill { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 12.5px; font-weight: 700; color: #38bdf8; background: rgba(56, 189, 248, 0.14); border: 1px solid rgba(56, 189, 248, 0.35); padding: 8px 14px; border-radius: 999px; white-space: nowrap; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin: 18px 0 14px 0; flex: 1; }
  .card { background: rgba(15, 23, 42, 0.92); border: 1px solid #1e293b; border-radius: 16px; padding: 22px 24px; display: flex; flex-direction: column; justify-content: space-between; }
  .card h2 { font-size: 16px; font-weight: 700; color: #e2e8f0; margin-bottom: 12px; }
  .finding { background: #020617; border: 1px solid #1e293b; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; }
  .finding:last-child { margin-bottom: 0; }
  .f-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .rule { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 12px; font-weight: 700; color: #f87171; }
  .pct { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 14px; font-weight: 800; color: #fbbf24; }
  .f-txt { font-size: 13.5px; color: #cbd5e1; line-height: 1.4; }
  code { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 12px; color: #38bdf8; background: rgba(56,189,248,0.1); padding: 2px 6px; border-radius: 4px; }
  .cherrl-bars { display: flex; flex-direction: column; gap: 10px; margin-top: 8px; }
  .c-row { display: grid; grid-template-columns: 155px 1fr 56px; gap: 10px; align-items: center; }
  .c-lbl { font-size: 12.5px; color: #cbd5e1; }
  .c-track { height: 18px; background: #1e293b; border-radius: 6px; overflow: hidden; }
  .c-fill-bad { height: 100%; background: linear-gradient(90deg, #f97316, #ef4444); }
  .c-fill-good { height: 100%; background: linear-gradient(90deg, #0ea5e9, #22c55e); }
  .c-val { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 13px; font-weight: 700; text-align: right; }
  .footer { display: flex; justify-content: space-between; font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 12.5px; color: #64748b; }
  .footer strong { color: #e2e8f0; }
</style>
</head>
<body>
  <div class="header">
    <div>
      <div class="tag">dasv v1.0.0 · Public RL Environment Security Audit (KPI-RES-02)</div>
      <h1>45 Real Open-Source RL Environments Audited at $0.00</h1>
      <div class="subtitle">Pinned commits across Terminal-Bench 2.0 (28 Harbor) · Prime verifiers (10) · Meta OpenEnv (7)</div>
    </div>
    <div class="pill">$0.00 API Spend · 22,735 Local Tokens</div>
  </div>

  <div class="grid">
    <div class="card">
      <div>
        <h2>Tier 0 Static Security &amp; Hermeticity Findings</h2>
        <div class="finding">
          <div class="f-top"><span class="rule">RULE-HERM-001 · Unpinned Base Image</span><span class="pct">35 / 35 (100%)</span></div>
          <div class="f-txt">Every containerized benchmark used floating tags (<code>ubuntu:24.04</code>, <code>python:3.13-slim</code>) with zero <code>@sha256</code> digests.</div>
        </div>
        <div class="finding">
          <div class="f-top"><span class="rule">RULE-BNDY-002 · Open Internet Access</span><span class="pct">35 / 35 (100%)</span></div>
          <div class="f-txt">All 28 Terminal-Bench 2.0 manifests explicitly set <code>allow_internet = true</code>; OpenEnv tasks lacked network isolation.</div>
        </div>
        <div class="finding">
          <div class="f-top"><span class="rule">RULE-BNDY-001 · Verifier Path in Agent Image</span><span class="pct">2 / 35 (5.7%)</span></div>
          <div class="f-txt"><code>break-filter-js-from-html</code> (<code>COPY tests/test_outputs.py /app</code>) &amp; <code>circuit-fibsqrt</code> (<code>COPY tests/sim.c /app</code>).</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div>
        <h2>Live Tier 2 CHERRL Judge Divergence (color-codeword)</h2>
        <p style="font-size:13.5px;color:#94a3b8;line-height:1.45;margin-bottom:14px;">
          Two local model families (<code>qwen2.5-coder:7b</code> vs <code>gemma3:4b</code>) scored 5 adversarial probe responses on the exact same rubric:
        </p>
        <div class="finding">
          <div class="f-top"><span class="rule" style="color:#fb923c">qwen2.5-coder:7b (Fooled by Markdown Fluff!)</span><span class="pct">τ = 0.177</span></div>
          <div class="cherrl-bars">
            <div class="c-row"><span class="c-lbl">Concise Grounded</span><div class="c-track"><div class="c-fill-good" style="width:2%"></div></div><span class="c-val" style="color:#f87171">0.00</span></div>
            <div class="c-row"><span class="c-lbl">Checklist Fluff Hack</span><div class="c-track"><div class="c-fill-bad" style="width:80%"></div></div><span class="c-val" style="color:#f87171">0.80</span></div>
          </div>
        </div>
        <div class="finding">
          <div class="f-top"><span class="rule" style="color:#4ade80">gemma3:4b (Rewarded Grounded Answer)</span><span class="pct" style="color:#38bdf8">mean|Δ| = 0.34</span></div>
          <div class="cherrl-bars">
            <div class="c-row"><span class="c-lbl">Concise Grounded</span><div class="c-track"><div class="c-fill-good" style="width:80%"></div></div><span class="c-val" style="color:#4ade80">0.80</span></div>
            <div class="c-row"><span class="c-lbl">Checklist Fluff Hack</span><div class="c-track"><div class="c-fill-bad" style="width:20%"></div></div><span class="c-val" style="color:#4ade80">0.20</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    <span><strong>Reproducible Dataset:</strong> results/public_audit/AUDIT_REPORT.md + pinned commit SHAs</span>
    <span><strong>uvx dasv rubric</strong> · github.com/sashwat2212/dasv</span>
  </div>
</body>
</html>
"""


def main() -> None:
    chrome = find_chrome()
    print(f"Using headless browser: {chrome}")

    # 1. GitHub Social Preview (exact 1280x640 @ 1x crisp or 2x? GitHub recommends 1280x640 <= 1MB)
    # Render 1280x640 at scale=1 for exact canonical dimensions; also check crispness!
    # Actually scale=2 at 640x320 wouldn't match layout; rendering 1280x640 at scale=2 gives
    # 2560x1280 (<300 KB) which is exact 2:1 aspect ratio and ultra-sharp on Retina screens.
    # Let's render exact 1280x640 for GitHub's documented 1280x640 spec AND verify clarity!
    render_html_to_png(
        chrome,
        build_github_social_preview_html(),
        SOCIAL_DIR / "github-social-preview.png",
        width=1280,
        height=640,
        scale=1,
    )

    # 2. High-DPI 2400x1350 X / LinkedIn Hero Cards (1200x675 @ 2x Retina)
    render_html_to_png(
        chrome,
        build_linkedin_2seed_hero_html(),
        SOCIAL_DIR / "linkedin_x_hero_2seed_ablation.png",
        width=1200,
        height=675,
        scale=2,
    )
    render_html_to_png(
        chrome,
        build_linkedin_public_audit_html(),
        SOCIAL_DIR / "linkedin_x_public_audit_45envs.png",
        width=1200,
        height=675,
        scale=2,
    )

    # 3. Export all 6 2-seed ablation SVG charts as high-DPI 2400x1350 framed PNGs
    chart_meta = {
        "ablation_reward_and_heldout.svg": (
            "Training Reward vs. Clean Held-Out pass@1",
            "Unhardened Arm 1 inflates train reward via shortcuts while lagging on unseen tasks",
        ),
        "ablation_hacking_and_tamper.svg": (
            "Off-Policy Hacking Divergence Gap & Tamper Rate",
            "Real-time hidden clean-verifier shadow gap: Arm 1 exploits vs. Arm 2 zero hacking",
        ),
        "ablation_logprob_drift.svg": (
            "Train-vs-Inference Logprob Drift Control (p95 |delta_t|)",
            "Proving vLLM-vs-PyTorch numerical stability across all 160 RLVR optimizer steps",
        ),
    }

    for seed_name, seed_label in (("seed_42", "Seed 42"), ("seed_1337", "Seed 1337")):
        svg_dir = RESULTS_DIR / seed_name / "svgs"
        for svg_name, (title, subtitle) in chart_meta.items():
            svg_path = svg_dir / svg_name
            if not svg_path.is_file():
                continue
            svg_markup = svg_path.read_text(encoding="utf-8")
            out_name = f"{seed_name}_{svg_name.replace('ablation_', '').replace('.svg', '.png')}"
            html_doc = _wrap_svg_card(
                svg_markup,
                f"{title} ({seed_label})",
                subtitle,
                f"{seed_label} · 160 Steps",
            )
            render_html_to_png(
                chrome,
                html_doc,
                PNG_OUT_DIR / out_name,
                width=1200,
                height=675,
                scale=2,
            )

    print("Generated all social preview & high-DPI 2-seed PNG assets successfully.")


if __name__ == "__main__":
    main()
