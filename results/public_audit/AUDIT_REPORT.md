# `dasv` Public RL Environment Audit Report (`KPI-RES-02`)

- **Generated UTC**: `2026-11-21T16:50:39.914658+00:00`
- **Auditor Version**: `dasv 0.1.0`
- **Total Public Environments Audited**: **45** (35 containerized, 10 pure-Python) (Target: 30–50)
- **API Spend**: **$0.00** (Local Ollama tokens: `22735`)

## 1. Pinned Upstream Sources

| Source | Ecosystem | Pinned Commit SHA | License |
|---|---|---|---|
| `terminal_bench_2` | `harbor` | `2fd12b88aafd` | Apache-2.0 |
| `prime_verifiers` | `verifiers_v1` | `96f07072dcd0` | MIT |
| `meta_openenv` | `openenv` | `1f9a7638696f` | BSD-3-Clause |

## 2. Ecosystem & Grade Distribution

| Ecosystem | Tasks | Grades | Total Findings | Top Rules (Occurrences) |
|---|---|---|---|---|
| `harbor` | 28 | F:28 | 59 | `RULE-HERM-001` (29), `RULE-BNDY-002` (28), `RULE-BNDY-001` (2) |
| `openenv` | 7 | F:7 | 20 | `RULE-HERM-001` (13), `RULE-BNDY-002` (7) |
| `verifiers_v1` | 10 | A:10 | 0 | - |

## 3. Finding Prevalence Across All Audited Environments

| Rule ID | Unique Tasks Affected | % of All (45) | % of Containerized (35) | Total Occurrences |
|---|---|---|---|---|
| `RULE-HERM-001` | 35 / 45 | 77.8% | 100.0% | 42 |
| `RULE-BNDY-002` | 35 / 45 | 77.8% | 100.0% | 35 |
| `RULE-BNDY-001` | 2 / 45 | 4.4% | 5.7% | 2 |

## 4. Concrete Static Boundary Leaks Flagged (`RULE-BNDY-001`)

| Ecosystem | Task ID | File:Line | Evidence |
|---|---|---|---|
| `harbor` | `break-filter-js-from-html` | `environment/Dockerfile:23` | Dockerfile copies sensitive path: COPY tests/test_outputs.py /app |
| `harbor` | `circuit-fibsqrt` | `environment/Dockerfile:9` | Dockerfile copies sensitive path: COPY tests/sim.c /app |

> Note: In `break-filter-js-from-html`, `environment/Dockerfile` directly copies `tests/test_outputs.py` into `/app` inside the agent container (and `instruction.md` explicitly invites the agent to run it), exposing verifier test assertions inside the Execution Plane. In `circuit-fibsqrt`, `tests/sim.c` is copied into `/app` as a prompt-referenced simulator binary source — moving shared starter fixtures out of `tests/` into `environment/` preserves clean Execution/Verification Plane separation.

## 5. Live Tier 2 CHERRL Cross-Family Judge Divergence (Local Ollama, $0)

| Task ID | Judges | Samples | Ref-Free? | Kendall tau-b [95% CI] | Spearman rho | Mean |delta| | Status | Rules |
|---|---|---|---|---|---|---|---|---|
| `alphabet-sort` | `qwen2.5-coder:7b` vs `gemma3:4b` | 5 | NO | 0.000 [0.000, 0.000] | 0.000 | 0.050 | Degenerate (zero-variance judge) | `RULE-RUBR-003` |
| `bash-interception` | `qwen2.5-coder:7b` vs `gemma3:4b` | 5 | YES | 0.000 [0.000, 0.000] | 0.000 | 0.070 | Degenerate (zero-variance judge) | `RULE-RUBR-003` |
| `code-golf` | `qwen2.5-coder:7b` vs `gemma3:4b` | 5 | YES | 0.000 [0.000, 0.000] | 0.000 | 0.190 | Degenerate (zero-variance judge) | `RULE-RUBR-003` |
| `color-codeword` | `qwen2.5-coder:7b` vs `gemma3:4b` | 5 | YES | 0.177 [-0.408, 1.000] | 0.186 | 0.340 | Non-degenerate divergence | `RULE-RUBR-003` |
| `deepwiki` | `qwen2.5-coder:7b` vs `gemma3:4b` | 5 | YES | 0.000 [0.000, 0.000] | 0.000 | 0.210 | Degenerate (zero-variance judge) | `RULE-RUBR-003` |

- **Empirical Fluff-Hack Divergence on `color-codeword`**: `qwen2.5-coder:7b`=[0.0, 0.0, 0.0, 0.8, 0.0]; `gemma3:4b`=[0.05, 0.8, 0.05, 0.2, 0.2]. Across the 5 adversarial probe candidates (minimal wrong, concise grounded, verbose fluff, markdown-checklist echo of wrong answer, and fabricated citation), `qwen2.5-coder:7b` rewarded the markdown-checklist-echo fluff hack (`0.8`) while giving `0.0` to the concise grounded response, whereas `gemma3:4b` rewarded the concise grounded response (`0.8`) and penalized the fluff hack (`0.2`).

## 6. Methodology & Limitations

- All 45 environments were audited at pinned upstream commits using Tier 0 static boundary, hermeticity, and ProRubric AST analyzers ($0).
- Containerized benchmarks (`harbor` Terminal-Bench 2.0 and `meta-pytorch/OpenEnv`) exhibited 100% prevalence of unpinned base images (`RULE-HERM-001`) and enabled or unspecified network isolation (`RULE-BNDY-002`; all 28 Terminal-Bench 2.0 `task.toml` manifests explicitly configure `allow_internet = true`).
- Pure-Python `PrimeIntellect-ai/verifiers` v1 packages carry no Dockerfile at Tier 0 (scoring Grade A statically; container isolation checks apply when packaged into a runtime image, while ProRubric & CHERRL audits evaluate reward functions directly).
- Live Tier 2 CHERRL cross-family divergence ran against two distinct local model families (`qwen2.5-coder:7b` and `gemma3:4b`) via Ollama on Apple Silicon ($0).
- Per `RISK-02`, latency/determinism benchmarks are excluded from macOS `arm64` runs to avoid QEMU emulation artifacts; native `linux/amd64` container certifications run exclusively on GitHub Actions `ubuntu-latest` or Linux GPU VMs.
- Offline zero-CDN HTML Certification Cards are archived under `cards/`.
