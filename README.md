<p align="center">
  <img src="https://raw.githubusercontent.com/sashwat2212/dasv/main/docs/assets/dasv-hero.png" alt="dasv — Deterministic Adversarial Sandbox Verifier" width="100%" />
</p>

# `dasv` — Deterministic Adversarial Sandbox Verifier

[![CI](https://github.com/sashwat2212/dasv/actions/workflows/ci.yml/badge.svg)](https://github.com/sashwat2212/dasv/actions/workflows/ci.yml)
[![dasv-action](https://github.com/sashwat2212/dasv/actions/workflows/dasv-action-ci.yml/badge.svg)](https://github.com/sashwat2212/dasv/actions/workflows/dasv-action-ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dasv?color=blue&label=PyPI)](https://pypi.org/project/dasv/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org/downloads/)
[![GitHub Action](https://img.shields.io/badge/GitHub_Action-dasv%40v1-2088ff.svg?logo=github-actions&logoColor=white)](https://github.com/sashwat2212/dasv/blob/main/action.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-22c55e.svg)](./LICENSE)

**Pre-flight adversarial security, construct-validity, and RLVR policy-gradient signal auditor for RL post-training environments.**

`dasv` sits *above* native schema linters (`harbor check`, `openenv validate`, `vf-validate`) across **Harbor**, **OpenEnv**, **Prime Intellect `verifiers` v1**, and **HUD**. It catches verifier boundary leaks, floating base images, shallow tests, rubric compensation hacks, cross-family judge divergence, and zero-gradient GRPO/DAPO compute waste **before** you burn GPU budgets.

> **Read the full empirical findings in [`docs/TECHNICAL-REPORT.md`](./docs/TECHNICAL-REPORT.md)** and the 45-environment public audit in [`results/public_audit/AUDIT_REPORT.md`](./results/public_audit/AUDIT_REPORT.md).

---

## Why `dasv`? Headline Empirical Results

| Empirical Study | Scope | Key Verified Outcome |
|---|---|---|
| **Planted Vulnerability Corpus** (`KPI-RES-01`) | 7 planted vulnerability classes + 3 hardened controls | **100.0% recall (7/7)**, **0.0% false-alarm rate (0/3)** |
| **Real Public RL Env Audit** (`KPI-RES-02`) | **45 public tasks** across `terminal-bench-2` (28), `verifiers` (10), `OpenEnv` (7) at **\$0.00 API spend** | **100%** of containerized tasks (35/35) unpinned (`RULE-HERM-001`) & internet-enabled (`RULE-BNDY-002`); **2 concrete `COPY tests/` leaks** (`RULE-BNDY-001`); live *CHERRL* fluff-hack judge divergence on `color-codeword` ($\tau_b = 0.177$) |
| **Causal LoRA-Dr.GRPO Ablation** (`KPI-RES-03`) | `Qwen2.5-Coder-3B-Instruct`, 160 steps/seed × **2 seeds (`42`, `1337`)** on 1× NVIDIA L4 (`N=64` unseen tasks, 512 rollouts/arm) | Unhardened **Arm 1** had *higher* mean train reward (`48.90%` vs `47.29%`, hack gap up to `+31.25 pp` on `47/320` steps) yet **non-significant** pooled held-out gains (`pass@1` `+2.34 pp`, $p=0.268$; `pass^4` `+5.14 pp`, $p=0.064$) & `-4.69 pp` `pass@4` drop. `dasv`-Hardened **Arm 2** eliminated hacking (`0/320` steps), beat Arm 1 on **both seeds (2/2)**, cut `pass^4` variance **2.2×**, and achieved **statistically significant** pooled lift: **`+3.71 pp` `pass@1`** ($p=0.0456$), **`+7.03 pp` `pass^4`** ($p=0.0072$), and **`+10.75 pp` MBPP+ `pass^4`** ($p=0.0088$) |

<p align="center">
  <img src="https://raw.githubusercontent.com/sashwat2212/dasv/main/assets/social/linkedin_x_hero_2seed_ablation.png" alt="dasv Controlled 2-Seed LoRA-Dr.GRPO RLVR Ablation Summary" width="100%" />
</p>

<details>
<summary><b>View per-seed telemetry charts (Seed 42 &amp; Seed 1337)</b></summary>

<p align="center">
  <img src="https://raw.githubusercontent.com/sashwat2212/dasv/main/results/v1_ablation/pngs/seed_42_reward_and_heldout.png" alt="Seed 42 Reward and Clean Held-Out pass@1" width="100%" />
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/sashwat2212/dasv/main/results/v1_ablation/pngs/seed_1337_hacking_and_tamper.png" alt="Seed 1337 Hacking Divergence Gap and Tamper Rate" width="100%" />
</p>

</details>

---

## 60-Second Quickstart

Run directly with [`uvx`](https://docs.astral.sh/uv/) (zero setup) or install via `pip`:

```bash
# Run without installing (recommended)
uvx dasv doctor

# Or install globally / into a project
pip install dasv
# Optional: install GPU ablation recipe extras
pip install "dasv[recipes]"
```

Audit any Harbor, OpenEnv, `verifiers` v1, or HUD environment in seconds:

```bash
# 1. Verify Python, Docker/OCI arch parity, and optional native validators
dasv doctor

# 2. Auto-detect environment standard and print canonical EnvSpec
dasv inspect ./benchmarks/planted_vulns/hardened/01_hardened_calculator

# 3. Run $0 Tier 0 static boundary & hermeticity audit with JSON + SARIF + Offline HTML
dasv audit ./benchmarks/planted_vulns/vulnerable/01_test_tampering_copy \
  --tiers 0 \
  --out-dir ./artifacts/run-01 \
  --sarif ./artifacts/run-01/report.sarif \
  --html ./artifacts/run-01/report.html \
  --fail-on high

# 4. Run $0 Tier 1 clean-image Oracle/No-op determinism + anti-patch mutations
dasv audit ./path/to/env --tiers 0,1 --replays 10 --with-mutations

# 5. Run budget-capped Tier 2 adversarial Hacker-Fixer red-team (local Ollama = $0)
dasv redteam ./path/to/env --model qwen2.5-coder:7b --propose-fix --max-cost-usd 2.00

# 6. Stress-test LLM rubrics (ProRubric conjunctive gating + CHERRL cross-family divergence)
dasv rubric ./path/to/rubric_env --judges qwen2.5-coder:7b,gemma3:4b

# 7. Profile GRPO/DAPO zero-gradient waste & export certified GOLDILOCKS curriculum
dasv profile ./path/to/taskset --rollouts 8 --export-curriculum ./curriculum.jsonl
dasv export --format trl --curriculum ./curriculum.jsonl --out ./trl_dataset.jsonl
```

---

## Drop-In GitHub Action (`dasv-action`)

Gate every RL environment PR with inline **SARIF 2.1.0 Code Scanning** annotations and an offline **HTML Certification Card** artifact:

```yaml
name: dasv RL Verifier Gate
on: [pull_request]

permissions:
  contents: read
  actions: read
  security-events: write  # required for inline SARIF Code Scanning alerts

jobs:
  dasv-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: sashwat2212/dasv@v1
        with:
          path: ./envs/my_task
          tiers: "0,1"
          with-mutations: "true"
          fail-on: high
```

*(The subdirectory alias `sashwat2212/dasv/action@v1` also remains supported for backwards compatibility.)*

Outputs exposed to downstream workflow steps: `grade` (`A–F`), `hackability-score` (`0–100`), `sarif-file`, `html-file`, `report-json`, and `exit-code`.

---

## Architecture: 4 Planes & 4 Cost-Aware Tiers

<p align="center">
  <img src="https://raw.githubusercontent.com/sashwat2212/dasv/main/docs/assets/dasv-architecture.png" alt="dasv 4-Plane Isolation and Cost-Aware Verification Funnel" width="100%" />
</p>

`dasv` enforces strict isolation between the **Untrusted Execution Plane** (`--network=none`, `--cap-drop=ALL`, `--pids-limit`, zero host secrets) and the **Trusted Verification Plane** (**Clean-Image Out-of-Container Grading** against a pristine OCI digest):

1. **Tier 0 (\$0 Static)**: Native CLI delegation (`harbor`, `openenv`, `vf-validate`, `hud`) + `RULE-BNDY-001..004` (writable tests, network egress, secrets, `.git` history leaks) + `RULE-HERM-001..002` (unpinned tags, boot-time installs) + static `ProRubric` AST scan.
2. **Tier 1 (\$0 Local Docker)**: Clean-image $N$-trial Oracle & $M$-trial No-op determinism (`pass^k`, replay variance, p50/p95 latency) + deterministic Anti-Patch & *ImpossibleBench* spec-contradiction mutations (`RULE-MUT-001..002`).
3. **Tier 2 (Capped \$ / Local LLM)**: Atomic `BudgetGuard`-protected ReAct **Hacker-Fixer** loop across 6 hack classes + 3-invariant **Gold-Sanity Gate** + *ProRubric* verbosity perturbation & *CHERRL* cross-family judge divergence (`RULE-RUBR-001..003`).
4. **Tier 3 (Capped \$ / Offline Traces)**: Base-policy `pass@1`/`pass@k`/`pass^k` & GRPO zero-gradient probability ($p^G + (1-p)^G$) profiler classifying `GOLDILOCKS`, `TRIVIAL_ZERO_GRAD`, `STARVED_ZERO_GRAD`, and `FLAKY` tasks, with one-command exporters for **`TRL`**, **`veRL`**, **`prime-rl`**, and **`OpenEnv`**.

---

## CLI Command Reference

| Command | Purpose |
|---|---|
| `dasv doctor` | Check Python `>=3.11`, Docker/OCI daemon, host/server arch parity, and native CLIs |
| `dasv inspect <path>` | Auto-detect Harbor / OpenEnv / `verifiers` v1 / HUD environment into canonical `EnvSpec` |
| `dasv audit <path>` | Execute tiered audit pipeline (`0,1,2,3`), emitting JSON, SARIF 2.1.0, HTML & `manifest.json` |
| `dasv mutate <path>` | Run \$0 deterministic Anti-Patch & *ImpossibleBench* mutation suite (`--dry-run` supported) |
| `dasv redteam <path>` | Run budget-capped Tier 2 adversarial Hacker-Fixer loop & 3-invariant Gold-Sanity Gate |
| `dasv rubric <path>` | Run *ProRubric* additive-loophole/fluff checks & *CHERRL* cross-family judge divergence |
| `dasv profile <path>` | Profile RLVR advantage variance & zero-gradient compute waste (`--traces` for \$0 offline mode) |
| `dasv export` | Export certified `GOLDILOCKS` curricula & exploit traces to `trl`, `verl`, `prime_rl`, or `openenv` |
| `dasv report <report.json>` | Re-render validated JSON, SARIF 2.1.0, or offline zero-CDN HTML from an existing report |

### Deterministic Exit Codes

| Exit Code | Meaning |
|---|---|
| `0` | Audit passed (no findings at or above `--fail-on`) |
| `1` | Audit failed (finding(s) met or exceeded `--fail-on` severity threshold) |
| `2` | Configuration or CLI validation error |
| `3` | Runtime or Docker/sandbox execution error |
| `4` | Hard USD or token budget ceiling reached (`BudgetGuard` tripped; partial results saved) |
| `5` | Required native validator missing or failed (`--require-native`) |

---

## Development & Reproducibility

```bash
uv sync --frozen
uv run ruff format --check .
uv run ruff check src tests recipes scripts
uv run pyright
uv run pytest -v
```

Reproduce the **45-task public audit** (`KPI-RES-02`) or **controlled 3-arm LoRA-Dr.GRPO ablation** (`KPI-RES-03`):
- Public audit script & report: [`scripts/run_public_audit.py`](./scripts/run_public_audit.py) → [`results/public_audit/AUDIT_REPORT.md`](./results/public_audit/AUDIT_REPORT.md)
- GPU ablation recipe & CPU dry-run: [`recipes/lora_dr_grpo_ablation/README.md`](./recipes/lora_dr_grpo_ablation/README.md)

---

## Documentation Index

| Document | Description |
|---|---|
| [docs/TECHNICAL-REPORT.md](./docs/TECHNICAL-REPORT.md) | Full arXiv-style empirical technical report (`KPI-RES-01/02/03`) |
| [results/public_audit/AUDIT_REPORT.md](./results/public_audit/AUDIT_REPORT.md) | 45-environment public audit report + offline HTML certification cards |
| [CHANGELOG.md](./CHANGELOG.md) | Release history and semantic versioning notes |
| [docs/RESEARCH-DOSSIER.md](./docs/RESEARCH-DOSSIER.md) | 2025–2026 RLVR post-training, rubric RL & reward-hacking literature review |
| [docs/00-CHARTER.md](./docs/00-CHARTER.md) | Project Charter & scope guardrails |
| [docs/01-BRD.md](./docs/01-BRD.md) | Business Requirements (`BRD-REQ-01`–`18`) & KPI definitions |
| [docs/02-PRD-FEATURES.md](./docs/02-PRD-FEATURES.md) | PRD (`FEAT-01`–`22`) & frozen CLI / Pydantic v2 SDK contract |
| [docs/03-ARCHITECTURE.md](./docs/03-ARCHITECTURE.md) | 4-Plane Architecture & Architecture Decision Records (`ADR-001`–`007`) |
| [docs/04-DESIGN-SANDBOX-RUNTIME.md](./docs/04-DESIGN-SANDBOX-RUNTIME.md) | `SDD-01`: Orchestrator, `BudgetGuard`, Sandbox Runtime & Probes |
| [docs/04-DESIGN-VERIFIER-ENGINE.md](./docs/04-DESIGN-VERIFIER-ENGINE.md) | `SDD-02`: Clean-Image Grader, Mutators, Hacker-Fixer & Rubric Engine |
| [docs/04-DESIGN-INTEROP-ADAPTERS.md](./docs/04-DESIGN-INTEROP-ADAPTERS.md) | `SDD-03`: Interop Adapters & `NativeDelegateRunner` |
| [docs/04-DESIGN-TELEMETRY-EVAL.md](./docs/04-DESIGN-TELEMETRY-EVAL.md) | `SDD-04`: `pass@k`/`pass^k`, `HackabilityScore`, Grade `A–F` & SARIF/HTML |
| [docs/04-DESIGN-RL-INTEGRATION.md](./docs/04-DESIGN-RL-INTEGRATION.md) | `SDD-05`: RL Exporters, LoRA-Dr.GRPO Ablation & Logprob-Drift Hook |

---

## Citation

If you use `dasv`, the 45-environment public audit dataset, or the LoRA-Dr.GRPO ablation recipe in academic or industrial work, cite it via [`CITATION.cff`](./CITATION.cff):

```bibtex
@software{dasv2026,
  title   = {dasv: Deterministic Adversarial Sandbox Verifier},
  author  = {{dasv contributors}},
  year    = {2026},
  version = {1.0.1},
  url     = {https://github.com/sashwat2212/dasv},
  license = {Apache-2.0}
}
```

## Contributing, Security & License

- **Contributing**: See [CONTRIBUTING.md](./CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).
- **Security**: Report vulnerabilities privately per [SECURITY.md](./SECURITY.md).
- **License**: Apache License 2.0 — see [LICENSE](./LICENSE).
