# `dasv` — Deterministic Adversarial Sandbox Verifier

Pre-flight adversarial security, construct-validity, and RLVR policy-gradient signal auditor for RL post-training environments.

`dasv` sits *above* native schema linters (`harbor check`, `openenv validate`, `vf-validate`) to catch verifier loopholes, `.git`/secret boundary leaks, flakiness, rubric compensation hacks, and zero-gradient GRPO/DAPO compute waste **before** burning GPU budgets.

---

## Current Status: Stage 5 Walking Skeleton (`v0.1.0-alpha`)

This repository contains the complete locked SDLC Stages 0–4 design corpus (`docs/`) and the verified Stage 5 Python foundation:

- **Core Schemas & Config**: Frozen Pydantic v2 models (`EnvSpec`, `AuditFinding`, `AuditReport`, `RunManifest`, `DeterminismMetrics`, `ExploitTrace`, `SignalProfile`), strict `dasv.toml` loader (`extra="forbid"`), and deterministic exit codes `0–5`.
- **Atomic `BudgetGuard`**: `asyncio.Lock`-protected pre-flight reservation & reconciliation circuit breaker with `$0` local-model support and unknown-cloud-model safety.
- **Cross-Standard Interop Adapters**: Auto-detection & canonical `EnvSpec` parsing for **Harbor**, **OpenEnv**, **Prime Intellect `verifiers` v1** (with static AST introspection & legacy v0 `RULE-ADPT-001` deprecation detection), and **HUD** via `dasv.adapters` entry points, plus `NativeDelegateRunner`.
- **Tier 0 Static Boundary & Hermeticity Probes (`$0`)**: `RULE-BNDY-001..004` and `RULE-HERM-001..002`.
- **RLVR Statistical Estimators & Grading**: Unbiased `pass@k`, `pass^k`, binary reward variance, GRPO/DAPO zero-gradient probability $p^G + (1-p)^G$, `CurriculumBucket`, `HackabilityScore (0–100)`, and deterministic Grade `A–F`.
- **Reporting & CLI**: `report.json`, `manifest.json`, **SARIF 2.1.0**, and working `dasv doctor`, `dasv inspect`, `dasv audit --tiers 0`, and `dasv report` commands.

Tier 1–3 container execution, anti-patch/*ImpossibleBench* mutators, Hacker-Fixer loop, *ProRubric*/*CHERRL* stress tests, offline HTML reports, and the controlled *LoRA Without Regret* Dr. GRPO/DAPO ablation arrive in **Stage 6 (M1–M4)**.

---

## Quickstart

```bash
# Install dependencies (Python >= 3.11 managed automatically by uv)
uv sync

# 1. Check local runtime, host/Docker arch parity, and optional native CLIs
uv run dasv doctor

# 2. Auto-detect and inspect a Harbor, OpenEnv, verifiers v1, or HUD environment
uv run dasv inspect ./path/to/env

# 3. Run $0 Tier 0 static boundary & hermeticity audit with JSON + SARIF 2.1.0 output
uv run dasv audit ./path/to/env \
  --tiers 0 \
  --out-dir ./artifacts/run-01 \
  --sarif ./artifacts/run-01/report.sarif \
  --fail-on high
```

### Development Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

---

## Documentation Index (SDLC Stages 0–4)

| Stage | Document | Description |
|---|---|---|
| Research | [docs/RESEARCH-DOSSIER.md](./docs/RESEARCH-DOSSIER.md) | 2025–2026 RL post-training, rubric RL & reward-hacking dossier |
| Stage 0 | [docs/00-CHARTER.md](./docs/00-CHARTER.md) | Project Charter, scope guardrails & `$0–$50` budget envelope |
| Stage 1 | [docs/01-BRD.md](./docs/01-BRD.md) | Business Requirements (`BRD-REQ-01`–`BRD-REQ-18`) & KPIs |
| Stage 2 | [docs/02-PRD-FEATURES.md](./docs/02-PRD-FEATURES.md) | PRD, `FEAT-01`–`FEAT-22`, frozen CLI & Pydantic v2 SDK contract |
| Stage 3 | [docs/03-ARCHITECTURE.md](./docs/03-ARCHITECTURE.md) | 4-Plane Architecture (`ARCH-COMP-01`–`10`) & `ADR-001`–`007` |
| Stage 4 | [docs/04-DESIGN-SANDBOX-RUNTIME.md](./docs/04-DESIGN-SANDBOX-RUNTIME.md) | `SDD-01`: Orchestrator, `BudgetGuard`, Sandbox Runtime & Probes |
| Stage 4 | [docs/04-DESIGN-VERIFIER-ENGINE.md](./docs/04-DESIGN-VERIFIER-ENGINE.md) | `SDD-02`: Clean-Image Grader, Mutators, Hacker-Fixer & Rubric Engine |
| Stage 4 | [docs/04-DESIGN-INTEROP-ADAPTERS.md](./docs/04-DESIGN-INTEROP-ADAPTERS.md) | `SDD-03`: Interop Adapters & `NativeDelegateRunner` |
| Stage 4 | [docs/04-DESIGN-TELEMETRY-EVAL.md](./docs/04-DESIGN-TELEMETRY-EVAL.md) | `SDD-04`: `pass@k`/`pass^k`, `HackabilityScore`, Grade `A–F` & SARIF/HTML |
| Stage 4 | [docs/04-DESIGN-RL-INTEGRATION.md](./docs/04-DESIGN-RL-INTEGRATION.md) | `SDD-05`: RL Exporters, LoRA-Dr.GRPO Ablation & Logprob-Drift Hook |

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
