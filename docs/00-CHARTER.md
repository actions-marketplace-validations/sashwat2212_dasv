# Stage 0: Project Charter — `dasv`

> [!NOTE]
> **Document Status**: Locked (Stage 0)
> **Companion Dossier**: [RESEARCH-DOSSIER.md](./RESEARCH-DOSSIER.md)
> **Master Roadmap**: `implementation_plan.md` *(Stage 0–4 planning roadmap)*
> **Stage 1 BRD**: [01-BRD.md](./01-BRD.md)
> **Stage 2 PRD**: [02-PRD-FEATURES.md](./02-PRD-FEATURES.md)

---

## 1. Project Identity & Mission

- **Project / Package Name**: **`dasv`** — **D**eterministic **A**dversarial **S**andbox **V**erifier *(PyPI fallback if needed: `dasver`)*
- **Product Category**: Open-source CLI + Python SDK + GitHub Action + Reproducible Empirical RLVR Study.
- **Mission Statement**: Build the pre-flight adversarial security, construct-validity, and RLVR policy-gradient signal auditing tool for RL post-training environments—catching verifier loopholes, reward-hacking vectors, flakiness, and zero-gradient waste *before* frontier labs or researchers burn GPU compute.
- **Positioning Persona**: Authored from the perspective of an **AI / ML Systems & Applied Research Engineer**—combining clean developer-grade CLI engineering (`uv`, Pydantic, async sandboxing, SARIF/HTML reporting) with rigorous post-training science (*ImpossibleBench* mutations, clean-image verifier isolation, *ProRubric*/*CHERRL* rubric stress-testing, `pass@k` vs. `pass^k` estimation, and a controlled *LoRA Without Regret* Dr. GRPO/DAPO ablation).

---

## 2. Locked Stage 0 Parameters

| Parameter | Locked Decision | Engineering / Research Implication |
|---|---|---|
| **Flagship Product** | **Option A**: **`dasv`** (Deterministic Adversarial Sandbox Verifier — Adversarial RL Verifier Hardener & RLVR Signal Profiler CLI) | Layered *above* native validators (`openenv validate`, `harbor check`, `vf-validate`) to own verified 2026 whitespace without rebuilding schema linters. |
| **Core Technical Edge** | **AI / ML Systems & Applied Research** (Python, PyTorch, LLM evals/agents) | Emphasizes RL policy-gradient signal diagnostics (`pass@k` advantage variance, dynamic-sampling waste, reward-hacking causal ablation) alongside clean sandbox engineering. |
| **Target Timeline** | **8-Week Full Researcher Launch** | Weeks 1–2: SDLC Stages 0–4 + hands-on RLVR lab; Weeks 3–5: Stage 5 repo migration + v0.1 MVP & v0.2 Hardening Suite; Weeks 6–8: Empirical LoRA-RLVR ablation, public audit dataset & launch. |
| **Budget Envelope** | **Minimal (\$0–\$50 total)** | **Cost-aware by architecture**: Tier 0 (static/boundary) and Tier 1 (deterministic mutations & `N`-replay) cost **\$0**; Tier 2 (adversarial LLM red-teaming) supports local models (Ollama/MLX) + strict dollar-capped cheap APIs; Tier 3 (LoRA-RLVR ablation) targets 1.5B–3B models on free credits / spot single-GPU runs. |
| **Workspace & Repo Strategy** | **Artifact-First (Stages 0–4) → New Standalone Repo (Stage 5)** | Keep `post-training` untouched during design. At Stage 5, initialize a fresh sibling GitHub repo (e.g., `~/projects/dasv`) and migrate all SDLC docs into `docs/`. |

---

## 3. Target Standards & Interoperability Scope

1. **Primary v0.1–v0.2 Standards (First-Class Adapters)**:
   - **Harbor** (`harbor-framework/harbor` — `task.toml`, `instruction.md`, `environment/Dockerfile`, `solution/`, `tests/`).
   - **OpenEnv** (`meta-pytorch/OpenEnv` — HTTP/WS Gymnasium container contract & `TrainingTrace`).
   - **Prime Intellect `verifiers` v1** (`PrimeIntellect-ai/verifiers` — `Taskset`, `Rubric` / `JudgeRubric` / `RubricGroup`).
2. **Extensible Plugin Adapter (v0.2 / v1.0)**:
   - **HUD** (`hud-evals/hud-python` — MCP/protocol-first tasks & `tasks.grade`) via the same `EnvironmentAdapter` interface.

---

## 4. Explicit Non-Goals (Scope Guardrails)

To prevent scope creep and preserve an 8-week launch window, `dasv` will **explicitly NOT**:
1. **Re-implement native schema validators**: Basic manifest syntax checks or single-shot oracle/no-op execution delegate to or wrap native tools (`harbor check`, `openenv validate`, `vf-validate`).
2. **Build a distributed RL training engine**: `dasv` profiles, hardens, and emits calibration manifests/datasets consumed by existing trainers (`TRL+Unsloth`, `veRL`, `prime-rl`, `SkyRL`, `Tinker`), rather than competing with them.
3. **Operate a hosted multi-tenant SaaS cloud**: All execution is local-first (macOS/Linux Docker) or bring-your-own-runner (GitHub Actions / Modal); hosted infrastructure is deferred to Stage 7 (`docs/05-INFRA-HOSTING.md`).
4. **Target live or unauthorized systems**: Adversarial "Hacker-Fixer" red-teaming runs strictly inside hermetic `--network=none` sandboxes on user-owned or open-license RL environments.

---

## 5. Frugal Budget Allocation Plan (\$0–\$50 Ceiling)

| Spend Bucket | Target Allocation | Execution Strategy |
|---|---|---|
| **Static Boundary Audit, Mutation Testing & `N`-Replay** | **\$0** | 100% deterministic local Python + Docker (OrbStack/Docker Desktop on Mac; `ubuntu-latest` free GitHub Actions minutes for native `linux/amd64` CI). |
| **Adversarial Red-Team & Baseline `pass@k` Sweeps** | **\$10–\$20** | Local small models (Ollama/MLX) for dev/testing; cost-efficient API models (e.g., Gemini Flash / DeepSeek / GPT-4o-mini / Haiku) with hard CLI `--max-cost-usd` circuit breakers for the 30–50 task public audit. |
| **Closed-Loop Single-GPU LoRA-RLVR Ablation** | **\$10–\$25** | 1.5B–3B model (`Qwen2.5` / `Qwen3`) via **Unsloth + TRL** (*LoRA Without Regret*: all-linear, `r=8..16`, 10× LR, batch `< 32`) on Modal free monthly credits / Colab / RunPod spot 1× GPU (~4–10 GPU-hours total). |
| **Contingency Reserve** | **\$5–\$10** | Extra judge-panel cross-family checks or re-running ablation seeds. |
| **Total Hard Ceiling** | **≤ \$50** | Enforced by built-in cost accounting in `dasv` and cloud billing caps. |

---

## 6. SDLC Stage Gate Tracker

| Stage | Artifact | Status |
|---|---|---|
| **Stage 0** | `00_charter.md` (Project Charter) | **LOCKED** |
| **Stage 1** | [01-BRD.md](./01-BRD.md) (Business Requirements Document) | **LOCKED** |
| **Stage 2** | [02-PRD-FEATURES.md](./02-PRD-FEATURES.md) (PRD, Feature Breakdown & CLI/SDK Contract) | **LOCKED** |
| **Stage 3** | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md) + ADRs (High-Level System Architecture) | **LOCKED** |
| **Stage 4** | [SDD-01](./04-DESIGN-SANDBOX-RUNTIME.md) · [SDD-02](./04-DESIGN-VERIFIER-ENGINE.md) · [SDD-03](./04-DESIGN-INTEROP-ADAPTERS.md) · [SDD-04](./04-DESIGN-TELEMETRY-EVAL.md) · [SDD-05](./04-DESIGN-RL-INTEGRATION.md) | **IN REVIEW** |
| **Stage 5** | New Standalone GitHub Repo & CI Scaffold (`~/projects/dasv`) | Pending Stage 4 Approval |
| **Stage 6** | Phased Implementation (M1–M4) & Closed-Loop LoRA-RLVR Study | Pending Stage 5 Exit Gate |
| **Stage 7** | `05_infra_hosting.md` (Scaling, Compute & Hosting Runbook) | Deferred Post-Design |
