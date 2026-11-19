# Stage 2: Product Requirements Document (PRD), Feature Breakdown & CLI/SDK Contract — `dasv`

> [!NOTE]
> **Product**: **`dasv`** — **D**eterministic **A**dversarial **S**andbox **V**erifier
> **Document Status**: Draft for Review (Stage 2 Exit Gate)
> **Upstream References**: [00-CHARTER.md](./00-CHARTER.md) | [01-BRD.md](./01-BRD.md)
> **Next Stage**: Stage 3 — High-Level System Architecture & ADRs (`03_architecture.md`)

---

## 1. Release Cutlines & Delivery Strategy

`dasv` ships across three incremental milestones aligned to the 8-week schedule and **\$0–\$50** budget envelope:

| Release | Target Week | Budget Tier | Scope Summary | Shipped User Value |
|---|---|---|---|---|
| **v0.1 MVP** | Week 3 | **\$0** | Epics 1, 2, 3 (core), 7 (core): `dasv doctor`, `inspect`, `audit --tiers 0,1`, `report`. Harbor + OpenEnv + `verifiers` v1 adapters, native CLI delegation, Verifier Boundary Security Auditor, Clean-Image Grader, Hermeticity/Reset checks, `N`-trial Oracle/No-op/`pass^k` determinism, JSON + SARIF + HTML + `manifest.json`, hard sandbox circuit breakers. | Immediately usable local/CI verifier security & determinism auditor at zero API cost. |
| **v0.2 Research-Grade** | Week 5 | **\$0–\$15** | Epics 3 (mutations), 4, 5 (core), 6 (profiler), 7 (GH Action): `dasv mutate`, `redteam`, `rubric`, `profile`. Deterministic anti-patches & *ImpossibleBench* spec-contradiction mutations, budget-capped Hacker-Fixer loop + Gold-Sanity Gate (Ollama/MLX + cloud APIs), *ProRubric* additive-compensation audit, GRPO/DAPO `pass@k` signal profiler + curriculum export, drop-in GitHub Action. | Full pre-training adversarial hardening & RLVR signal calibration suite. |
| **v1.0 Researcher Showcase** | Weeks 6–8 | **\$15–\$35** | Epics 5 (advanced), 6 (trainers/ablation), HUD plugin: CHERRL bias injection & cross-family judge divergence, `recipes/` reproducible *LoRA Without Regret* Dr. GRPO/DAPO single-GPU Unsloth+TRL ablation + train-inference logprob drift hook, `veRL`/`prime-rl`/`TrainingTrace` export, 30–50 task public audit dataset & arXiv-style report. | Frontier-lab researcher proof-of-work proving causal RLVR post-training impact. |

---

## 2. BRD-to-Feature Traceability Matrix (100% Coverage)

| BRD ID | Priority | Epic | Feature ID(s) | Release |
|---|---|---|---|---|
| `BRD-REQ-01` | P0 | `EPIC-01` | `FEAT-01`, `FEAT-02`, `FEAT-03`, `FEAT-04` | v0.1 (`FEAT-04` HUD in v1.0) |
| `BRD-REQ-02` | P0 | `EPIC-01` | `FEAT-05` | v0.1 |
| `BRD-REQ-03` | P0 | `EPIC-02` | `FEAT-06`, `FEAT-07` | v0.1 |
| `BRD-REQ-04` | P0 | `EPIC-02` | `FEAT-08` | v0.1 |
| `BRD-REQ-05` | P0 | `EPIC-02` | `FEAT-09` | v0.1 |
| `BRD-REQ-06` | P0 | `EPIC-03` | `FEAT-10` | v0.1 |
| `BRD-REQ-07` | P1 | `EPIC-03` | `FEAT-11` | v0.2 |
| `BRD-REQ-08` | P1 | `EPIC-04` | `FEAT-12` | v0.2 |
| `BRD-REQ-09` | P1 | `EPIC-04` | `FEAT-13` | v0.2 |
| `BRD-REQ-10` | P1 | `EPIC-04` | `FEAT-14` | v0.2 |
| `BRD-REQ-11` | P1 | `EPIC-05` | `FEAT-15` | v0.2 |
| `BRD-REQ-12` | P2 | `EPIC-05` | `FEAT-16` | v1.0 |
| `BRD-REQ-13` | P1 | `EPIC-06` | `FEAT-17` | v0.2 |
| `BRD-REQ-14` | P2 | `EPIC-06` | `FEAT-18` | v1.0 |
| `BRD-REQ-15` | P0 | `EPIC-07` | `FEAT-19` | v0.1 |
| `BRD-REQ-16` | P0 | `EPIC-07` | `FEAT-20` | v0.1 |
| `BRD-REQ-17` | P0 | `EPIC-07` | `FEAT-21` | v0.1 |
| `BRD-REQ-18` | P1 | `EPIC-07` | `FEAT-22` | v0.2 |

---

## 3. Detailed Epics, Features, User Stories & Acceptance Criteria

### `EPIC-01`: Cross-Standard Ingestion & Native Delegation
- **`FEAT-01`: Harbor Adapter (`BRD-REQ-01` | v0.1 | \$0)**
  - *User Story*: As an RL environment author using Harbor / Terminal-Bench 2.0, I want `dasv` to parse `task.toml`, `instruction.md`, `environment/Dockerfile`, `solution/`, and `tests/` automatically so I can audit tasks with zero boilerplate.
  - *Acceptance Criteria*:
    - Auto-detects Harbor directories via `task.toml` presence and parses `[metadata]`, `[verifier]`, `[agent]`, and `[environment]` into canonical `EnvSpec`.
    - Extracts Oracle solution path (`solution/solve.sh` or configured entrypoint) and test harness path without modifying the source repo.
- **`FEAT-02`: OpenEnv Adapter (`BRD-REQ-01` | v0.1 | \$0)**
  - *User Story*: As a PyTorch/HF OpenEnv builder, I want `dasv` to inspect containerized HTTP/WS Gymnasium environments (`reset()`, `step()`, `state()`) and ingest `TrainingTrace` schemas natively.
  - *Acceptance Criteria*:
    - Detects OpenEnv manifests/Dockerfiles and validates container boot + health/reset endpoints.
    - Maps OpenEnv actions, observations, rewards, and metadata into `EnvSpec` and `TrainingTrace`.
- **`FEAT-03`: Prime Intellect `verifiers` v1 Adapter (`BRD-REQ-01` | v0.1 | \$0)**
  - *User Story*: As a Prime Intellect Environments Hub contributor, I want `dasv` to introspect `verifiers` v1 `Taskset` packages (`pyproject.toml`, `load_environment()`, `Rubric`/`JudgeRubric`/`RubricGroup`).
  - *Acceptance Criteria*:
    - Loads v1 `Taskset` metadata and enumerates programmatic vs. LLM-judge rubric components.
    - Rejects deprecated legacy v0 packages with an explicit migration warning.
- **`FEAT-04`: Extensible HUD Plugin Adapter (`BRD-REQ-01` | v1.0 | \$0)**
  - *User Story*: As a DataVendor / HUD builder, I want to plug in `hud-python` MCP tasks through Python entry points (`dasv.adapters`) without forking `dasv`.
  - *Acceptance Criteria*:
    - Discovers third-party adapters dynamically via `importlib.metadata.entry_points(group="dasv.adapters")`.
    - Ships a reference `HudAdapter` mapping `tasks.grade` and MCP tool schemas to `EnvSpec`.
- **`FEAT-05`: Native Validator Delegation (`BRD-REQ-02` | v0.1 | \$0)**
  - *User Story*: As a developer who already uses `harbor check`, `openenv validate`, or `vf-validate`, I want `dasv` to run and aggregate native checks first rather than reinventing schema linters.
  - *Acceptance Criteria*:
    - Detects installed native binaries (`harbor`, `openenv`, `vf-validate`) via `shutil.which` and executes them in subprocesses with timeout caps.
    - Normalizes exit status and stdout/stderr into `AuditFinding` records with `source="native_delegate"`.
    - When a native CLI is not installed, runs minimal structural fallback checks and logs an `INFO` advisory rather than crashing.

---

### `EPIC-02`: Verifier Boundary Security & Hermeticity Engine
- **`FEAT-06`: Static Boundary & Leakage Auditor (`BRD-REQ-03` | v0.1 | \$0)**
  - *User Story*: As a post-training researcher, I want static analysis of Dockerfiles, task configs, and workspaces before spinning up containers so obvious leaks are caught in < 1 second.
  - *Acceptance Criteria*:
    - Flags (`CRITICAL`): `COPY tests/` or verifier scripts copied into the agent image; missing `network_mode = "no-network"` / `--network=none`; hardcoded API keys or verifier secrets in `ENV`/`ARG`.
    - Flags (`HIGH`): unpinned `:latest` or floating base images (no `@sha256` digest); live `pip install` / `apt-get` / `npm install` executed inside runtime boot scripts instead of build layers.
    - Flags (`CRITICAL`): `.git` directory containing commits/reflogs/tags newer than task base SHA or containing solution/test history.
- **`FEAT-07`: Runtime Boundary & Container Escape Auditor (`BRD-REQ-03` | v0.1 | \$0)**
  - *User Story*: As an RL infra engineer, I want runtime probes inside the booted agent container to verify what an untrusted agent can actually read, write, or reach over the network.
  - *Acceptance Criteria*:
    - Executes a non-destructive read/write probe inside the Execution Plane verifying test paths are absent or non-writable (`chmod`/mount check), DNS/outbound TCP fails under `--network=none`, and verifier env vars are scrubbed from `/proc/self/environ`.
    - Records concrete path/command evidence per `AuditFinding`.
- **`FEAT-08`: Clean-Image Out-of-Container Grader (`BRD-REQ-04` | v0.1 | \$0)**
  - *User Story*: As a researcher, I want grading to happen strictly in a pristine reference container using only extracted agent artifacts/diffs so sandbox tampering cannot forge `reward == 1`.
  - *Acceptance Criteria*:
    - Extracts agent output strictly via a declared `ArtifactExtractor` (git patch diff, specific output file glob, or DB/state snapshot) into an isolated staging dir.
    - Boots a fresh Verification Plane container from the clean reference image, mounts test files read-only (`:ro`), applies the extracted artifact, and computes reward.
    - Verified by an adversarial unit test where an agent deletes all test files inside the Execution Plane and still receives `reward == 0`.
- **`FEAT-09`: Hermeticity & Cross-Episode Reset Auditor (`BRD-REQ-05` | v0.1 | \$0)**
  - *User Story*: As an RL systems engineer, I want proof that calling `reset()` between rollouts leaves zero state residue (files, SQLite/Postgres rows, background daemons, temp sockets).
  - *Acceptance Criteria*:
    - Runs Episode 1 with deliberate sentinel filesystem/state mutations, invokes `reset()`, and snapshots Episode 2 initial state against a clean baseline hash.
    - Flags any non-empty state diff as `HIGH` (`RULE-HERM-003: Cross-episode state leakage`).

---

### `EPIC-03`: Deterministic Reliability & Spec-Contradiction Mutation Engine
- **`FEAT-10`: `N`-Trial Oracle, No-op & `pass^k` Determinism Runner (`BRD-REQ-06` | v0.1 | \$0)**
  - *User Story*: As an env builder, I want automated multi-trial stress runs (`N=10` default) quantifying Oracle pass rate, No-op pass rate, replay variance, `pass^k`, and p50/p95 latency/memory.
  - *Acceptance Criteria*:
    - Executes `N` isolated trials for Oracle and `M` trials (default 3) for No-op with bounded concurrency (`--max-concurrent`).
    - Computes exact `oracle_pass_rate`, `noop_pass_rate`, reward variance, unbiased `pass@k` estimator, `pass^k = p^k`, p50/p95 wall-clock latency, and peak container RSS.
    - Marks determinism `FAIL` if `oracle_pass_rate < 1.0`, `noop_pass_rate > 0.0`, or `variance > 0.0`.
- **`FEAT-11`: Deterministic Anti-Patch & *ImpossibleBench* Mutation Suite (`BRD-REQ-07` | v0.2 | \$0)**
  - *User Story*: As an AI researcher, I want zero-cost deterministic mutations that prove whether a verifier rejects trivial stubs, exception-swallowing patches, and contradictory specs before spending LLM tokens.
  - *Acceptance Criteria*:
    - **Anti-Patch Library (Code/CLI envs)**: Generates empty diff, pass-through/return-constant stubs, `try/except: pass` error swallowers, and `pytest`/`unittest` exit-0 spoofing patches; measures **Mutation Survival Rate** (% erroneously scoring `reward > 0`).
    - ***ImpossibleBench* Spec-Contradiction Generator**: Synthesizes *One-Off* (single inverted assertion) and *Conflicting* (mutually exclusive spec constraints) test variants to flag over-permissive graders and measure false-positive surfaces.

---

### `EPIC-04`: Adversarial "Hacker-Fixer" Red-Team & Gold-Sanity Loop
- **`FEAT-12`: Budget-Capped Adversarial "Hacker" Agent (`BRD-REQ-08` | v0.2 | Capped \$ / Local)**
  - *User Story*: As a post-training researcher, I want an adversarial LLM agent explicitly prompted to cheat the verifier inside the sandbox so I discover subtle reward-hacking vectors before RL training does.
  - *Acceptance Criteria*:
    - Connects to any OpenAI-compatible endpoint (`--base-url` for local Ollama/MLX/vLLM or cloud APIs) with strict `--max-cost-usd`, `--max-tokens`, and `--max-turns` circuit breakers.
    - Executes targeted exploit strategies (test-file tampering, mock/monkey-patch injection, git-history mining, hardcoded lookup tables, stdout/format spoofing, environment probing).
    - Persists every confirmed exploit (`reward > 0` without passing an independent held-out/anti-cheat check or while violating boundary rules) as a `TrainingTrace`-compatible JSONL artifact and computes **Hackability Score (`0–100`)**.
- **`FEAT-13`: "Fixer" Hardening Patch Generator (`BRD-REQ-09` | v0.2 | Capped \$ / Local)**
  - *User Story*: As an env author, I want concrete unified diffs fixing each discovered exploit (tighter assertions, clean-image test separation, Dockerfile `.git` scrubbing, stricter rubric gates) rather than vague warnings.
  - *Acceptance Criteria*:
    - Given an `ExploitTrace` and source verifier/Dockerfile, generates a scoped unified `.diff` patch explaining the root cause and fix.
    - Never auto-applies patches directly to user files without `--apply` (dry-run `.diff` artifact by default).
- **`FEAT-14`: Automated Gold-Sanity Gate (`BRD-REQ-10` | v0.2 | \$0)**
  - *User Story*: As a researcher, I want mathematical certainty that a Fixer patch actually closes the exploit without breaking valid Oracle solutions.
  - *Acceptance Criteria*:
    - Applies candidate patch to an ephemeral copy of the environment and verifies all three invariants: (1) confirmed exploit replay now scores `reward == 0`, (2) Oracle scores `100%` across `N` runs, (3) No-op scores `0%`.
    - Tags patch `VERIFIED_FIX` only if all three invariants hold; otherwise tags `REJECTED_FIX` with failure reason.

---

### `EPIC-05`: Rubric & LLM-Judge Stress-Testing (Hybrid / Knowledge-Work Envs)
- **`FEAT-15`: *ProRubric* Structural & Verbosity Compensation Audit (`BRD-REQ-11` | v0.2 | \$0 + Capped \$)**
  - *User Story*: As a builder of rubric-graded knowledge-work envs (`verifiers` v1 `RubricGroup`), I want to detect additive rubric compensation loopholes where fluff buys back points for wrong answers.
  - *Acceptance Criteria*:
    - Statically parses rubric definitions to flag pure linear additive sums lacking mandatory conjunctive gating clauses (`RULE-RUBR-001`).
    - Runs synthetic perturbations (length padding, authoritative jargon injection, partial-wrong + verbose-fluff hybrids) to measure score inflation delta and outputs a recommended **ProRubric conjunctive gate spec**.
- **`FEAT-16`: *CHERRL* Bias Injection & Cross-Family Judge Divergence (`BRD-REQ-12` | v1.0 | Capped \$ / Local)**
  - *User Story*: As an AI researcher, I want to quantify whether a training judge diverges from a stronger cross-family reference judge—and be warned if my rubric relies on an unanchored reference-free judge.
  - *Acceptance Criteria*:
    - Evaluates a sampled batch across ≥ 2 distinct judge models (e.g., local judge vs. frontier/alternate-family API judge) and computes rank correlation (Kendall's $\tau$ / Spearman $\rho$) and agreement delta.
    - Explicitly checks for reference anchors or deterministic short-circuit gates; flags purely reference-free judge pipelines as `HIGH` risk (citing plausibility-basin persistence even under multi-judge ensembles).

---

### `EPIC-06`: Pre-Training RLVR Signal Calibration & Trainer Integration
- **`FEAT-17`: `pass@k` / `pass^k` Signal & Zero-Gradient Waste Profiler (`BRD-REQ-13` | v0.2 | Capped \$ / Local)**
  - *User Story*: As an RLVR researcher, I want to estimate `pass@1`, `pass@k`, `pass^k`, within-group advantage variance, and zero-gradient compute waste across a taskset before launching GRPO/DAPO.
  - *Acceptance Criteria*:
    - Runs `G` rollouts per task (default `G=8`) against a target base policy (local MLX/Ollama/vLLM or API).
    - Computes unbiased `pass@1`, `pass@k`, `pass^k`, binary reward variance $p(1-p)$, and `zero_grad_waste_pct` (fraction of tasks with 0% or 100% pass rate).
    - Classifies tasks into `GOLDILOCKS`, `TRIVIAL_ZERO_GRAD`, `STARVED_ZERO_GRAD`, or `FLAKY`, and exports a filtered curriculum JSONL/YAML manifest.
- **`FEAT-18`: Reproducible LoRA-RLVR Ablation Recipes & Trainer Exporters (`BRD-REQ-14` | v1.0 | Spot GPU ≤ \$25)**
  - *User Story*: As an AI researcher showcasing causal impact, I want turnkey configs and scripts that train a 1.5B–3B model with *LoRA Without Regret* (Dr. GRPO / DAPO) on Unhardened vs. `dasv`-Hardened tasksets and log train-inference logprob drift.
  - *Acceptance Criteria*:
    - Exports calibrated tasksets into `TRL`/`Unsloth`, `veRL`, `prime-rl`, and OpenEnv `TrainingTrace` schemas.
    - Ships `recipes/lora_dr_grpo_ablation/` enforcing *LoRA Without Regret* defaults (all-linear target modules, `r=8..16`, 10× LR scaling, effective batch `< 32`) plus a lightweight hook logging $\log \pi_{\text{train}} - \log \pi_{\text{infer}}$ statistics.
    - Produces reproducible plotting scripts comparing reward, response length, KL, entropy, hack rate, and held-out `pass@1`/`pass^k`.

---

### `EPIC-07`: Certification Reporting, Circuit Breakers, Reproducibility & CI/CD
- **`FEAT-19`: Synchronized JSON, SARIF 2.1.0 & Interactive HTML Certification Reports (`BRD-REQ-15` | v0.1 | \$0)**
  - *User Story*: As a team lead or env vendor, I want machine-readable CI annotations (SARIF) and a polished offline HTML Certification Card (`Grade A–F`) I can attach to PRs or lab submissions.
  - *Acceptance Criteria*:
    - Generates `report.json`, `report.sarif` (valid SARIF 2.1.0 schema), and `report.html` (zero external CDN requests; inline SVG/CSS charts for `pass@k`, `pass^k`, latency, and expandable exploit traces).
    - Deterministic grading rubric (`A` = 0 critical/high, Oracle=100%, No-op=0%, 0% variance, Hackability=0 → `F` = critical boundary breach or unpatched exploit).
- **`FEAT-20`: Hard Cost, Time & Container Resource Circuit Breakers (`BRD-REQ-16` | v0.1 | \$0)**
  - *User Story*: As a frugal researcher on a \$0–\$50 budget, I want strict enforcement of spend, token, timeout, and Docker resource caps with guaranteed cleanup on abort/SIGINT.
  - *Acceptance Criteria*:
    - Tracks live prompt/completion tokens and USD cost via an atomic `BudgetGuard`; halts immediately before initiating any LLM call that could breach `--max-cost-usd`, saves partial report, and exits with code `3`.
    - Enforces `--cpus`, `--memory`, `--pids-limit`, `--network=none`, `--cap-drop=ALL`, and registers `atexit`/signal handlers removing all labeled ephemeral containers (`dasv.run_id=<uuid>`).
- **`FEAT-21`: Reproducibility Manifest (`manifest.json`) & `dasv doctor` (`BRD-REQ-17` | v0.1 | \$0)**
  - *User Story*: As a researcher publishing audit data, I want every run pinned to exact software, container digest, dataset hash, model, and seed metadata—plus a pre-flight `dasv doctor` check warning about macOS `arm64` emulation.
  - *Acceptance Criteria*:
    - Writes `manifest.json` on every run with `dasv_version`, `git_sha`, `host_os`, `host_arch`, `oci_image_digests`, `dataset_sha256`, `model_ids`, `rng_seed`, and `cli_args`.
    - `dasv doctor` verifies Docker daemon health, native CLI availability (`harbor`, `openenv`, `vf-validate`), and explicitly warns if `linux/amd64` containers are running under macOS `arm64` QEMU emulation.
- **`FEAT-22`: Drop-In GitHub Action (`dasv-action`) (`BRD-REQ-18` | v0.2 | \$0)**
  - *User Story*: As an open-source environment maintainer, I want a 10-line GitHub Actions workflow that fails PRs introducing verifier leaks or flakiness with inline SARIF annotations.
  - *Acceptance Criteria*:
    - Composite GitHub Action running natively on `ubuntu-latest` (`linux/amd64`), executing `dasv audit --tiers 0,1 --sarif report.sarif`, and uploading SARIF via `github/codeql-action/upload-sarif`.

---

## 4. Frozen CLI Contract (`v0.1` / `v0.2`)

### 4.1 Command Surface
```text
dasv [GLOBAL OPTIONS] <COMMAND> [COMMAND OPTIONS]

Commands:
  doctor     Verify local Docker/OCI runtime, host arch (arm64/amd64 parity), and native CLIs.
  inspect    Auto-detect environment format (Harbor/OpenEnv/verifiers/HUD) and print canonical EnvSpec.
  audit      Run the tiered audit pipeline (Tier 0 Boundary -> Tier 1 Determinism/Mutations -> Tier 2 Red-Team -> Tier 3 Signal).
  mutate     Run Tier 1 deterministic anti-patch & ImpossibleBench spec-contradiction mutations ($0).
  redteam    Run Tier 2 budget-capped adversarial Hacker-Fixer loop + Gold-Sanity Gate.
  rubric     Run ProRubric additive-compensation & CHERRL/cross-family judge stress tests.
  profile    Run Tier 3 pass@k / pass^k & GRPO/DAPO zero-gradient signal calibration + curriculum export.
  report     Re-render JSON/SARIF/HTML Certification Reports from saved run artifacts.
```

### 4.2 Core CLI Examples
```bash
# 1. Pre-flight environment & architecture parity check ($0)
dasv doctor

# 2. Zero-cost CI gate: Boundary Security + Clean-Image Oracle/No-op + N=10 Replay Determinism ($0)
dasv audit ./tasks/my-env \
  --tiers 0,1 \
  --replays 10 \
  --out-dir ./artifacts/run-01 \
  --sarif ./artifacts/run-01/report.sarif \
  --html ./artifacts/run-01/report.html \
  --fail-on high

# 3. Adversarial Hacker-Fixer red-team with hard $1.50 circuit breaker (local Ollama or cloud API)
dasv redteam ./tasks/my-env \
  --model openai/gpt-4o-mini \
  --max-cost-usd 1.50 \
  --max-turns 12 \
  --propose-fix \
  --gold-sanity-replays 5

# 4. Pre-training GRPO/DAPO signal profiler using local Ollama/MLX base policy ($0)
dasv profile ./taskset_dir \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:3b \
  --rollouts 8 \
  --export-curriculum ./artifacts/calibrated_tasks.jsonl
```

### 4.3 Deterministic Exit Code Contract
| Exit Code | Constant | Meaning |
|---|---|---|
| `0` | `EXIT_PASS` | All requested tiers completed and no findings met or exceeded `--fail-on` threshold. |
| `1` | `EXIT_AUDIT_FAIL` | Completed run found certification failures (`CRITICAL`/`HIGH`, Oracle < 100%, No-op > 0%, flakiness, or unpatched exploit). |
| `2` | `EXIT_CONFIG_ERROR` | Invalid CLI flags, malformed `dasv.toml`, or unreadable/unsupported environment layout. |
| `3` | `EXIT_BUDGET_EXCEEDED` | Hard `--max-cost-usd` or `--max-tokens` circuit breaker triggered (partial artifacts preserved). |
| `4` | `EXIT_RUNTIME_ERROR` | Docker/OCI sandbox failure, OOM/timeout unrecoverable error, or host runtime unavailable. |
| `5` | `EXIT_NATIVE_ERROR` | Explicit `--require-native` flag set and underlying `harbor`/`openenv`/`vf-validate` binary missing or crashed. |

### 4.4 Declarative Config (`dasv.toml`)
```toml
[project]
default_tiers = [0, 1]
fail_on = "high"                  # critical | high | medium | low
strict_arch_benchmark = true      # Warn/block publishing latency stats under QEMU emulation

[sandbox]
runtime = "docker"                # docker (v0.1) | modal (Stage 7 plugin)
network_mode = "none"
cpus = 2.0
memory = "2g"
pids_limit = 256
episode_timeout_sec = 120
max_concurrent = 4

[determinism]
oracle_replays = 10
noop_replays = 3

[budget]
max_cost_usd = 2.00
max_tokens = 250000
max_redteam_turns = 12

[llm]
provider = "openai_compat"        # Supports Ollama, MLX server, vLLM, OpenRouter, OpenAI
base_url = "http://localhost:11434/v1"
attacker_model = "qwen2.5-coder:7b"
fixer_model = "qwen2.5-coder:7b"
```

---

## 5. Frozen Python SDK & Core Pydantic v2 Schemas (`v0.1`)

All external inputs/outputs serialize deterministically via **Pydantic v2** (`extra="forbid"` on configs, strict enums, ISO-8601 UTC timestamps).

```python
from __future__ import annotations
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol
from pydantic import BaseModel, Field

class EnvStandard(StrEnum):
    HARBOR = "harbor"
    OPENENV = "openenv"
    VERIFIERS_V1 = "verifiers_v1"
    HUD = "hud"

class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class CurriculumBucket(StrEnum):
    GOLDILOCKS = "goldilocks"              # 5% <= pass@1 <= 40% (or configurable band)
    TRIVIAL_ZERO_GRAD = "trivial_zero_grad" # pass@1 == 100%
    STARVED_ZERO_GRAD = "starved_zero_grad" # pass@k == 0%
    FLAKY = "flaky"                         # Non-zero replay variance

class EnvSpec(BaseModel):
    task_id: str
    standard: EnvStandard
    root_path: Path
    instruction_text: str
    image_ref: str | None = None
    image_digest: str | None = None
    has_oracle: bool = False
    oracle_entrypoint: str | None = None
    verifier_paths: list[str] = Field(default_factory=list)
    rubric_type: Literal["deterministic", "rubric_llm", "hybrid"] = "deterministic"
    metadata: dict[str, Any] = Field(default_factory=dict)

class AuditFinding(BaseModel):
    rule_id: str                           # e.g., "RULE-BNDY-001", "RULE-HERM-003", "RULE-RUBR-001"
    title: str
    severity: Severity
    tier: Literal[0, 1, 2, 3]
    source: Literal["dasv", "native_delegate"]
    file_path: str | None = None
    line_number: int | None = None
    evidence: str
    remediation: str

class DeterminismMetrics(BaseModel):
    oracle_trials: int
    oracle_pass_rate: float                # Must equal 1.0 for PASS
    noop_trials: int
    noop_pass_rate: float                  # Must equal 0.0 for PASS
    replay_variance: float                 # Must equal 0.0 for PASS
    pass_at_1: float
    pass_pow_k: dict[int, float]           # e.g., {3: 1.0, 5: 1.0, 10: 1.0}
    latency_p50_ms: float
    latency_p95_ms: float
    peak_memory_mb: float
    emulated_arch_warning: bool = False

class ExploitTrace(BaseModel):
    exploit_id: str
    task_id: str
    hack_class: str                        # e.g., "test_tampering", "git_leak", "rubric_fluff"
    reward_obtained: float
    steps: list[dict[str, Any]]            # OpenEnv TrainingTrace-compatible turn records
    proposed_fix_diff: str | None = None
    gold_sanity_status: Literal["not_run", "verified_fix", "rejected_fix"] = "not_run"

class SignalProfile(BaseModel):
    task_id: str
    base_model: str
    rollouts_g: int
    pass_at_1: float
    pass_at_k: dict[int, float]
    pass_pow_k: dict[int, float]
    advantage_variance: float              # p * (1 - p) for binary rewards
    zero_grad_risk: bool
    bucket: CurriculumBucket

class RunManifest(BaseModel):
    run_id: str
    dasv_version: str
    timestamp_utc: str
    git_sha: str | None
    host_os: str
    host_arch: str
    docker_server_arch: str | None
    oci_image_digests: dict[str, str]
    dataset_sha256: str
    models_used: list[str]
    rng_seed: int
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    cli_args: list[str]

class AuditReport(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    manifest: RunManifest
    env_spec: EnvSpec
    grade: Literal["A", "B", "C", "D", "F"]
    hackability_score: float | None = None # 0.0 (hardened) to 100.0 (trivially hacked)
    findings: list[AuditFinding] = Field(default_factory=list)
    determinism: DeterminismMetrics | None = None
    mutation_survival_rate: float | None = None
    exploits: list[ExploitTrace] = Field(default_factory=list)
    signal_profile: SignalProfile | None = None

class EnvironmentAdapter(Protocol):
    standard: EnvStandard
    def detect(self, path: Path) -> bool: ...
    def load(self, path: Path) -> EnvSpec: ...
    async def run_native_check(self, spec: EnvSpec) -> list[AuditFinding]: ...
    async def run_oracle(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]: ...
    async def run_noop(self, spec: EnvSpec, sandbox: Any) -> tuple[float, dict[str, Any]]: ...
```

---

## 6. Non-Functional Requirements (NFRs)

| Category | Requirement | Verification |
|---|---|---|
| **Security & Isolation** | Zero host home directory mounts; mandatory `--network=none`, `--cap-drop=ALL`, `--pids-limit`, `--memory`, `--cpus`, and isolated clean-image Verification Plane. | Adversarial container-escape & test-deletion test suite (`tests/adversarial/`). |
| **Resource Hygiene** | 0 orphaned containers or dangling temp directories after normal exit, timeout, budget abort, or `SIGINT`/`SIGTERM`. | Chaos test sending `SIGINT` mid-audit and asserting `docker ps -q --filter label=dasv.run_id=<id>` is empty. |
| **Performance** | Tier 0 static audit completes in **< 1.0 s** per task; Tier 1 `N=10` replays scale cleanly via `asyncio.Semaphore(max_concurrent)`. | Benchmark test on reference Harbor task. |
| **Cross-Arch Parity** | Full native dev workflow on macOS Apple Silicon (`arm64`); automatic warning badge (`emulated_arch_warning=True`) when running `linux/amd64` images under Rosetta/QEMU; official latency/determinism certification verified on native Linux (`ubuntu-latest`). | `dasv doctor` + GitHub Actions CI workflow. |
| **Determinism & Auditability** | Every report artifact embeds its `RunManifest` (`manifest.json`); SARIF validates strictly against SARIF 2.1.0 JSON schema; HTML works 100% offline. | Schema validation unit tests. |

---

## 7. Stage 2 Exit Gate (Approval Required for Stage 3)

> [!IMPORTANT]
> **To advance to Stage 3 (`03_architecture.md` + ADRs — High-Level System Architecture & Trust Boundaries)**, please review and confirm:
> 1. Do the **7 Epics (`EPIC-01`–`EPIC-07`)**, **22 Features (`FEAT-01`–`FEAT-22`)**, and **v0.1 / v0.2 / v1.0 release cutlines** feel tightly scoped and realistic for our 8-week schedule?
> 2. Does the **frozen `dasv` CLI surface** (`doctor`, `inspect`, `audit`, `mutate`, `redteam`, `rubric`, `profile`, `report`) and **Pydantic SDK contract** cover everything you want to showcase?
> 3. Any CLI commands, schema fields, or feature cutlines you want tweaked before we lock Stage 2 and design the full High-Level System Architecture & ADRs in Stage 3?
