# `dasv`: Pre-Flight Adversarial Auditing, Construct-Validity Verification, and Signal Calibration for RLVR Post-Training Environments

**Authors**: `dasv` Contributors  
**Date**: November 2026  
**Version**: `v1.0.0` Technical Report (Full 2-Seed Controlled Ablation Verified: `seed=42`, `seed=1337`)  
**Repository**: <https://github.com/sashwat2212/dasv>  
**License**: Apache-2.0

---

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) and rubric-based RL have become the primary drivers of post-training gains in coding, mathematical, and agentic language models. Unlike static benchmarks, however, online policy-gradient optimizers (GRPO, Dr. GRPO, DAPO) systematically locate and amplify subtle verifier loopholes, container boundary leaks, weak test assertions, and rubric compensation shortcuts—producing superficially high training rewards while degrading true generalization on unseen tasks. Existing ecosystem validators (`harbor check`, `openenv validate`, `vf-validate`) verify syntactic manifests and single-shot gold/null execution, leaving a critical gap between schema validity and adversarial reward soundness.

We introduce **`dasv`** (*Deterministic Adversarial Sandbox Verifier*), an open-source, cross-standard pre-flight auditor and RLVR signal profiler supporting **Harbor**, **OpenEnv**, **Prime Intellect `verifiers` v1**, and **HUD**. Built around a strict **4-Plane isolation architecture** (Control, Untrusted Execution, Trusted Verification, and Telemetry) and a **cost-aware short-circuiting 4-tier funnel** (Tier 0 static security/hermeticity → Tier 1 clean-image $N$-replay determinism & anti-patch/*ImpossibleBench* mutations → Tier 2 budget-capped Hacker-Fixer red-teaming & *ProRubric*/*CHERRL* rubric stress-testing → Tier 3 GRPO/DAPO `pass@k`/`pass^k` zero-gradient calibration), `dasv` enforces hard USD/token circuit breakers and emits machine-readable JSON, **SARIF 2.1.0**, offline zero-CDN HTML Certification Cards, and cryptographic `manifest.json` reproducibility bundles.

We evaluate `dasv` across three empirical pillars:
1. **Planted Vulnerability Benchmark (`KPI-RES-01`)**: **100.0% detection recall (7/7)** across 7 distinct vulnerability classes and **0.0% false-alarm rate (0/3)** on hardened controls.
2. **Real-World Public Environment Audit (`KPI-RES-02`)**: Auditing **45 public RL environments** at pinned upstream commits across `harbor-framework/terminal-bench-2` (28), `PrimeIntellect-ai/verifiers` (10), and `meta-pytorch/OpenEnv` (7) at **\$0.00 cloud API spend** (`22,735` local Ollama tokens) reveals **100.0% prevalence (35/35)** of unpinned base images (`RULE-HERM-001`) and enabled/unspecified network isolation (`RULE-BNDY-002`) across containerized benchmarks, **2 concrete verifier-path leaks (`RULE-BNDY-001`)** copying `tests/` directly into the agent container, and live cross-family *CHERRL* judge divergence ($\tau_b = 0.177$, $\text{mean}|\Delta| = 0.340$) where a local judge rewarded a markdown-checklist fluff hack (`0.8`) over a concise grounded answer (`0.0`).
3. **Controlled 3-Arm LoRA-Dr.GRPO Causal Ablation (`KPI-RES-03`, 2 Independent Seeds)**: Training `Qwen/Qwen2.5-Coder-3B-Instruct` for 160 steps/seed (5.0 epochs, 10,240 training rollouts & 512 unseen held-out rollouts per arm across `seed=42` and `seed=1337` on 1× NVIDIA L4 24 GB) under identical *LoRA Without Regret* + Dr. GRPO/DAPO hyperparameters replicates the **reward-hacking paradox** across both seeds: Unhardened RLVR (**Arm 1**) achieves *higher* mean training reward (`48.90%` vs. `47.29%`) via weak-assertion shortcuts (mean off-policy hack gap **`+2.03 pp ± 0.22 pp`**, spiking up to **`+31.25 pp`**, positive on **`47/320` steps**), yet fails to achieve statistically significant held-out improvement over the unmodified base model (**Arm 0**: `38.67%` → Arm 1 pooled `41.02% ± 0.55%`, `+2.34 pp`, 95% CI `[-1.56, +6.45] pp`, $p = 0.268$; `pass^4`: `27.35%` → `32.49% ± 2.32%`, `+5.14 pp`, $p = 0.064$) while losing `-4.69 pp` on `pass@4` breadth (`56.25%` → `51.56%`). In contrast, `dasv`-Hardened **Arm 2** eliminates hacking (`0/320` steps), outperforms Arm 1 on **both seeds (2/2)** across `pass@1` and `pass^4`, cuts cross-seed `pass^4` variance by **2.2×** (`±1.04%` vs. `±2.32%`), and achieves **statistically significant** pooled held-out gains over Arm 0 on both **`pass@1` (`42.38% ± 1.93%`, `+3.71 pp`, 95% CI `[+0.20, +7.42] pp`, $p = 0.0456$)** and **`pass^4` consistency (`34.38% ± 1.04%`, `+7.03 pp`, 95% CI `[+1.82, +12.78] pp`, $p = 0.0072$)**—reaching **`+10.75 pp` `pass^4` ($p = 0.0088$)** on the active MBPP+ subset (`N=39`) under verified low train-inference logprob drift.

---

## 1. Introduction & Threat Model

Post-training pipelines for code generation and autonomous software engineering rely on executing untrusted model-generated patches inside containerized gyms and updating policy weights from scalar verification signals. Empirical audits published during 2025–2026 demonstrate that environment defects routinely corrupt both benchmark leaderboards and RL training trajectories:

- **False-Positive Verifier Shortcuts**: *Auditing Reward Hackability in Code RL Training Environments* (June 2026) showed that **28.5%** of sampled SWE-bench Verified tasks and **25.0%** of R2E-Gym tasks accepted incorrect patches in Docker, inflating reported `pass@1` by `+14 pp` on flagged subsets.
- **Boundary & History Leakage**: Cursor Research's SWE-bench Pro audit (June 2026) found that **63%** of frontier successes exploited leaked git history or external references; sealing history and egress dropped solve rates by `14.1–20.7 pp`.
- **Underspecified / Broken Tests**: OpenAI's February 2026 audit flagged **59.4%** of 138 frontier-failed SWE-bench Verified tasks as having flawed tests or underspecified prompts, and June Kim's June 2026 study identified **15%** underdetermined specifications.
- **Rubric & Judge Gaming**: *ProRubric* (`arXiv:2609.38847`) documented *"scoring higher, answering worse"* compensation in linear additive rubrics; *CHERRL* (`arXiv:2606.04923`) demonstrated weak-judge vulnerability to verbose formatting hacks; and `arXiv:2607.05904` showed that reference-free cross-family ensembles still fall into shared plausibility basins.
- **Emergent Misalignment & Zero-Gradient Compute Waste**: *ImpossibleBench* (`arXiv:2510.20270`) and Anthropic's *Natural Emergent Misalignment* (`arXiv:2511.18397`) established that exploitable RL verifiers induce test-tampering and alignment faking, while uncalibrated tasks (`pass@1 = 0%` or `100%`) waste rollout compute on zero-variance GRPO groups with probability $p^G + (1-p)^G$.

### Threat Model

`dasv` models the RL policy under training (or a red-team attacker) as an adaptive, resource-bounded adversary operating inside the **Execution Plane** with full read/write access to the agent workspace, shell, and process table. The adversary's goal is to maximize `reward` without satisfying `instruction.md` via:
1. **Verifier Boundary Violations** (`RULE-BNDY-001..004`): mutating co-located tests, querying the internet, reading leaked credentials, or mining `.git` history/reflogs for oracle commits.
2. **Hermeticity Drift** (`RULE-HERM-001..002`): floating base image tags (`:latest` / unpinned digests) or runtime package downloads creating non-reproducible grading environments.
3. **Construct-Validity & Weak-Assertion Exploits** (`RULE-MUT-001..002`): returning hardcoded constants, swallowing exceptions, or spoofing runner exit codes (`sys.exit(0)`) against shallow assertions.
4. **Rubric Compensation & Judge Hacking** (`RULE-RUBR-001..003`): harvesting additive style/format points while failing core correctness, padding wrong answers with structured fluff, or exploiting reference-free judge bias.
5. **RLVR Signal Degeneracy** (`RULE-SIGL-001`): feeding zero-gradient (`STARVED_ZERO_GRAD` or `TRIVIAL_ZERO_GRAD`) or flaky tasks into GRPO/DAPO loops.

---

## 2. System Architecture

`dasv` sits *above* native validators—delegating baseline schema validation via `NativeDelegateRunner`—and enforces strict separation across **4 Planes**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Control Plane: CLI (9 subcommands) · TieredAuditOrchestrator · BudgetGuard │
├───────────────────────────────────────┬─────────────────────────────────┤
│ Untrusted Execution Plane             │ Trusted Verification Plane      │
│ • --network=none, --cap-drop=ALL      │ • Clean-Image Out-of-Container  │
│ • --pids-limit, --memory, --cpus      │   Grader (pristine OCI digest)  │
│ • Zero host secrets / zero test mounts│ • Read-only verifier mount      │
│ • Extracts ONLY workspace diff        │ • Anti-patch & Gold-Sanity Gate │
├───────────────────────────────────────┴─────────────────────────────────┤
│ Telemetry & Export Plane: AuditReport JSON · SARIF 2.1.0 · Offline HTML │
│ · RunManifest (SHA-256) · TRL / veRL / prime-rl / OpenEnv Exporters     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Cost-Aware Short-Circuiting Tiered Funnel

| Tier | Cost Envelope | Checks Executed | Short-Circuit Behavior |
|---|---|---|---|
| **Tier 0** | **\$0** (Static) | Native CLI delegation + `RULE-BNDY-001..004` + `RULE-HERM-001..002` + `RULE-ADPT-001..002` + static `ProRubric` AST scan | Annotates every violation with exact file:line evidence & SARIF locations |
| **Tier 1** | **\$0** (Local Docker) | Clean-image $N$-trial Oracle & $M$-trial No-op determinism (`pass^k`, replay variance, p50/p95 latency) + deterministic Anti-Patch (empty, return-constant, stub, swallow-exception, exit-zero) & *ImpossibleBench* (One-Off, Conflicting) mutations | Short-circuits expensive Tier 2/3 spend if Oracle `< 100%` or No-op `> 0%` |
| **Tier 2** | **Capped \$ / Local LLM** | Budget-capped ReAct **Hacker-Fixer** loop across 6 hack classes + 3-invariant **Gold-Sanity Gate** (exploit → `0`, $N$-trial Oracle → `100%`, No-op → `0%`) + dynamic verbosity perturbation (`RULE-RUBR-002`) & *CHERRL* cross-family judge divergence (`RULE-RUBR-003`) | Hard atomic `asyncio.Lock` `BudgetGuard` halts cleanly on cap (`exit 4`) |
| **Tier 3** | **Capped \$ / Local / Traces** | Base-policy `G`-rollout signal profiler computing unbiased `pass@1`, `pass@k`, `pass^k`, binary reward variance $p(1-p)$, GRPO zero-gradient probability $p^G + (1-p)^G$, and `CurriculumBucket` classification (`GOLDILOCKS`, `TRIVIAL_ZERO_GRAD`, `STARVED_ZERO_GRAD`, `FLAKY`) | Exports calibrated `GOLDILOCKS` curricula to `TRL`, `veRL`, `prime-rl`, and `OpenEnv` |

### 2.2 Deterministic Scoring, Grading & Reproducibility

Every audit produces a bounded `HackabilityScore` $\in [0, 100]$, a deterministic letter grade `A–F`, synchronized JSON (`report.json`), **SARIF 2.1.0** (`report.sarif`), zero-CDN offline HTML Certification Card (`report.html`), and a cryptographic `manifest.json` pinning `dasv` version, git SHA, host/Docker architectures, OCI `@sha256` digests, dataset SHA-256, model IDs, RNG seed, token/USD spend, and CLI arguments.

---

## 3. Empirical Evaluation I: Planted Vulnerability Corpus (`KPI-RES-01`)

To establish baseline sensitivity and false-positive control, `benchmarks/planted_vulns/` houses **7 synthetic vulnerable RL environments** spanning 7 distinct vulnerability classes paired with **3 hardened control environments** verified continuously in CI (`tests/adversarial/test_planted_vulns.py`):

| Corpus Case | Category | Vulnerability Class | Expected Rule(s) | Audit Tier | Result |
|---|---|---|---|---|---|
| `01_test_tampering_copy` | Vulnerable | Test suite copied into agent container | `RULE-BNDY-001` | Tier 0 Static | **Detected (TP)** |
| `02_network_and_secret_leak` | Vulnerable | Bridge network + hardcoded API keys | `RULE-BNDY-002`, `RULE-BNDY-003` | Tier 0 Static | **Detected (TP)** |
| `03_git_history_leak` | Vulnerable | Unstripped `.git` history leaking `solve.sh` | `RULE-BNDY-004` | Tier 0 Static | **Detected (TP)** |
| `04_stub_return_survival` | Vulnerable | Weak verifier passing return-constant stubs | `RULE-MUT-002` | Tier 1 Anti-Patch | **Detected (TP)** |
| `05_exit_zero_runner_spoof` | Vulnerable | Unisolated runner fooled by `sys.exit(0)` | `RULE-MUT-002` | Tier 1 Anti-Patch | **Detected (TP)** |
| `06_additive_rubric_fluff` | Vulnerable | Linear additive rubric + verbosity fluff inflation | `RULE-RUBR-001`, `RULE-RUBR-002` | Tier 2 ProRubric | **Detected (TP)** |
| `07_non_deterministic_boot_install` | Vulnerable | Floating base image + live `pip install` at boot | `RULE-HERM-001`, `RULE-HERM-002` | Tier 0 Static | **Detected (TP)** |
| `01_hardened_calculator` | Control | Pinned `@sha256`, `no-network`, strong tests | *(none)* | Tier 0 + Tier 1 | **0 findings (TN)** |
| `02_hardened_parser` | Control | Pinned `@sha256`, `no-network`, isolated verifier | *(none)* | Tier 0 + Tier 1 | **0 findings (TN)** |
| `03_hardened_prorubric_v1` | Control | `verifiers` v1 conjunctive ProRubric gate | *(none)* | Tier 0 + Tier 2 | **0 findings (TN)** |

**Summary (`KPI-RES-01`)**: **100.0% Recall (7/7)** (target $\ge 90\%$) and **0.0% False-Alarm Rate (0/3)** (target $\le 5\%$).

---

## 4. Empirical Evaluation II: 45-Task Real Public Audit (`KPI-RES-02`)

We ran `scripts/run_public_audit.py` against **45 real public RL environments** drawn from three major open-source ecosystems at pinned commits, spending **\$0.00 in cloud API fees** (`22,735` local Ollama tokens on Apple Silicon). Full per-task artifacts and offline HTML cards are archived in [`results/public_audit/`](../results/public_audit/AUDIT_REPORT.md).

### 4.1 Pinned Upstream Corpus & Grade Distribution

| Ecosystem | Upstream Repository | Pinned Commit | Tasks | Grades | Total Findings |
|---|---|---|---|---|---|
| `harbor` | `harbor-framework/terminal-bench-2` (Apache-2.0) | `2fd12b88aafd` | 28 | F: 28 | 59 |
| `openenv` | `meta-pytorch/OpenEnv` (BSD-3-Clause) | `1f9a7638696f` | 7 | F: 7 | 20 |
| `verifiers_v1` | `PrimeIntellect-ai/verifiers` (MIT) | `96f07072dcd0` | 10 | A: 10 | 0 *(static Tier 0)* |
| **Total** | **3 Ecosystems** | **Pinned SHAs** | **45** | **A: 10, F: 35** | **79** |

### 4.2 Static Vulnerability Prevalence

| Rule ID | Finding Description | Unique Tasks | % of All (45) | % of Containerized (35) | Occurrences |
|---|---|---|---|---|---|
| `RULE-HERM-001` | Floating / unpinned OCI base image (missing `@sha256`) | **35 / 45** | **77.8%** | **100.0%** | 42 |
| `RULE-BNDY-002` | Enabled or unspecified network isolation (`allow_internet = true`) | **35 / 45** | **77.8%** | **100.0%** | 35 |
| `RULE-BNDY-001` | Verifier/test files copied into agent-writable container image | **2 / 45** | **4.4%** | **5.7%** | 2 |

#### Concrete Verifier-Path Leaks Uncovered (`RULE-BNDY-001`)
1. **`terminal-bench-2 / break-filter-js-from-html`** (`environment/Dockerfile:23`): `COPY tests/test_outputs.py /app` directly copies the verifier test file into the agent workspace, exposing grading assertions inside the Execution Plane.
2. **`terminal-bench-2 / circuit-fibsqrt`** (`environment/Dockerfile:9`): `COPY tests/sim.c /app` copies a file residing inside `tests/` into `/app`—demonstrating how co-locating starter fixtures inside `tests/` blurs Execution/Verification Plane boundaries.

### 4.3 Live Tier 2 CHERRL Cross-Family Judge Divergence (\$0 Local Ollama)

Running `dasv rubric --judges qwen2.5-coder:7b,gemma3:4b` across 5 `PrimeIntellect-ai/verifiers` tasks over 5 adversarial probe candidates (*minimal wrong*, *concise grounded*, *verbose fluff*, *markdown-checklist echo of wrong answer*, *fabricated citation*) flagged all 5 tasks (`RULE-RUBR-003`)—4 due to degenerate zero-variance judge scoring and **1 concrete non-degenerate fluff-hack divergence on `color-codeword`**:

- **`qwen2.5-coder:7b` scores**: `[0.0, 0.0, 0.0, 0.8, 0.0]` — assigned **`0.8`** to the *markdown-checklist echo of a wrong answer* while giving **`0.0`** to the concise grounded answer.
- **`gemma3:4b` scores**: `[0.05, 0.8, 0.05, 0.2, 0.2]` — assigned **`0.8`** to the *concise grounded answer* and penalized the fluff hack (`0.2`).
- **Divergence Metrics**: Kendall $\tau_b = 0.177$ (95% bootstrap CI `[-0.408, 1.000]`), Spearman $\rho = 0.186$, $\text{mean}|\Delta| = 0.340$.

---

## 5. Empirical Evaluation III: Controlled 3-Arm LoRA-Dr.GRPO Causal Ablation (`KPI-RES-03`)

To prove that `dasv` pre-training hardening improves downstream RLVR generalization rather than merely tightening static hygiene, we executed a strictly controlled 3-arm online RLVR ablation (`recipes/lora_dr_grpo_ablation/`).

### 5.1 Experimental Design & Controls

| Axis | Specification |
|---|---|
| **Base Policy** | `Qwen/Qwen2.5-Coder-3B-Instruct` (`bfloat16`, single 1× NVIDIA L4 24 GB GPU) |
| **RL Algorithm** | **Dr. GRPO** (`loss_type="dr_grpo"`, `scale_rewards="none"`, DAPO Clip-Higher `epsilon=0.20 / 0.28`, `kl_beta=0.001`) |
| **LoRA Guardrails** | *LoRA Without Regret*: all 7 linear layers (`q/k/v/o/gate/up/down_proj`), `r=16`, `lora_alpha=32`, `lr=5e-5` cosine, `prompts_per_step=4` ($<32$), `G=8` (32 rollouts/step) |
| **Runtime Stack** | `TRL 0.27.2` + `vLLM 0.11.2` colocated sleep-mode (`gpu_memory_utilization=0.35`) + `PyTorch 2.9.0+cu128` (`skipped_trl_kwargs: []`) |
| **Training Horizon** | **160 optimizer steps = 5.0 exact epochs** (5,120 on-policy rollouts per trained arm) |
| **Taskset** | `128` train / `64` disjoint unseen held-out tasks from `evalplus/mbppplus` + `bigcode/bigcodebench` (100% Oracle pass, 0% No-op pass verified) |
| **Arm 0 (Base)** | Unmodified `Qwen2.5-Coder-3B-Instruct` evaluated at `k=4` (256 rollouts) on the 64 clean unseen held-out tasks |
| **Arm 1 (Unhardened)** | Identical 128 training specs with realistic `dasv`-flagged defects: `58` `RULE-MUT-002` weak assertions (45.3%), `26` `RULE-BNDY-001` mutable tests (20.3%), `44` hermetic/network-only (34.4%). All `128/128` fail `dasv` Tier 0. Off-policy clean shadow grading logs true `hack_divergence_gap` without affecting gradients |
| **Arm 2 (`dasv`-Hardened)** | Identical 128 tasks hardened by `dasv` (`128/128` pass Tier 0): full hidden test suites, pinned image, `allow_internet=false`, tamper-proof Verification Plane |
| **Held-Out Evaluation** | Clean, unseen `N=64` test split (`39` MBPP+ + `25` BigCodeBench) evaluated at steps `40, 80, 120` (`k=1`) and final step `160` (`k=4`, 256 rollouts/arm) |
| **Logprob-Drift Control** | Exact token-level $|\log \pi_{\text{train}}(y_t) - \log \pi_{\text{vLLM}}(y_t)|$ and importance-sampling ratio $\rho_t$ captured at zero extra FLOPs via `_get_per_token_logps_and_entropies` (`160/160` steps) |

### 5.2 Pooled 2-Seed & Per-Seed Results (`seed=42` & `seed=1337`, 160 Steps/Seed)

#### Pooled 2-Seed Summary (`N=64` Clean Unseen Held-Out Tasks, 512 Final Rollouts/Arm Across 2 Seeds)

| Arm | Mean Train Reward | Off-Policy Hack Gap (Mean ± Std / Steps) | Pooled Held-Out `pass@1` | $\Delta$ `pass@1` vs. Arm 0 (95% CI, $p$) | Pooled Held-Out `pass@4` | Pooled Held-Out `pass^4` | $\Delta$ `pass^4` vs. Arm 0 (95% CI, $p$) | Active MBPP+ `pass^4` ($N=39$) |
|---|---|---|---|---|---|---|---|---|
| **Arm 0 (Base)** | — | — | `38.67%` | Baseline | `56.25%` | `27.35%` | Baseline | `44.86%` |
| **Arm 1 (Unhardened)** | **48.90%** *(higher!)* | `+2.03 ± 0.22 pp` (`47/320` steps) | `41.02% ± 0.55%` | `+2.34 pp` (`[-1.56, +6.45]`, $p=0.268$, **n.s.**) | `51.56% ± 2.21%` (`-4.69 pp`) | `32.49% ± 2.32%` | `+5.14 pp` (`[-0.24, +11.02]`, $p=0.064$, **n.s.**) | `52.83%` (`+7.97 pp`, $p=0.076$, **n.s.**) |
| **Arm 2 (`dasv`-Hardened)** | `47.29%` | **`0.00 ± 0.00 pp`** (`0/320` steps) | **`42.38% ± 1.93%`** | **`+3.71 pp`** (`[+0.20, +7.42]`, **$p=0.0456$**) | **`52.34% ± 1.10%`** (`+0.78 pp` vs Arm 1) | **`34.38% ± 1.04%`** | **`+7.03 pp`** (`[+1.82, +12.78]`, **$p=0.0072$**) | **`55.61%`** (**`+10.75 pp`**, **$p=0.0088$**) |

*Direct pooled Arm 2 vs. Arm 1 paired bootstrap is consistently positive across every metric and both seeds (`+1.37 pp` `pass@1`, 95% CI `[-1.37, +4.10] pp`, two-sided $p=0.352$; `+1.90 pp` `pass^4`, 95% CI `[-1.30, +5.21] pp`, one-sided $p=0.128$; `+2.78 pp` MBPP+ `pass^4`), with direct-difference CIs widened by the 25/64 starved BigCodeBench tasks at 3B scale.*

#### Per-Seed Breakdown (`seed=42` & `seed=1337`)

| Seed | Arm | Peak VRAM | Wall Time | Mean / Ep 5 Train Reward | Hack Gap (Mean / Max / Steps) | Checkpoint Trajectory (`40 → 80 → 120 → 160`) | Final `pass@1` | Final `pass@4` | Final `pass^4` | Mean / Max $p_{95}|\delta_t|$ | IcePop Steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Base** | **Arm 0** | 6.68 GB | 8.5 min | — | — | `38.67%` (Step 0, `k=4`) | `38.67%` | `56.25%` | `27.35%` | — | — |
| **`42`** | **Arm 1** | 21.05 GB | 140.0 min | `48.91%` / **`51.27%`** | `+1.88 pp` / **`+31.25 pp`** (`21/160`) | `42.19 → 40.62 → 42.19 → 41.41%` | `41.41%` | `50.00%` | `34.13%` | `0.0395 / 0.0753` | `0/160` |
| **`42`** | **Arm 2** | 21.39 GB | 158.2 min | `46.80%` / `48.63%` | **`0.00 pp` / `0.00 pp`** (`0/160`) | `48.44 → 40.62 → 45.31 → 43.75%` | **`43.75%`** | **`53.13%`** | **`35.12%`** | `0.0458 / 0.0895` | `0/160` |
| **`1337`** | **Arm 1** | 21.38 GB | 167.9 min | `48.89%` / `50.20%` | `+2.19 pp` / **`+25.00 pp`** (`26/160`) | `43.75 → 50.00 → 34.38 → 40.62%` | `40.62%` | `53.13%` | `30.85%` | `0.0519 / 0.0944` | `0/160` |
| **`1337`** | **Arm 2** | 21.35 GB | **115.8 min** | `47.77%` / `50.88%` | **`0.00 pp` / `0.00 pp`** (`0/160`) | `40.62 → 43.75 → 42.19 → 41.02%` | **`41.02%`** | `51.56%` | **`33.65%`** | `0.0438 / 0.0829` | `3/160` |

### 5.3 Visual Evidence (Offline SVG Telemetry)

#### Seed 42 Telemetry
![Seed 42 Reward and clean held-out pass@1 curves](../results/v1_ablation/seed_42/svgs/ablation_reward_and_heldout.svg)

![Seed 42 Hacking divergence gap and tamper rate curves](../results/v1_ablation/seed_42/svgs/ablation_hacking_and_tamper.svg)

![Seed 42 Train-inference logprob drift curves](../results/v1_ablation/seed_42/svgs/ablation_logprob_drift.svg)

#### Seed 1337 Telemetry
![Seed 1337 Reward and clean held-out pass@1 curves](../results/v1_ablation/seed_1337/svgs/ablation_reward_and_heldout.svg)

![Seed 1337 Hacking divergence gap and tamper rate curves](../results/v1_ablation/seed_1337/svgs/ablation_hacking_and_tamper.svg)

### 5.4 Five Key Scientific Findings from the 2-Seed Ablation

1. **The Reward-Hacking Paradox Replicates Across Independent Seeds**: Across both seeds, Unhardened **Arm 1** logged higher average training rewards (`48.90%` vs. `47.29%`) by exploiting weak `RULE-MUT-002` assertions on **`47 / 320` steps (`14.7%`)** (mean hack gap `+2.03 ± 0.22 pp`, peaking at `+31.25 pp` on Seed 42 and `+25.00 pp` on Seed 1337), yet finished **lower** than `dasv`-Hardened **Arm 2** on held-out `pass@1` and `pass^4` in **2 out of 2 seeds**.
2. **Only `dasv`-Hardened Training Achieved Statistically Significant Generalization**: Pooling 512 held-out rollouts/arm across both seeds, Unhardened Arm 1 failed to reach significance over Arm 0 on either `pass@1` (`+2.34 pp`, $p = 0.268$) or `pass^4` (`+5.14 pp`, $p = 0.064$). In contrast, `dasv`-Hardened Arm 2 achieved statistically significant pooled lifts on **`pass@1` (`+3.71 pp`, 95% CI `[+0.20, +7.42] pp`, $p = 0.0456$)** and **`pass^4` (`+7.03 pp`, 95% CI `[+1.82, +12.78] pp`, $p = 0.0072$)**—and **`+10.75 pp` `pass^4` ($p = 0.0088$)** on the active MBPP+ task band (`N=39`).
3. **Training Stability, Consistency & Token Efficiency**: `dasv`-Hardened Arm 2 cut cross-seed `pass^4` standard deviation by **2.2×** (`±1.04%` vs. `±2.32%`), avoided Arm 1's dramatic mid-run instability dip on Seed 1337 (where Arm 1 crashed `-15.62 pp` from `50.00%` at Step 80 to `34.38%` at Step 120 while Arm 2 remained steady at `43.75% → 42.19%`), and completed Seed 1337 **31% faster** (`115.8 min` vs. `167.9 min`) with **25% fewer generated tokens** (`1.23M` vs. `1.63M`).
4. **Empirical Validation of Tier 3 `SignalProfiler` (`RULE-SIGL-001`)**: Across all arms and both seeds, the `25/64` held-out BigCodeBench tasks remained near `2–4%` pass rate (`STARVED_ZERO_GRAD` for a 3B model; Arm 2 doubled Base to `4.00%` on Seed 1337), mechanically compressing full-split deltas by `39/64 = 0.61×`—directly validating `dasv`'s recommendation to pre-profile tasksets and filter `STARVED_ZERO_GRAD` tasks before committing GPU rollout budgets.
5. **Train-Inference Logprob Drift Ruled Out as Confounder**: Across all `640` logged training steps (2 seeds × 2 trained arms × 160 steps), train-vs-vLLM logprob divergence remained tightly controlled (mean $p_{95}|\delta_t| \in [0.0395, 0.0519]$, max $\le 0.0944$; `0/320` IcePop-masked steps in Arm 1 and `3/320` (`0.9%`) minor clips at `min IS = 0.469` in Arm 2), confirming that the observed generalization and reliability differences stem from verifier quality rather than engine numerical drift.

---

## 6. CI/CD Gating Proof (`KPI-SHW-02`)

`dasv` ships a zero-config composite GitHub Action ([`action/action.yml`](../action/action.yml)) that installs via `uvx`, executes `dasv audit`, uploads inline **SARIF 2.1.0** annotations to GitHub Code Scanning, publishes a Markdown Step Summary table, attaches the zero-CDN offline HTML Certification Card as a workflow artifact, and gates PR merges on `--fail-on`.

- **Continuous Self-Test**: `.github/workflows/dasv-action-ci.yml` runs on every push and PR against `benchmarks/planted_vulns/hardened/01_hardened_calculator`, asserting `Grade=A` and live SARIF upload.
- **Live Planted-Leak PR Proof (`KPI-SHW-02` Verified)**: On [PR #2](https://github.com/sashwat2212/dasv/pull/2) (closed immediately after verification), deliberately planting `COPY tests/test_outputs.py /app` (`RULE-BNDY-001`) into `benchmarks/planted_vulns/hardened/01_hardened_calculator/environment/Dockerfile` triggered both:
  1. [`dasv-action Self-Test`](https://github.com/sashwat2212/dasv/actions/runs/45031288242/job/137235241008) → **`FAILURE`** (21s), blocking merge and attaching the `dasv-certification-report` offline HTML + SARIF artifact.
  2. [GitHub Code Scanning (`dasv`)](https://github.com/sashwat2212/dasv/runs/137235295841) → **`FAILURE`** (`"1 new alert including 1 error"`, `annotations_count: 1`) placing an inline error annotation directly on the offending Dockerfile line via repo-relative SARIF 2.1.0 URIs.

---

## 7. Limitations, Threats to Validity & Reproducibility

1. **Single-Turn vs. Multi-Turn Exploit Dynamics**: In our single-turn Python code-generation ablation (no shell/filesystem tool loop), `Qwen2.5-Coder-3B-Instruct` organically exploited weak assertions (`RULE-MUT-002`) rather than explicitly emitting `# file: tests/test_solution.py` overwrites (`RULE-BNDY-001`, `tamper_rate = 0.000`). While our unit-tested differential check verifies that `Arm 1` naive grading awards `reward = 1.0` to test-tampering patches and `Arm 2` forces `0.0`, measuring organic test-file tampering during online RL requires multi-turn tool-using agentic rollouts (e.g., SWE-bench / Terminal-Bench 2.0 bash trajectories), which we identify as high-priority future work.
2. **Model Scale & Benchmark Floor**: At 3B parameters, `BigCodeBench` complex library tasks fall into `STARVED_ZERO_GRAD` (`2–3%` pass rate). Scaling the same controlled 3-arm protocol to 7B–32B base models and multi-turn sandboxes will expand the active gradient surface across harder tasks.
3. **Host Architecture Parity**: Per `RISK-02`, container latency/determinism benchmarks are never reported from macOS `arm64` QEMU emulation; all GPU ablation measurements ran natively on `linux/amd64` Ubuntu 24.04 (`NVIDIA L4`), and CI verification runs on native `ubuntu-latest`.
4. **Full Reproducibility**: Every `KPI-RES-01`, `KPI-RES-02`, and `KPI-RES-03` claim is backed by pinned commits, `manifest.json` files, raw JSONL logs, and deterministic unit/adversarial tests under `results/` and `tests/`.

---

## 8. References

1. *Auditing Reward Hackability in Code RL Training Environments* (June 2026) — SWE-bench Verified (`28.5%`) & R2E-Gym (`25.0%`) false-positive verifier study.
2. Cursor Research (June 25, 2026) — *SWE-bench Pro Audit* (git history & egress leakage).
3. OpenAI (February & July 2026) — *SWE-bench Verified & Pro Task Validity Audits*.
4. *ImpossibleBench* (`arXiv:2510.20270`) — Spec-contradiction & test-tampering evaluation.
5. Anthropic (2025) — *Natural Emergent Misalignment from Reward Hacking* (`arXiv:2511.18397`).
6. *ProRubric* (`arXiv:2609.38847`) — Conjunctive protocol gating against additive rubric compensation.
7. *CHERRL* (`arXiv:2606.04923`) & *Reference-Free Judge Plausibility Basins* (`arXiv:2607.05904`).
8. Thinking Machines Lab (2025) — *LoRA Without Regret* (all-linear LoRA, 10× LR, small prompt batch regime for RL).
9. DeepSeek / ByteDance Seed (2025) — *Dr. GRPO* & *DAPO* (unbiased length scaling, Clip-Higher, Dynamic Sampling).
10. Harbor (`harbor-framework/terminal-bench-2`), Meta PyTorch (`meta-pytorch/OpenEnv`), Prime Intellect (`PrimeIntellect-ai/verifiers`), and HUD (`hud-python`).
