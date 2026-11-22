# `dasv`: Pre-Flight Adversarial Auditing, Construct-Validity Verification, and Signal Calibration for RLVR Post-Training Environments

**Authors**: `dasv` Contributors  
**Date**: November 2026  
**Version**: `v1.0.0-rc1` Technical Report (Seed 42 Verified; Seed 1337 Multi-Seed Replication In Progress)  
**Repository**: <https://github.com/sashwat2212/dasv>  
**License**: Apache-2.0

---

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) and rubric-based RL have become the primary drivers of post-training gains in coding, mathematical, and agentic language models. Unlike static benchmarks, however, online policy-gradient optimizers (GRPO, Dr. GRPO, DAPO) systematically locate and amplify subtle verifier loopholes, container boundary leaks, weak test assertions, and rubric compensation shortcuts—producing superficially high training rewards while degrading true generalization on unseen tasks. Existing ecosystem validators (`harbor check`, `openenv validate`, `vf-validate`) verify syntactic manifests and single-shot gold/null execution, leaving a critical gap between schema validity and adversarial reward soundness.

We introduce **`dasv`** (*Deterministic Adversarial Sandbox Verifier*), an open-source, cross-standard pre-flight auditor and RLVR signal profiler supporting **Harbor**, **OpenEnv**, **Prime Intellect `verifiers` v1**, and **HUD**. Built around a strict **4-Plane isolation architecture** (Control, Untrusted Execution, Trusted Verification, and Telemetry) and a **cost-aware short-circuiting 4-tier funnel** (Tier 0 static security/hermeticity → Tier 1 clean-image $N$-replay determinism & anti-patch/*ImpossibleBench* mutations → Tier 2 budget-capped Hacker-Fixer red-teaming & *ProRubric*/*CHERRL* rubric stress-testing → Tier 3 GRPO/DAPO `pass@k`/`pass^k` zero-gradient calibration), `dasv` enforces hard USD/token circuit breakers and emits machine-readable JSON, **SARIF 2.1.0**, offline zero-CDN HTML Certification Cards, and cryptographic `manifest.json` reproducibility bundles.

We evaluate `dasv` across three empirical pillars:
1. **Planted Vulnerability Benchmark (`KPI-RES-01`)**: **100.0% detection recall (7/7)** across 7 distinct vulnerability classes and **0.0% false-alarm rate (0/3)** on hardened controls.
2. **Real-World Public Environment Audit (`KPI-RES-02`)**: Auditing **45 public RL environments** at pinned upstream commits across `harbor-framework/terminal-bench-2` (28), `PrimeIntellect-ai/verifiers` (10), and `meta-pytorch/OpenEnv` (7) at **\$0.00 cloud API spend** (`22,735` local Ollama tokens) reveals **100.0% prevalence (35/35)** of unpinned base images (`RULE-HERM-001`) and enabled/unspecified network isolation (`RULE-BNDY-002`) across containerized benchmarks, **2 concrete verifier-path leaks (`RULE-BNDY-001`)** copying `tests/` directly into the agent container, and live cross-family *CHERRL* judge divergence ($\tau_b = 0.177$, $\text{mean}|\Delta| = 0.340$) where a local judge rewarded a markdown-checklist fluff hack (`0.8`) over a concise grounded answer (`0.0`).
3. **Controlled 3-Arm LoRA-Dr.GRPO Causal Ablation (`KPI-RES-03`)**: Training `Qwen/Qwen2.5-Coder-3B-Instruct` for 160 steps (5.0 epochs, 5,120 rollouts/arm) on 128 MBPP+ / BigCodeBench tasks under identical *LoRA Without Regret* + Dr. GRPO/DAPO hyperparameters (`seed=42`, 1× NVIDIA L4 24 GB) demonstrates the classic **reward-hacking paradox**: Unhardened RLVR (**Arm 1**) achieves *higher* final-epoch training reward (**51.27%**) than `dasv`-Hardened RLVR (**Arm 2**, **48.63%**) via weak-assertion shortcuts (mean off-policy hack gap `+1.88 pp`, peaking at **`+31.25 pp`**), yet yields a **statistically non-significant** held-out gain over the base model (**Arm 0**: `38.67%` → Arm 1: `41.41%`, `+2.73 pp`, $p = 0.249$) while collapsing `pass@4` by **`-6.25 pp`** (`56.25%` → `50.00%`). In contrast, `dasv`-Hardened **Arm 2** eliminates hacking (`0/160` steps), beats or ties Arm 1 at all 4 evaluation checkpoints, and achieves a **statistically significant** held-out lift on both `pass@1` (**`43.75%`**, **`+5.08 pp`**, 95% CI `[+0.78, +9.77] pp`, $p = 0.0240$) and `pass^4` consistency (**`35.12%`**, **`+7.77 pp`**, $p = 0.0202$), doubling Arm 1's gain on the active MBPP+ task band (**`+7.69 pp`**, $p = 0.0242$ vs. `+3.85 pp`, n.s.) under verified low train-inference logprob drift (`0` IcePop-masked steps).

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

### 5.2 Seed 42 Results (Full 160-Step Run)

#### Main Comparison Table (`N=64` Clean Unseen Held-Out Tasks, Final `k=4` = 256 Rollouts/Arm)

| Arm | Peak VRAM | Wall Time | Ep 5 Train Reward | Off-Policy Hack Gap (Mean / Max) | Held-Out `pass@1` | $\Delta$ vs. Arm 0 (95% CI, $p$) | Held-Out `pass@4` | Held-Out `pass^4` | Active MBPP+ `pass@1` ($N=39$) |
|---|---|---|---|---|---|---|---|---|---|
| **Arm 0 (Base)** | 6.68 GB | 8.5 min | — | — | `38.67%` | Baseline | `56.25%` | `27.35%` | `62.18%` |
| **Arm 1 (Unhardened)** | 21.06 GB | 140.0 min | **51.27%** *(higher!)* | `+1.88 pp` / **`+31.25 pp`** (`21/160` steps) | `41.41%` | `+2.73 pp` (`[-1.56, +7.03]`, $p=0.249$, **n.s.**) | `50.00%` (**`-6.25 pp`**) | `34.13%` (`+6.78 pp`) | `66.03%` (`+3.85 pp`, n.s.) |
| **Arm 2 (`dasv`-Hardened)** | 21.40 GB | 158.2 min | `48.63%` | **`0.00 pp` / `0.00 pp`** (`0/160` steps) | **`43.75%`** | **`+5.08 pp`** (`[+0.78, +9.77]`, **$p=0.0240$**) | **`53.13%`** (`+3.13 pp` vs Arm 1) | **`35.12%`** (**`+7.77 pp`**, **$p=0.0202$**) | **`69.87%`** (**`+7.69 pp`**, **$p=0.0242$**) |

#### Checkpoint Trajectory (`clean_heldout_pass_at_1`, $N=64$)

| Checkpoint | Step 0 (Base) | Step 40 (`k=1`) | Step 80 (`k=1`) | Step 120 (`k=1`) | Step 160 (`k=4`) |
|---|---|---|---|---|---|
| **Arm 1 (Unhardened)** | 38.67% | 42.19% | 40.62% | 42.19% | 41.41% |
| **Arm 2 (`dasv`-Hardened)** | 38.67% | **48.44%** | **40.62%** | **45.31%** | **43.75%** |
| **Arm 2 $-$ Arm 1** | `0.00 pp` | **`+6.25 pp`** | `0.00 pp` | **`+3.12 pp`** | **`+2.34 pp`** |

*Arm 2 beat or tied Arm 1 at all 4 held-out evaluation checkpoints. On Seed 42 alone, direct Arm 2 vs. Arm 1 paired bootstrap yields `+2.34 pp` (`95% CI [-0.39, +5.47] pp`, one-sided $p = 0.075$, two-sided $p = 0.150$; on the active MBPP+ subset `+3.85 pp`, one-sided $p = 0.081$).*

### 5.3 Visual Evidence (Seed 42 Offline SVG Telemetry)

![Reward and clean held-out pass@1 curves](../results/v1_ablation/seed_42/svgs/ablation_reward_and_heldout.svg)

![Hacking divergence gap and tamper rate curves](../results/v1_ablation/seed_42/svgs/ablation_hacking_and_tamper.svg)

![Train-inference logprob drift curves](../results/v1_ablation/seed_42/svgs/ablation_logprob_drift.svg)

### 5.4 Four Key Scientific Findings from the Ablation

1. **The Reward-Hacking Paradox**: Arm 1's Epoch 5 training reward (`51.27%`) exceeded Arm 2's (`48.63%`) by `+2.64 pp`, yet its unseen held-out accuracy was `-2.34 pp` *lower* and its `pass@4` solution diversity dropped `-6.25 pp` below the unmodified base model (`56.25% → 50.00%`). Off-policy shadow grading proves why: Arm 1's hacking divergence gap grew across training (`+1.46 pp` in Epoch 2 → `+2.25 pp` in Epoch 5), spiking up to `+31.25 pp` on individual steps as the policy reinforced shallow implementations passing weak `RULE-MUT-002` assertions.
2. **Only `dasv`-Hardened Training Achieved Statistical Significance**: Against the Arm 0 baseline, Arm 1's `+2.73 pp` lift was statistically indistinguishable from zero ($p = 0.249$), whereas Arm 2 achieved statistically significant improvements on both `pass@1` (`+5.08 pp`, $p = 0.0240$) and `pass^4` (`+7.77 pp`, $p = 0.0202$)—and **2×** Arm 1's improvement on active MBPP+ tasks (`+7.69 pp`, $p = 0.0242$ vs. `+3.85 pp`, n.s.).
3. **Empirical Validation of Tier 3 `SignalProfiler` (`RULE-SIGL-001`)**: Across all 3 arms, the `25/64` held-out BigCodeBench tasks remained near `2–3%` pass rate (`STARVED_ZERO_GRAD` for a 3B model), mechanically compressing overall `N=64` deltas by `39/64 = 0.61×` while consuming ~39% of evaluation capacity—directly validating `dasv`'s recommendation to pre-profile tasksets and filter `STARVED_ZERO_GRAD` tasks before RL training.
4. **Train-Inference Logprob Drift Ruled Out as Confounder**: Across all `160/160` steps per arm, train-vs-vLLM logprob divergence remained tightly bounded (Arm 1: mean $p_{95}|\delta_t| = 0.0395$, max `0.0753`, IS $\in [0.724, 1.587]$; Arm 2: mean $p_{95}|\delta_t| = 0.0458$, max `0.0895`, IS $\in [0.734, 1.351]$) with **`0` IcePop-masked steps** in either arm. The generalization gap is attributable to verifier hardening, not engine numerical instability.

### 5.5 Multi-Seed Replication Status (`seed=1337`)

> [!NOTE]
> An independent second seed (`SEED=1337`, identical NVIDIA L4 24 GB VM, 160 steps/arm, reusing the exact `taskset_v1` manifest and Arm 0 baseline) is currently executing. Upon completion, pooled 2-seed `mean ± std` and combined paired-bootstrap $p$-values generated via `plot_ablation.py --seeds-dir results/v1_ablation` will be appended to Section 5 before cutting the final `v1.0.0` release.

---

## 6. CI/CD Gating Proof (`KPI-SHW-02`)

`dasv` ships a zero-config composite GitHub Action ([`action/action.yml`](../action/action.yml)) that installs via `uvx`, executes `dasv audit`, uploads inline **SARIF 2.1.0** annotations to GitHub Code Scanning, publishes a Markdown Step Summary table, attaches the zero-CDN offline HTML Certification Card as a workflow artifact, and gates PR merges on `--fail-on`.

- **Continuous Self-Test**: `.github/workflows/dasv-action-ci.yml` runs on every push and PR against `benchmarks/planted_vulns/hardened/01_hardened_calculator`, asserting `Grade=A` and live SARIF upload.
- **Live Planted-Leak PR Proof**: *(Recorded during Phase 5D verification on an in-repo branch before `v1.0.0` tag).*

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
