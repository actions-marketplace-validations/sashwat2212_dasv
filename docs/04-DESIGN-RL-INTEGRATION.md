# Stage 4 / SDD-05: RL Trainer Integration, Causal Ablation & Logprob-Drift Hook — `dasv`

> [!NOTE]
> **Subsystem**: `SDD-05` (`ARCH-COMP-10` RL Trainer Exporters, Ablation Recipes & Train-Inference Drift Diagnostics)
> **Linked Features**: `FEAT-17`, `FEAT-18`
> **Upstream Docs**: [02-PRD-FEATURES.md](./02-PRD-FEATURES.md) | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Package Modules**: `src/dasv/exporters/`, `recipes/lora_dr_grpo_ablation/`
> **Document Status**: Draft for Review (**Stage 4 Exit Gate** — covers `SDD-01` through `SDD-05`)

---

## 1. Downstream RL Trainer Exporters (`src/dasv/exporters/`)

After `dasv audit` and `dasv profile` certify and bucket a taskset, `ARCH-COMP-10` exports strictly **Grade A/B, `GOLDILOCKS`-calibrated** subsets directly into standard 2026 RL post-training schemas:

| Exporter Module | Target Ecosystem | Output Artifact Format & Key Fields |
|---|---|---|
| **`trl_export.py`** | **TRL (`GRPOTrainer` / `AsyncGRPOTrainer`) + Unsloth** | Hugging Face Dataset JSONL (`prompt`, `task_id`, `verifier_ref`, `dasv_grade`, `baseline_pass_at_1`) + Python `reward_func` wrapper delegating directly to `CleanImageGrader`. |
| **`verl_export.py`** | **veRL (HybridFlow)** | Parquet / JSONL matching veRL `data_source`, `prompt`, `ability`, `reward_model` (`style="rule"`, `ground_truth`), and `extra_info` (`dasv_manifest_sha`). |
| **`prime_rl_export.py`** | **Prime Intellect `prime-rl` + `verifiers` v1** | Filtered `Taskset` split manifest + `eval_config.toml` / `rl_config.toml` snippet pointing only to certified `GOLDILOCKS` task IDs. |
| **`trace_export.py`** | **OpenEnv `TrainingTrace`** | Framework-neutral JSONL storing `episode_id`, `task_id`, `token_ids`, `sampled_logprobs`, `loss_mask` (0 for env observations, 1 for policy actions), `reward`, and `tamper_flags`. |

---

## 2. Controlled Closed-Loop LoRA-RLVR Ablation Design (`recipes/lora_dr_grpo_ablation/`)

This experiment satisfies **`KPI-RES-03`** and establishes **AI Researcher proof-of-work**—proving *causally* that `dasv` hardening prevents reward hacking and yields true held-out capability gains, strictly within **≤ \$25 cloud GPU spend**.

### 2.1 Experimental Matrix (3 Arms, Same Compute & Seed)

| Arm | Training Environment Set | Verifier Mode | Expected Outcome (Hypothesis) |
|---|---|---|---|
| **Arm 0 (Baseline)** | Pre-RL base model (`Qwen2.5-Coder-1.5B` or `3B-Instruct`, zero RL steps) | Evaluated on clean held-out suite | Reference `pass@1` and `pass^k`. |
| **Arm 1 (Unhardened RL)** | Unfiltered taskset containing realistic verifier loopholes (in-sandbox writable tests / weak stub-passing assertions / uncalibrated tasks) | Naive in-container verifier | High apparent training reward + **high reward-hacking divergence gap** (`train_reward >> clean_heldout_pass@1`) + flat or degraded held-out `pass^k`. |
| **Arm 2 (`dasv`-Hardened RL)** | Same base tasks after `dasv audit` + `dasv redteam --propose-fix` (clean-image grading + `VERIFIED_FIX` patches + `GOLDILOCKS` filtering) | `CleanImageGrader` + hardened verifiers | Lower early training reward (no free cheats), near-zero hacking divergence gap, and **statistically higher clean held-out `pass@1` and `pass^k`**. |

### 2.2 *LoRA Without Regret* + Dr. GRPO / DAPO Hyperparameter Protocol
Strictly follows Thinking Machines Lab's *LoRA Without Regret* findings and *Dr. GRPO* (`arXiv:2503.20783`) / *DAPO* (`arXiv:2503.14476`) best practices:

```yaml
# recipes/lora_dr_grpo_ablation/config.yaml
model:
  base_model: "Qwen/Qwen2.5-Coder-3B-Instruct"  # Fallback: 1.5B for < 16GB VRAM GPUs
  dtype: "bfloat16"
  engine: "unsloth + trl"

lora:
  # LoRA Without Regret Rule 1: ALL linear layers (attention + MLP), NEVER attention-only
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  r: 16                                         # Rule 3: Ultra-low rank sufficient for ~1-bit RLVR
  lora_alpha: 32
  lora_dropout: 0.0

optimizer:
  # LoRA Without Regret Rule 2: ~10x FullFT LR
  learning_rate: 5.0e-5
  scheduler: "cosine"
  warmup_ratio: 0.05
  # LoRA Without Regret Rule 4 (CRITICAL): Keep effective prompt batch size strictly < 32
  prompts_per_step: 4
  group_size_g: 8                               # 32 trajectories/step (4 prompts x 8 rollouts)
  gradient_accumulation_steps: 1

rl_objective:
  variant: "dr_grpo"                            # Removes GRPO length division & std-dev bias
  clip_low: 0.20
  clip_high: 0.28                               # DAPO Clip-Higher preserves exploratory tokens
  dynamic_sampling: true                        # Drop 0% and 100% pass groups
  kl_beta: 0.001
  max_steps: 200
  seed: 42
```

### 2.3 Budget Envelope (≤ \$25 Total GPU Spend)
- **Target Hardware**: 1× NVIDIA L4 (24 GB) or RTX 4090 (24 GB) or A10G (24 GB) on Modal (free monthly credits) / RunPod Spot (~\$0.35–\$0.75/hr).
- **Runtime**: 200 RLVR steps on 1.5B–3B with Unsloth + vLLM fast generation takes ~1.5–3 hours per arm (~4–6 hours total across Arms 1–2 + optional GRPO-vs-Dr.GRPO length-bias check = **\$2–\$10 actual spend**, leaving generous headroom under the \$25 cap).
- **Reproducibility**: Every training run writes `experiments/<run-id>/manifest.json` pinning model SHA, dataset SHA, `dasv` version, seeds, and logged JSONL metrics (`step`, `train_reward`, `clean_heldout_pass_at_1`, `clean_heldout_pass_pow_k`, `hack_divergence_gap`, `mean_response_len`, `policy_entropy`, `kl_div`, `zero_grad_batches_skipped`).

---

## 3. Train-Inference Logprob Drift Diagnostic Hook (`src/dasv/exporters/logprob_drift.py`)

To demonstrate mastery of 2025–2026 RLVR systems internals, `recipes/` includes a lightweight PyTorch/TRL callback measuring **rollout-to-training log-probability mismatch** between the inference engine (vLLM / SGLang) and the training backward pass (PyTorch / FSDP / Unsloth):

$$\Delta_t = \log \pi_{\text{train}}(a_t \mid s_t) - \log \pi_{\text{infer}}(a_t \mid s_t), \qquad w_t = \exp(\Delta_t)$$

```python
class LogprobDriftStats(BaseModel):
    step: int
    mean_abs_delta: float                 # E[|Δ_t|] across policy action tokens
    p95_abs_delta: float
    p99_abs_delta: float
    max_is_ratio: float                   # max_t exp(Δ_t)
    min_is_ratio: float                   # min_t exp(Δ_t)
    icepop_masked_frac: float             # Fraction of tokens outside [0.5, 2.0] in-band window
    recommended_mitigation: Literal["none", "tis_clip", "icepop_mask", "check_quant_precision"]

def compute_logprob_drift(
    train_logprobs: torch.Tensor,         # Shape: [B, T]
    infer_logprobs: torch.Tensor,         # Shape: [B, T]
    action_mask: torch.Tensor,            # Shape: [B, T], 1 on policy tokens, 0 on prompt/env obs
    icepop_bounds: tuple[float, float] = (0.5, 2.0),
) -> LogprobDriftStats: ...
```
- **Research Output**: Automatically plots $|\Delta_t|$ p95/p99 and `icepop_masked_frac` over RL training steps alongside the ablation curves—flagging hidden off-policy collapse caused by BF16/FP8 kernel divergence before it invalidates results.

---

## 4. Automated Verification Matrix (`SDD-05`)

1. **Unit (`tests/unit/test_exporters_and_drift.py`)**:
   - Schema validation for `trl_export`, `verl_export`, `prime_rl_export`, and OpenEnv `TrainingTrace` JSONL (asserting `loss_mask == 0` on environment observation tokens and `loss_mask == 1` on policy action tokens).
   - Synthetic tensor test for `compute_logprob_drift()`: zero drift returns `mean_abs_delta == 0.0`, `recommended_mitigation == "none"`; injected tail drift triggers `"icepop_mask"` with exact expected `icepop_masked_frac`.
2. **Dry-Run Recipe Smoke Test (`tests/integration/test_recipe_dryrun.py`)**:
   - Validates `recipes/lora_dr_grpo_ablation/config.yaml` against Pydantic schema (asserting *LoRA Without Regret* guardrails: all-linear targets present, `prompts_per_step < 32`) on CPU with 0 GPU cost.

---

## 5. Stage 4 Exit Gate (Approval Required for Stage 5: New Repo & CI Scaffold)

> [!IMPORTANT]
> All **5 Stage 4 Subsystem System Design Docs** are now complete:
> - **[SDD-01: Sandbox Runtime, Orchestrator & BudgetGuard](./04-DESIGN-SANDBOX-RUNTIME.md)**
> - **[SDD-02: Verifier Engine, Mutators, Hacker-Fixer & Rubric Stress-Testing](./04-DESIGN-VERIFIER-ENGINE.md)**
> - **[SDD-03: Interop Adapters & Native Validator Delegation](./04-DESIGN-INTEROP-ADAPTERS.md)**
> - **[SDD-04: Telemetry, Statistical Signal Analysis & Reporting](./04-DESIGN-TELEMETRY-EVAL.md)**
> - **[SDD-05: RL Trainer Integration, Causal Ablation & Logprob-Drift Hook](./04-DESIGN-RL-INTEGRATION.md)** (this document)
>
> **To unlock Stage 5 (Workspace Migration & Standalone GitHub Repo Setup)**:
> 1. Review and approve the 5 SDDs (or note any technical adjustments).
> 2. Confirm the exact local path for the new standalone repository (default: **`~/projects/dasv`**) and whether you want us to initialize it now with `uv`, migrate all Stage 0–4 docs into `docs/`, wire up `ubuntu-latest` GitHub Actions CI, and verify the walking skeleton (`dasv doctor` + `dasv inspect`).
