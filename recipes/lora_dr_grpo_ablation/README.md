# Reproducible *LoRA Without Regret* + Dr. GRPO / DAPO Ablation Recipe (`FEAT-18`)

This recipe provides a budget-capped (**<= $25** on 1x L4 / RTX 4090 / A10G 24 GB spot GPU) causal RLVR post-training ablation comparing:

- **Arm 0 (Base)**: Zero-RL reference `Qwen/Qwen2.5-Coder-3B-Instruct` on the clean held-out suite.
- **Arm 1 (Unhardened RL)**: Unfiltered taskset with naive in-container grading.
- **Arm 2 (`dasv`-Hardened RL)**: Same tasks after `dasv audit` + `dasv redteam --propose-fix` (`VERIFIED_FIX`) + `dasv profile` (`GOLDILOCKS` filtering) + `dasv export --format trl` (`CleanImageGrader` reward wrapper).

## *LoRA Without Regret* Guardrails Enforced by `config.yaml`

1. **All-linear modules** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) — never attention-only.
2. **Low rank** (`r=16`, `lora_alpha=32`) with ~10x FullFT learning rate (`5e-5`).
3. **Effective prompt batch size strictly < 32** (`prompts_per_step=4`, `gradient_accumulation_steps=1`).
4. **Dr. GRPO + DAPO Clip-Higher** (`clip_low=0.20`, `clip_high=0.28`, `dynamic_sampling=true`).
5. **Train-Inference Logprob Drift Monitoring**: attaches `dasv.drift.LogprobDriftTracker` to detect off-policy kernel divergence (`RULE-DRFT-001`) before collapse.

## Usage

```bash
# 1. Zero-cost dry-run config guardrail validation (CPU, $0)
bash recipes/lora_dr_grpo_ablation/run.sh recipes/lora_dr_grpo_ablation/config.yaml --dry-run

# 2. Optional GPU install (only on training nodes; core `dasv` stays torch-free)
uv sync --extra recipes

# 3. Plot Arm 1 vs. Arm 2 JSONL training trajectories to a zero-CDN HTML/SVG report
uv run python recipes/lora_dr_grpo_ablation/plot_ablation.py \
  --arm1 experiments/arm1/metrics.jsonl \
  --arm2 experiments/arm2/metrics.jsonl \
  --out experiments/ablation_comparison.html
```
