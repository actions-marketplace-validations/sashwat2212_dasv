# `dasv` LoRA Without Regret + Dr. GRPO Causal RLVR Ablation (`KPI-RES-03`)

This recipe runs the 3-arm controlled RLVR ablation proving the causal impact of `dasv` pre-training environment hardening while holding base model (`Qwen/Qwen2.5-Coder-3B-Instruct`), *LoRA Without Regret* hyperparameters (`all-linear`, `r=16`, `lr=5e-5`, `prompts_per_step=4 < 32`), Dr. GRPO / DAPO Clip-Higher (`0.20 / 0.28`), seed, and compute strictly identical:

- **Arm 0**: Unmodified Base Model (`Qwen/Qwen2.5-Coder-3B-Instruct`) on the 64 clean unseen held-out tasks.
- **Arm 1 (Unhardened)**: Trained on 128 Harbor tasks containing realistic `RULE-BNDY-001` (20% exposed mutable workspace tests) and `RULE-MUT-002` (20% under-specified tests) flaws, with off-policy clean shadow grading logging true per-step `hack_divergence_gap` and `tamper_rate`.
- **Arm 2 (`dasv`-Hardened)**: Trained on the exact same 128 specs hardened by `dasv` (pinned base image, `allow_internet = false`, full hidden test suites, two-phase tamper-proof Verification Plane).

## 1. CPU Dry-Run Verification (₹0 / \$0)

Run locally on any Mac/Linux machine without a GPU:

```bash
./recipes/lora_dr_grpo_ablation/run.sh recipes/lora_dr_grpo_ablation/config.yaml --dry-run
```

## 2. E2E Networks 1× NVIDIA L4 (24 GB) Ubuntu VM Setup (~₹49/hr + 18% GST via UPI)

1. Provision a **1× NVIDIA L4 (24 GB VRAM)** GPU VM on **E2E Networks** (Ubuntu 22.04/24.04 with CUDA 12.x, >= 80 GB root disk) and SSH into the VM.
2. Bootstrap `uv`, clone `dasv`, and install the CUDA GPU stack:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/sashwat2212/dasv.git && cd dasv
uv sync --frozen
uv pip install -r recipes/lora_dr_grpo_ablation/requirements-gpu.txt
export HF_TOKEN="hf_your_read_token_here"   # never passed into grading subprocesses!
nvidia-smi
```

3. *(Optional)* To also verify native Docker `CleanImageGrader` on the E2E VM:
   `docker info >/dev/null && uv run dasv doctor` (and pass `GRADING_BACKEND=docker` for final Docker-backed held-out runs, or keep default `GRADING_BACKEND=hardened-subprocess` for fast parallel step grading).

## 3. Mandatory 10-Minute Smoke Gate (Run Before Full 200-Step Ablation!)

```bash
SEED=42 ./recipes/lora_dr_grpo_ablation/run.sh recipes/lora_dr_grpo_ablation/config.yaml --smoke
```

**GO / NO-GO Checklist (inspect `experiments/smoke_gate_seed_42/*/manifest.json`):**
- `arm0` base `clean_heldout_pass_at_1` is inside **GOLDILOCKS (`0.15 – 0.45`)**.
- `peak_vram_gb < 21.0` GB on 24 GB L4.
- `drift.jsonl` contains finite `p95_abs_delta` and `icepop_masked_frac` values.
- `arm1` exhibits measurable `hack_divergence_gap > 0.02` or `tamper_rate > 0`.

## 4. Full 3-Arm Controlled Run & Multi-Seed (`seed=42`, optional `seed=1337`)

Run inside `tmux` so SSH disconnects never interrupt training:

```bash
tmux new -s dasv-ablation
SEED=42 ./recipes/lora_dr_grpo_ablation/run.sh recipes/lora_dr_grpo_ablation/config.yaml --launch
```

If Seed 1 completes cleanly within `<= 50%` of your prepaid UPI budget, run Seed 2:

```bash
SEED=1337 ./recipes/lora_dr_grpo_ablation/run.sh recipes/lora_dr_grpo_ablation/config.yaml --launch
```

## 5. Pull Artifacts & Immediately Delete VM (Kill-Switch!)

From your local Mac, copy ONLY the lightweight JSONL/manifest/SVG/HTML telemetry (never copy multi-GB checkpoints into git):

```bash
mkdir -p results/v1_ablation
scp -r root@<E2E_VM_IP>:~/dasv/experiments/v1_ablation/seed_42/{arm0,arm1,arm2}/{manifest.json,metrics.jsonl,drift.jsonl,heldout_eval.jsonl} results/v1_ablation/
scp -r root@<E2E_VM_IP>:~/dasv/experiments/v1_ablation/seed_42/{ablation_comparison.html,svgs} results/v1_ablation/
```

**IMMEDIATELY go to the E2E Networks MyAccount dashboard and DELETE the L4 GPU node + attached disk** so ongoing billing drops to **₹0/hr**.
