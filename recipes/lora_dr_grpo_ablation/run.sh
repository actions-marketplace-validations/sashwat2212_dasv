#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"
MODE="${2:---dry-run}"

echo "[dasv-ablation] Validating LoRA Without Regret + Dr. GRPO guardrails: ${CONFIG_PATH}"
uv --directory "${REPO_ROOT}" run python -c "
from pathlib import Path
from dasv.drift import load_ablation_recipe_config
cfg = load_ablation_recipe_config(Path('${CONFIG_PATH}'))
print(f'VALIDATED: model={cfg.model.base_model} r={cfg.lora.r} lr={cfg.optimizer.learning_rate} prompts_per_step={cfg.optimizer.prompts_per_step} objective={cfg.rl_objective.variant}')
"

if [ "${MODE}" = "--dry-run" ]; then
  echo "[dasv-ablation] Dry-run complete (\$0 GPU spend). Pass '--launch' on a 24GB GPU node (L4/4090/A10G) with optional 'recipes' extras installed."
  exit 0
fi

echo "[dasv-ablation] Launching 3-arm ablation (Arm 0 Base, Arm 1 Unhardened, Arm 2 dasv-Hardened)..."
echo "Install optional GPU extras first: uv sync --extra recipes"
echo "Wire TRL GRPOTrainer to exported trl_dataset.jsonl + trl_reward_func.py and LogprobDriftTracker."

