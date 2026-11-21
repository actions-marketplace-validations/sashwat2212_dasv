#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"
MODE="${2:---dry-run}"
SEED="${SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GRADING_BACKEND="${GRADING_BACKEND:-hardened-subprocess}"
TASKSET_DIR="${TASKSET_DIR:-${REPO_ROOT}/experiments/taskset_v1}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/experiments/v1_ablation/seed_${SEED}}"
# Use .venv/bin/python directly (never bare `uv run`, which would re-sync and strip GPU packages!)
if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PY_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PY_BIN="uv --directory ${REPO_ROOT} run --no-sync python"
fi

echo "[dasv-ablation] Validating LoRA Without Regret + Dr. GRPO guardrails: ${CONFIG_PATH}"
${PY_BIN} -c "
from pathlib import Path
from dasv.drift import load_ablation_recipe_config
cfg = load_ablation_recipe_config(Path('${CONFIG_PATH}'))
print(f'VALIDATED: model={cfg.model.base_model} r={cfg.lora.r} lr={cfg.optimizer.learning_rate} prompts_per_step={cfg.optimizer.prompts_per_step} objective={cfg.rl_objective.variant}')
"

if [ "${MODE}" = "--dry-run" ]; then
  DRY_DIR="${REPO_ROOT}/experiments/dry_run_check"
  ${PY_BIN} "${SCRIPT_DIR}/prepare_taskset.py" \
    --source synthetic --n-train 12 --n-heldout 8 --seed "${SEED}" --out-dir "${DRY_DIR}/taskset"
  ${PY_BIN} "${SCRIPT_DIR}/train_dr_grpo.py" \
    --config "${CONFIG_PATH}" --taskset-dir "${DRY_DIR}/taskset" \
    --out-dir "${DRY_DIR}/arm2" --arm arm2 --seed "${SEED}" --dry-run
  echo "[dasv-ablation] Dry-run complete (\$0 GPU spend). Pass '--smoke' or '--launch' on a 24GB GPU VM."
  exit 0
fi

if [ ! -f "${TASKSET_DIR}/taskset_manifest.json" ]; then
  echo "[dasv-ablation] Preparing BigCodeBench + MBPP+ Harbor taskset -> ${TASKSET_DIR}"
  ${PY_BIN} "${SCRIPT_DIR}/prepare_taskset.py" \
    --source hf --n-train 128 --n-heldout 64 --seed "${SEED}" --out-dir "${TASKSET_DIR}"
fi

if [ "${MODE}" = "--smoke" ]; then
  SMOKE_OUT="${REPO_ROOT}/experiments/smoke_gate_seed_${SEED}"
  echo "[dasv-ablation] Running Mandatory Pre-Spend Smoke Gate (Arm 0 Base + 10-step Arm 1 & Arm 2)"
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${SMOKE_OUT}/arm0/manifest.json" ]; then
    echo "[dasv-ablation] Skipping completed arm0 (${SMOKE_OUT}/arm0/manifest.json exists)"
  else
    ${PY_BIN} "${SCRIPT_DIR}/train_dr_grpo.py" \
      --config "${CONFIG_PATH}" --taskset-dir "${TASKSET_DIR}" --out-dir "${SMOKE_OUT}/arm0" \
      --arm arm0 --eval-k 1 --seed "${SEED}" --grading-backend "${GRADING_BACKEND}" ${EXTRA_TRAIN_ARGS:-}
  fi
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${SMOKE_OUT}/arm1/manifest.json" ]; then
    echo "[dasv-ablation] Skipping completed arm1 (${SMOKE_OUT}/arm1/manifest.json exists)"
  else
    ${PY_BIN} "${SCRIPT_DIR}/train_dr_grpo.py" \
      --config "${CONFIG_PATH}" --taskset-dir "${TASKSET_DIR}" --out-dir "${SMOKE_OUT}/arm1" \
      --arm arm1 --max-steps 10 --eval-every 10 --quick-eval-every 0 --eval-k 1 \
      --seed "${SEED}" --grading-backend "${GRADING_BACKEND}" --max-wall-minutes 25 ${EXTRA_TRAIN_ARGS:-}
  fi
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${SMOKE_OUT}/arm2/manifest.json" ]; then
    echo "[dasv-ablation] Skipping completed arm2 (${SMOKE_OUT}/arm2/manifest.json exists)"
  else
    ${PY_BIN} "${SCRIPT_DIR}/train_dr_grpo.py" \
      --config "${CONFIG_PATH}" --taskset-dir "${TASKSET_DIR}" --out-dir "${SMOKE_OUT}/arm2" \
      --arm arm2 --max-steps 10 --eval-every 10 --quick-eval-every 0 --eval-k 1 \
      --seed "${SEED}" --grading-backend "${GRADING_BACKEND}" --max-wall-minutes 25 ${EXTRA_TRAIN_ARGS:-}
  fi
  echo "[dasv-ablation] Smoke gate finished. Inspect ${SMOKE_OUT}/*/manifest.json before running '--launch'."
  exit 0
fi

if [ "${MODE}" = "--launch" ]; then
  echo "[dasv-ablation] Launching Full 3-Arm Controlled Ablation (seed=${SEED}, backend=${GRADING_BACKEND})..."
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${OUT_BASE}/arm0/manifest.json" ]; then
    echo "[dasv-ablation] Skipping completed arm0 (${OUT_BASE}/arm0/manifest.json exists)"
  else
    ${PY_BIN} "${SCRIPT_DIR}/train_dr_grpo.py" \
      --config "${CONFIG_PATH}" --taskset-dir "${TASKSET_DIR}" --out-dir "${OUT_BASE}/arm0" \
      --arm arm0 --eval-k 4 --seed "${SEED}" --grading-backend "${GRADING_BACKEND}" ${EXTRA_TRAIN_ARGS:-}
  fi
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${OUT_BASE}/arm1/manifest.json" ]; then
    echo "[dasv-ablation] Skipping completed arm1 (${OUT_BASE}/arm1/manifest.json exists)"
  else
    ${PY_BIN} "${SCRIPT_DIR}/train_dr_grpo.py" \
      --config "${CONFIG_PATH}" --taskset-dir "${TASKSET_DIR}" --out-dir "${OUT_BASE}/arm1" \
      --arm arm1 --max-steps 200 --eval-every 50 --quick-eval-every 25 --eval-k 4 \
      --seed "${SEED}" --grading-backend "${GRADING_BACKEND}" --max-wall-minutes 180 ${EXTRA_TRAIN_ARGS:-}
  fi
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${OUT_BASE}/arm2/manifest.json" ]; then
    echo "[dasv-ablation] Skipping completed arm2 (${OUT_BASE}/arm2/manifest.json exists)"
  else
    ${PY_BIN} "${SCRIPT_DIR}/train_dr_grpo.py" \
      --config "${CONFIG_PATH}" --taskset-dir "${TASKSET_DIR}" --out-dir "${OUT_BASE}/arm2" \
      --arm arm2 --max-steps 200 --eval-every 50 --quick-eval-every 25 --eval-k 4 \
      --seed "${SEED}" --grading-backend "${GRADING_BACKEND}" --max-wall-minutes 180 ${EXTRA_TRAIN_ARGS:-}
  fi
  ${PY_BIN} "${SCRIPT_DIR}/plot_ablation.py" \
    --arm0 "${OUT_BASE}/arm0/manifest.json" \
    --arm1 "${OUT_BASE}/arm1/metrics.jsonl" \
    --arm2 "${OUT_BASE}/arm2/metrics.jsonl" \
    --arm1-eval "${OUT_BASE}/arm1/heldout_eval.jsonl" \
    --arm2-eval "${OUT_BASE}/arm2/heldout_eval.jsonl" \
    --out "${OUT_BASE}/ablation_comparison.html" \
    --svg-dir "${OUT_BASE}/svgs"
  echo "[dasv-ablation] Full 3-arm ablation complete -> ${OUT_BASE}/ablation_comparison.html"
  exit 0
fi

echo "Unknown mode '${MODE}'. Expected --dry-run, --smoke, or --launch." >&2
exit 2
