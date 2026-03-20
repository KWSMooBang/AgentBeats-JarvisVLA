#!/bin/bash

# ==============================================================================
# MCU Benchmark Runner — Scripted Policy Agent
#
# Usage:
#   bash scripts/run_benchmark.sh                       # default: combat
#   bash scripts/run_benchmark.sh mining_and_collecting  # specify category
#   bash scripts/run_benchmark.sh --list                 # list all categories
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Load .env file ───────────────────────────────────────────────────────────
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# ── Category (first positional arg, or default) ─────────────────────────────
CATEGORY="${1:-mining_and_collecting}"

# ── GPU ──────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── LLM Planner config ──────────────────────────────────────────────────────
PLANNER_API_KEY="${PLANNER_API_KEY:-${NEBIUS_API_KEY:-EMPTY}}"
PLANNER_URL="${PLANNER_URL:-https://api.tokenfactory.nebius.com/v1/}"
PLANNER_MODEL="${PLANNER_MODEL:-openai/gpt-oss-120b-fast}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.2}"

# ── VLM State Checker config ────────────────────────────────────────────────
VLM_API_KEY="${VLM_API_KEY:-${NEBIUS_API_KEY:-EMPTY}}"
VLM_URL="${VLM_URL:-https://api.tokenfactory.nebius.com/v1/}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}"
VLM_TEMPERATURE="${VLM_TEMPERATURE:-0.1}"

# ── Task / output config ────────────────────────────────────────────────────
TASKS_DIR="${TASKS_DIR:-assets/mcu_tasks}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MAX_STEPS="${MAX_STEPS:-600}"
OBS_SIZE="${OBS_SIZE:-128 128}"

# ==============================================================================
echo "========================================"
echo "  Scripted Policy — MCU Benchmark"
echo "========================================"
echo "  Category       : $CATEGORY"
echo "  Planner model  : $PLANNER_MODEL"
echo "  VLM model      : $VLM_MODEL  ($VLM_URL)"
echo "  Max steps      : $MAX_STEPS"
echo "  Output dir     : $OUTPUT_DIR"
echo "  CUDA devices   : $CUDA_VISIBLE_DEVICES"
echo "========================================"
echo ""

python examples/run_standalone.py \
    --category "$CATEGORY" \
    --tasks-dir "$TASKS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --obs-size $OBS_SIZE \
    --planner-api-key "$PLANNER_API_KEY" \
    --planner-url "$PLANNER_URL" \
    --planner-model "$PLANNER_MODEL" \
    --planner-temperature "$PLANNER_TEMPERATURE" \
    --vlm-api-key "$VLM_API_KEY" \
    --vlm-url "$VLM_URL" \
    --vlm-model "$VLM_MODEL" \
    --vlm-temperature "$VLM_TEMPERATURE" \
    --verbose
