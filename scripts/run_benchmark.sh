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
CATEGORY="${1:-mine_diamond_from_scratch}"

# ── GPU ──────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── LLM Planner config ──────────────────────────────────────────────────────
PLANNER_API_KEY="${PLANNER_API_KEY:-${API_KEY:-EMPTY}}"
PLANNER_URL="${PLANNER_URL:-${URL:-https://api.openai.com/v1}}"
PLANNER_MODEL="${PLANNER_MODEL:-${MODEL:-gpt-5.4-mini}}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.2}"

# ── Runtime VLM config (required) ───────────────────────────────────────────
VLM_API_KEY="${VLM_API_KEY:-${API_KEY:-EMPTY}}"
VLM_URL="${VLM_URL:-${URL:-https://api.openai.com/v1}}"
VLM_MODEL="${VLM_MODEL:-${MODEL:-gpt-5.4-mini}}"
VLM_TEMPERATURE="${VLM_TEMPERATURE:-0.2}"
VQA_INTERVAL_STEPS="${VQA_INTERVAL_STEPS:-600}"

# ── JarvisVLA config (required) ─────────────────────────────────────────────
VLA_CHECKPOINT_PATH="${VLA_CHECKPOINT_PATH:-./models/JarvisVLA-Qwen2-VL-7B}"
VLA_URL="${VLA_URL:-http://localhost:9020/v1}"
VLA_API_KEY="${VLA_API_KEY:-EMPTY}"
VLA_HISTORY_NUM="${VLA_HISTORY_NUM:-4}"
VLA_ACTION_CHUNK_LEN="${VLA_ACTION_CHUNK_LEN:-1}"
VLA_BPE="${VLA_BPE:-0}"
VLA_INSTRUCTION_TYPE="${VLA_INSTRUCTION_TYPE:-normal}"
VLA_TEMPERATURE="${VLA_TEMPERATURE:-0.7}"

if [[ ! -d "$VLA_CHECKPOINT_PATH" ]]; then
    echo "[ERROR] VLA_CHECKPOINT_PATH not found: $VLA_CHECKPOINT_PATH"
    exit 1
fi

# ── Task / output config ────────────────────────────────────────────────────
TASKS_DIR="${TASKS_DIR:-./tasks}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MAX_STEPS="${MAX_STEPS:-100}"
OBS_SIZE="${OBS_SIZE:-640 360}"

# ==============================================================================
echo "========================================"
echo "  Scripted Policy — MCU Benchmark"
echo "========================================"
echo "  Category       : $CATEGORY"
echo "  Planner model  : $PLANNER_MODEL"
echo "  VLM model      : $VLM_MODEL"
echo "  VLM endpoint   : $VLM_URL"
echo "  VLA model path : $VLA_CHECKPOINT_PATH"
echo "  VLA endpoint   : $VLA_URL"
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
    --vqa-interval-steps "$VQA_INTERVAL_STEPS" \
    --vla-checkpoint-path "$VLA_CHECKPOINT_PATH" \
    --vla-url "$VLA_URL" \
    --vla-api-key "$VLA_API_KEY" \
    --vla-history-num "$VLA_HISTORY_NUM" \
    --vla-action-chunk-len "$VLA_ACTION_CHUNK_LEN" \
    --vla-bpe "$VLA_BPE" \
    --vla-instruction-type "$VLA_INSTRUCTION_TYPE" \
    --vla-temperature "$VLA_TEMPERATURE" \
    --verbose
