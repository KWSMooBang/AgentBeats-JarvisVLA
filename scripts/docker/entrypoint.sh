#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env if present so planner-related config can be managed centrally.
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$PROJECT_DIR/.env"
  set +a
fi

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-9020}"
VLA_MODEL_PATH="${VLA_MODEL_PATH:-/models/JarvisVLA-Qwen2-VL-7B}"
VLA_BASE_URL="${VLA_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-jarvisvla}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-10}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-image=5}"
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-300}"

PURPLE_HOST="${PURPLE_HOST:-0.0.0.0}"
PURPLE_PORT="${PURPLE_PORT:-9019}"

PLANNER_API_KEY="${PLANNER_API_KEY:-${API_KEY:-EMPTY}}"
PLANNER_URL="${PLANNER_URL:-${URL:-https://api.tokenfactory.nebius.com/v1/}}"
PLANNER_MODEL="${PLANNER_MODEL:-${MODEL:-openai/gpt-oss-120b-fast}}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.2}"

VLA_API_KEY="${VLA_API_KEY:-EMPTY}"
VLA_HISTORY_NUM="${VLA_HISTORY_NUM:-4}"
VLA_ACTION_CHUNK_LEN="${VLA_ACTION_CHUNK_LEN:-1}"
VLA_BPE="${VLA_BPE:-0}"
VLA_INSTRUCTION_TYPE="${VLA_INSTRUCTION_TYPE:-normal}"
VLA_TEMPERATURE="${VLA_TEMPERATURE:-0.7}"

DEVICE="${DEVICE:-cuda}"

if [[ ! -d "${VLA_MODEL_PATH}" ]]; then
  echo "[entrypoint] ERROR: VLA_MODEL_PATH does not exist: ${VLA_MODEL_PATH}" >&2
  echo "[entrypoint] HINT: mount model dir or build with --build-arg DOWNLOAD_JARVIS_MODEL=1" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[entrypoint] stopping vLLM (pid=${VLLM_PID})"
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

echo "[entrypoint] starting vLLM on ${VLLM_HOST}:${VLLM_PORT}"
uv run vllm serve "${VLA_MODEL_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --trust-remote-code \
  --served-model-name "${VLLM_SERVED_MODEL_NAME}" \
  --limit-mm-per-prompt "${VLLM_LIMIT_MM_PER_PROMPT}" &
VLLM_PID=$!

echo "[entrypoint] waiting for vLLM readiness at http://127.0.0.1:${VLLM_PORT}/v1/models"
READY=0
for _ in $(seq 1 "${VLLM_START_TIMEOUT}"); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[entrypoint] ERROR: vLLM process exited before ready" >&2
    exit 1
  fi
  sleep 1
done

if [[ "${READY}" != "1" ]]; then
  echo "[entrypoint] ERROR: vLLM readiness timeout after ${VLLM_START_TIMEOUT}s" >&2
  exit 1
fi

echo "[entrypoint] vLLM ready, starting Purple server on ${PURPLE_HOST}:${PURPLE_PORT}"
exec uv run python -m src.server.app \
  --host "${PURPLE_HOST}" \
  --port "${PURPLE_PORT}" \
  --planner-api-key "${PLANNER_API_KEY}" \
  --planner-url "${PLANNER_URL}" \
  --planner-model "${PLANNER_MODEL}" \
  --planner-temperature "${PLANNER_TEMPERATURE}" \
  --vla-checkpoint-path "${VLA_MODEL_PATH}" \
  --vla-url "${VLA_BASE_URL}" \
  --vla-api-key "${VLA_API_KEY}" \
  --vla-history-num "${VLA_HISTORY_NUM}" \
  --vla-action-chunk-len "${VLA_ACTION_CHUNK_LEN}" \
  --vla-bpe "${VLA_BPE}" \
  --vla-instruction-type "${VLA_INSTRUCTION_TYPE}" \
  --vla-temperature "${VLA_TEMPERATURE}" \
  --device "${DEVICE}"
