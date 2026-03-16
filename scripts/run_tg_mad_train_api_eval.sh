#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o job-tgmad-train-api-%j.output
#SBATCH --time=24:00:00
#SBATCH -J tgmad-api

# TG-MAD training with API-based evaluator (e.g. kimi-k2.5).
# Only 1 GPU needed: local vLLM for debater, remote API for evaluator/backward.
# Requires KIMI_API_KEY env var (or pass --evaluator_api_key).

set -euo pipefail

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
DEBATER_PORT="${DEBATER_PORT:-8000}"
DEBATER_URL="http://${SERVER_HOST}:${DEBATER_PORT}/v1"
OUTPUT_DIR="${OUTPUT_DIR:-out/tg_mad}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-5}"
TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-2}"
TRAIN_SEED="${TRAIN_SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
EVALUATOR_MAX_NEW_TOKENS="${EVALUATOR_MAX_NEW_TOKENS:-2048}"
DEBATER_MODEL_NAME="${DEBATER_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
DEBATER_GPU_MEMORY="${DEBATER_GPU_MEMORY:-0.85}"
DEBATER_MAX_MODEL_LEN="${DEBATER_MAX_MODEL_LEN:-8192}"
DEBATER_MAX_NUM_SEQS="${DEBATER_MAX_NUM_SEQS:-4}"
DEBATER_ENFORCE_EAGER="${DEBATER_ENFORCE_EAGER:-1}"
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-vllm}"

DEBATER_LOG="${OUTPUT_DIR%/}/vllm_debater_api_train.log"

cleanup() {
    local exit_code=$?
    if [[ -n "${DEBATER_PID:-}" ]]; then
        kill "${DEBATER_PID}" 2>/dev/null || true
        wait "${DEBATER_PID}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}
trap cleanup EXIT

wait_for_health() {
    local name="$1"
    local url="$2"
    local log_file="$3"
    local pid="$4"
    local waited=0

    echo "Waiting for ${name} health check at ${url}..."
    while (( waited < MAX_WAIT_SECONDS )); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            echo "${name} server is healthy."
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: ${name} server exited before becoming healthy."
            echo "See ${log_file}"
            tail -n 40 "${log_file}" 2>/dev/null || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        echo "  waiting for ${name}... ${waited}/${MAX_WAIT_SECONDS}s"
    done

    echo "ERROR: ${name} server did not become healthy."
    echo "See ${log_file}"
    return 1
}

echo "=== TG-MAD Training (API Evaluator) ==="
echo "Node: $(hostname)"
date

# Load API key from .env if not already set
if [[ -z "${KIMI_API_KEY:-}" ]] && [[ -f .env ]]; then
    source .env
fi
if [[ -z "${KIMI_API_KEY:-}" ]]; then
    echo "ERROR: KIMI_API_KEY env var is not set."
    echo "Either: export KIMI_API_KEY=your_key, or put 'export KIMI_API_KEY=...' in .env"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export OPENAI_API_KEY=EMPTY
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_TOKEN
HF_TOKEN="$(cat ./token 2>/dev/null || echo "")"

if ! python -c "import vllm" >/dev/null 2>&1; then
    echo "ERROR: vllm is not installed in the current environment."
    exit 1
fi

# Start debater vLLM server (only model on GPU)
echo "Starting debater vLLM server..."
debater_args=(
    --host "${SERVER_HOST}"
    --port "${DEBATER_PORT}"
    --model "${DEBATER_MODEL_NAME}"
    --download-dir ./models
    --gpu-memory-utilization "${DEBATER_GPU_MEMORY}"
    --max-num-seqs "${DEBATER_MAX_NUM_SEQS}"
    --dtype bfloat16
    --tensor-parallel-size 1
    --generation-config "${VLLM_GENERATION_CONFIG}"
)
if [[ -n "${DEBATER_MAX_MODEL_LEN}" ]]; then
    debater_args+=(--max-model-len "${DEBATER_MAX_MODEL_LEN}")
fi
if [[ "${DEBATER_ENFORCE_EAGER}" == "1" ]]; then
    debater_args+=(--enforce-eager)
fi

python -m vllm.entrypoints.openai.api_server \
    "${debater_args[@]}" \
    > "${DEBATER_LOG}" 2>&1 &
DEBATER_PID=$!

wait_for_health "debater" "http://${SERVER_HOST}:${DEBATER_PORT}/health" "${DEBATER_LOG}" "${DEBATER_PID}"

echo "Running TG-MAD training with API evaluator..."
python -u tg_mad/train.py \
    --evaluator_type api \
    --debater_base_url "${DEBATER_URL}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --num_epochs "${TRAIN_NUM_EPOCHS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --evaluator_max_new_tokens "${EVALUATOR_MAX_NEW_TOKENS}" \
    --seed "${TRAIN_SEED}" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete."
date
