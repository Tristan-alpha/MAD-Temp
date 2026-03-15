#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o job-tgmad-all-in-one-%j.output
#SBATCH --time=24:00:00
#SBATCH -J tgmad-all

set -euo pipefail

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
DEBATER_PORT="${DEBATER_PORT:-8000}"
EVALUATOR_PORT="${EVALUATOR_PORT:-8001}"
DEBATER_URL="http://${SERVER_HOST}:${DEBATER_PORT}/v1"
EVALUATOR_URL="http://${SERVER_HOST}:${EVALUATOR_PORT}/v1"
OUTPUT_DIR="${OUTPUT_DIR:-out/tg_mad}"
PROMPT_HISTORY_PATH="${PROMPT_HISTORY_PATH:-${OUTPUT_DIR%/}/prompt_history.json}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-5}"
TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-2}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DEBATER_MODEL_NAME="${DEBATER_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
EVALUATOR_MODEL_NAME="${EVALUATOR_MODEL_NAME:-Qwen/Qwen3-8B}"
DEBATER_GPU_MEMORY="${DEBATER_GPU_MEMORY:-0.30}"
EVALUATOR_GPU_MEMORY="${EVALUATOR_GPU_MEMORY:-0.35}"
DEBATER_MAX_MODEL_LEN="${DEBATER_MAX_MODEL_LEN:-3072}"
EVALUATOR_MAX_MODEL_LEN="${EVALUATOR_MAX_MODEL_LEN:-3072}"
DEBATER_MAX_NUM_SEQS="${DEBATER_MAX_NUM_SEQS:-2}"
EVALUATOR_MAX_NUM_SEQS="${EVALUATOR_MAX_NUM_SEQS:-2}"
DEBATER_CPU_OFFLOAD_GB="${DEBATER_CPU_OFFLOAD_GB:-0}"
EVALUATOR_CPU_OFFLOAD_GB="${EVALUATOR_CPU_OFFLOAD_GB:-10}"
DEBATER_ENFORCE_EAGER="${DEBATER_ENFORCE_EAGER:-1}"
EVALUATOR_ENFORCE_EAGER="${EVALUATOR_ENFORCE_EAGER:-1}"
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-vllm}"

DEBATER_LOG="${OUTPUT_DIR%/}/vllm_debater.log"
EVALUATOR_LOG="${OUTPUT_DIR%/}/vllm_evaluator.log"

cleanup() {
    local exit_code=$?
    if [[ -n "${DEBATER_PID:-}" ]]; then
        kill "${DEBATER_PID}" 2>/dev/null || true
        wait "${DEBATER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${EVALUATOR_PID:-}" ]]; then
        kill "${EVALUATOR_PID}" 2>/dev/null || true
        wait "${EVALUATOR_PID}" 2>/dev/null || true
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

echo "=== TG-MAD All-in-One Job ==="
echo "Node: $(hostname)"
date

mkdir -p "${OUTPUT_DIR}"
export OPENAI_API_KEY=EMPTY
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_TOKEN
HF_TOKEN="$(cat ./token 2>/dev/null || echo "")"

if ! python -c "import vllm" >/dev/null 2>&1; then
    echo "ERROR: vllm is not installed in the current environment."
    echo "Install it in your user environment, for example:"
    echo "  INSTALL_VLLM=1 bash scripts/setup_textgrad_env.sh"
    exit 1
fi

echo "Starting debater vLLM server on ${DEBATER_URL}..."
debater_args=(
    --host "${SERVER_HOST}"
    --port "${DEBATER_PORT}"
    --model "${DEBATER_MODEL_NAME}"
    --download-dir ./models
    --gpu-memory-utilization "${DEBATER_GPU_MEMORY}"
    --max-model-len "${DEBATER_MAX_MODEL_LEN}"
    --max-num-seqs "${DEBATER_MAX_NUM_SEQS}"
    --cpu-offload-gb "${DEBATER_CPU_OFFLOAD_GB}"
    --dtype bfloat16
    --tensor-parallel-size 1
    --generation-config "${VLLM_GENERATION_CONFIG}"
)
if [[ "${DEBATER_ENFORCE_EAGER}" == "1" ]]; then
    debater_args+=(--enforce-eager)
fi

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    "${debater_args[@]}" \
    > "${DEBATER_LOG}" 2>&1 &
DEBATER_PID=$!

wait_for_health "debater" "http://${SERVER_HOST}:${DEBATER_PORT}/health" "${DEBATER_LOG}" "${DEBATER_PID}"

echo "Starting evaluator vLLM server on ${EVALUATOR_URL}..."
evaluator_args=(
    --host "${SERVER_HOST}"
    --port "${EVALUATOR_PORT}"
    --model "${EVALUATOR_MODEL_NAME}"
    --download-dir ./models
    --gpu-memory-utilization "${EVALUATOR_GPU_MEMORY}"
    --max-model-len "${EVALUATOR_MAX_MODEL_LEN}"
    --max-num-seqs "${EVALUATOR_MAX_NUM_SEQS}"
    --cpu-offload-gb "${EVALUATOR_CPU_OFFLOAD_GB}"
    --dtype bfloat16
    --tensor-parallel-size 1
    --generation-config "${VLLM_GENERATION_CONFIG}"
)
if [[ "${EVALUATOR_ENFORCE_EAGER}" == "1" ]]; then
    evaluator_args+=(--enforce-eager)
fi

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    "${evaluator_args[@]}" \
    > "${EVALUATOR_LOG}" 2>&1 &
EVALUATOR_PID=$!

wait_for_health "evaluator" "http://${SERVER_HOST}:${EVALUATOR_PORT}/health" "${EVALUATOR_LOG}" "${EVALUATOR_PID}"

echo "Both vLLM servers are healthy."

echo "Running TG-MAD training..."
python -u tg_mad/train.py \
    --debater_base_url "${DEBATER_URL}" \
    --evaluator_base_url "${EVALUATOR_URL}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --num_epochs "${TRAIN_NUM_EPOCHS}" \
    --seed "${TRAIN_SEED}" \
    --output_dir "${OUTPUT_DIR}"

echo "Running TG-MAD evaluation..."
python -u tg_mad/evaluate.py \
    --debater_base_url "${DEBATER_URL}" \
    --prompt_history "${PROMPT_HISTORY_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${TRAIN_SEED}"

echo "All-in-one TG-MAD job completed successfully."
date
