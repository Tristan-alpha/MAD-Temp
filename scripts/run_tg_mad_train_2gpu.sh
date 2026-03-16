#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o job-tgmad-train-2gpu-%j.output
#SBATCH --time=24:00:00
#SBATCH -J tgmad-train2

set -euo pipefail

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
DEBATER_PORT="${DEBATER_PORT:-8000}"
EVALUATOR_PORT="${EVALUATOR_PORT:-8001}"
DEBATER_URL="http://${SERVER_HOST}:${DEBATER_PORT}/v1"
EVALUATOR_URL="http://${SERVER_HOST}:${EVALUATOR_PORT}/v1"
OUTPUT_DIR="${OUTPUT_DIR:-out/tg_mad}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-2}"
TRAIN_N_AGENTS="${TRAIN_N_AGENTS:-3}"
TRAIN_N_ROUNDS="${TRAIN_N_ROUNDS:-3}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_SIZE="${TRAIN_SIZE:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
EVALUATOR_MAX_NEW_TOKENS="${EVALUATOR_MAX_NEW_TOKENS:-512}"
JOB_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
DEBATER_GPU_SLOTS="${DEBATER_GPU_SLOTS:-0}"
EVALUATOR_GPU_SLOTS="${EVALUATOR_GPU_SLOTS:-1}"
DEBATER_CUDA_VISIBLE_DEVICES="${DEBATER_CUDA_VISIBLE_DEVICES:-}"
EVALUATOR_CUDA_VISIBLE_DEVICES="${EVALUATOR_CUDA_VISIBLE_DEVICES:-}"
DEBATER_MODEL_NAME="${DEBATER_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
EVALUATOR_MODEL_NAME="${EVALUATOR_MODEL_NAME:-Qwen/Qwen3-8B}"
EVALUATOR_ENGINE_MODEL="${EVALUATOR_ENGINE_MODEL:-hosted_vllm/${EVALUATOR_MODEL_NAME}}"
DEBATER_GPU_MEMORY="${DEBATER_GPU_MEMORY:-0.45}"
EVALUATOR_GPU_MEMORY="${EVALUATOR_GPU_MEMORY:-0.55}"
DEBATER_MAX_MODEL_LEN="${DEBATER_MAX_MODEL_LEN:-8192}"
EVALUATOR_MAX_MODEL_LEN="${EVALUATOR_MAX_MODEL_LEN:-16384}"
DEBATER_MAX_NUM_SEQS="${DEBATER_MAX_NUM_SEQS:-4}"
EVALUATOR_MAX_NUM_SEQS="${EVALUATOR_MAX_NUM_SEQS:-1}"
DEBATER_TENSOR_PARALLEL_SIZE="${DEBATER_TENSOR_PARALLEL_SIZE:-1}"
EVALUATOR_TENSOR_PARALLEL_SIZE="${EVALUATOR_TENSOR_PARALLEL_SIZE:-1}"
DEBATER_ENFORCE_EAGER="${DEBATER_ENFORCE_EAGER:-1}"
EVALUATOR_ENFORCE_EAGER="${EVALUATOR_ENFORCE_EAGER:-1}"
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-vllm}"

DEBATER_LOG="${OUTPUT_DIR%/}/vllm_debater_train.log"
EVALUATOR_LOG="${OUTPUT_DIR%/}/vllm_evaluator_train.log"

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

resolve_visible_devices() {
    local explicit_devices="$1"
    local slot_spec="$2"

    if [[ -n "${explicit_devices}" ]]; then
        printf '%s\n' "${explicit_devices}"
        return 0
    fi

    if [[ -z "${JOB_CUDA_VISIBLE_DEVICES}" ]]; then
        printf '%s\n' "${slot_spec}"
        return 0
    fi

    IFS=',' read -r -a allocated <<< "${JOB_CUDA_VISIBLE_DEVICES}"
    IFS=',' read -r -a slots <<< "${slot_spec}"

    local resolved=()
    local slot
    for slot in "${slots[@]}"; do
        if [[ -z "${slot}" ]]; then
            continue
        fi
        if (( slot < 0 || slot >= ${#allocated[@]} )); then
            echo "ERROR: GPU slot ${slot} is outside allocated devices ${JOB_CUDA_VISIBLE_DEVICES}" >&2
            exit 1
        fi
        resolved+=("${allocated[$slot]}")
    done

    local joined=""
    local idx
    for idx in "${!resolved[@]}"; do
        if [[ -n "${joined}" ]]; then
            joined+=","
        fi
        joined+="${resolved[$idx]}"
    done
    printf '%s\n' "${joined}"
}

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

echo "=== TG-MAD Training ==="
echo "Node: $(hostname)"
echo "Allocated job CUDA_VISIBLE_DEVICES: ${JOB_CUDA_VISIBLE_DEVICES:-<unset>}"
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

DEBATER_VISIBLE_DEVICES="$(resolve_visible_devices "${DEBATER_CUDA_VISIBLE_DEVICES}" "${DEBATER_GPU_SLOTS}")"
EVALUATOR_VISIBLE_DEVICES="$(resolve_visible_devices "${EVALUATOR_CUDA_VISIBLE_DEVICES}" "${EVALUATOR_GPU_SLOTS}")"

echo "Starting debater vLLM server on devices ${DEBATER_VISIBLE_DEVICES} (TP=${DEBATER_TENSOR_PARALLEL_SIZE})..."
debater_args=(
    --host "${SERVER_HOST}"
    --port "${DEBATER_PORT}"
    --model "${DEBATER_MODEL_NAME}"
    --download-dir ./models
    --gpu-memory-utilization "${DEBATER_GPU_MEMORY}"
    --max-num-seqs "${DEBATER_MAX_NUM_SEQS}"
    --dtype bfloat16
    --tensor-parallel-size "${DEBATER_TENSOR_PARALLEL_SIZE}"
    --generation-config "${VLLM_GENERATION_CONFIG}"
)
if [[ -n "${DEBATER_MAX_MODEL_LEN}" ]]; then
    debater_args+=(--max-model-len "${DEBATER_MAX_MODEL_LEN}")
fi
if [[ "${DEBATER_ENFORCE_EAGER}" == "1" ]]; then
    debater_args+=(--enforce-eager)
fi

CUDA_VISIBLE_DEVICES="${DEBATER_VISIBLE_DEVICES}" python -m vllm.entrypoints.openai.api_server \
    "${debater_args[@]}" \
    > "${DEBATER_LOG}" 2>&1 &
DEBATER_PID=$!

echo "Starting evaluator vLLM server on devices ${EVALUATOR_VISIBLE_DEVICES} (TP=${EVALUATOR_TENSOR_PARALLEL_SIZE})..."
evaluator_args=(
    --host "${SERVER_HOST}"
    --port "${EVALUATOR_PORT}"
    --model "${EVALUATOR_MODEL_NAME}"
    --download-dir ./models
    --gpu-memory-utilization "${EVALUATOR_GPU_MEMORY}"
    --max-num-seqs "${EVALUATOR_MAX_NUM_SEQS}"
    --dtype bfloat16
    --tensor-parallel-size "${EVALUATOR_TENSOR_PARALLEL_SIZE}"
    --generation-config "${VLLM_GENERATION_CONFIG}"
)
if [[ -n "${EVALUATOR_MAX_MODEL_LEN}" ]]; then
    evaluator_args+=(--max-model-len "${EVALUATOR_MAX_MODEL_LEN}")
fi
if [[ "${EVALUATOR_ENFORCE_EAGER}" == "1" ]]; then
    evaluator_args+=(--enforce-eager)
fi

CUDA_VISIBLE_DEVICES="${EVALUATOR_VISIBLE_DEVICES}" python -m vllm.entrypoints.openai.api_server \
    "${evaluator_args[@]}" \
    > "${EVALUATOR_LOG}" 2>&1 &
EVALUATOR_PID=$!

wait_for_health "debater" "http://${SERVER_HOST}:${DEBATER_PORT}/health" "${DEBATER_LOG}" "${DEBATER_PID}"
wait_for_health "evaluator" "http://${SERVER_HOST}:${EVALUATOR_PORT}/health" "${EVALUATOR_LOG}" "${EVALUATOR_PID}"

echo "Running TG-MAD training..."
python -u tg_mad/train.py \
    --debater_base_url "${DEBATER_URL}" \
    --evaluator_base_url "${EVALUATOR_URL}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --num_epochs "${TRAIN_NUM_EPOCHS}" \
    --train_size "${TRAIN_SIZE}" \
    --n_agents "${TRAIN_N_AGENTS}" \
    --n_rounds "${TRAIN_N_ROUNDS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --evaluator_max_new_tokens "${EVALUATOR_MAX_NEW_TOKENS}" \
    --evaluator_model "${EVALUATOR_ENGINE_MODEL}" \
    --seed "${TRAIN_SEED}" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete."
date
