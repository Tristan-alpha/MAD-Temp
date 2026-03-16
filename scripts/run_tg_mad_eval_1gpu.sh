#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o job-tgmad-eval-1gpu-%j.output
#SBATCH --time=24:00:00
#SBATCH -J tgmad-eval1

set -euo pipefail

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
DEBATER_PORT="${DEBATER_PORT:-8000}"
DEBATER_URL="http://${SERVER_HOST}:${DEBATER_PORT}/v1"
OUTPUT_DIR="${OUTPUT_DIR:-out/tg_mad}"
PROMPT_HISTORY_PATH="${PROMPT_HISTORY_PATH:-${OUTPUT_DIR%/}/prompt_history.json}"
SPLIT_INFO_PATH="${SPLIT_INFO_PATH:-}"
RESULTS_FILE_PATH="${RESULTS_FILE_PATH:-}"
RUN_CONFIG_FILE_PATH="${RUN_CONFIG_FILE_PATH:-}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_N_AGENTS="${EVAL_N_AGENTS:-3}"
EVAL_N_ROUNDS="${EVAL_N_ROUNDS:-3}"
EVAL_MAX_TEST_SAMPLES="${EVAL_MAX_TEST_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SAVE_TEXT_HISTORY="${SAVE_TEXT_HISTORY:-0}"
TEXT_HISTORY_DIR="${TEXT_HISTORY_DIR:-}"
JOB_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
DEBATER_GPU_SLOTS="${DEBATER_GPU_SLOTS:-0}"
DEBATER_CUDA_VISIBLE_DEVICES="${DEBATER_CUDA_VISIBLE_DEVICES:-}"
DEBATER_MODEL_NAME="${DEBATER_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
DEBATER_GPU_MEMORY="${DEBATER_GPU_MEMORY:-0.45}"
DEBATER_MAX_MODEL_LEN="${DEBATER_MAX_MODEL_LEN:-8192}"
DEBATER_MAX_NUM_SEQS="${DEBATER_MAX_NUM_SEQS:-4}"
DEBATER_TENSOR_PARALLEL_SIZE="${DEBATER_TENSOR_PARALLEL_SIZE:-1}"
DEBATER_ENFORCE_EAGER="${DEBATER_ENFORCE_EAGER:-1}"
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-vllm}"

DEBATER_LOG="${OUTPUT_DIR%/}/vllm_debater_eval.log"

cleanup() {
    local exit_code=$?
    if [[ -n "${DEBATER_PID:-}" ]]; then
        kill "${DEBATER_PID}" 2>/dev/null || true
        wait "${DEBATER_PID}" 2>/dev/null || true
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

echo "=== TG-MAD Evaluation ==="
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

wait_for_health "debater" "http://${SERVER_HOST}:${DEBATER_PORT}/health" "${DEBATER_LOG}" "${DEBATER_PID}"

echo "Running TG-MAD evaluation..."
eval_args=(
    --debater_base_url "${DEBATER_URL}"
    --prompt_history "${PROMPT_HISTORY_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --n_agents "${EVAL_N_AGENTS}"
    --n_rounds "${EVAL_N_ROUNDS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --seed "${EVAL_SEED}"
)
if [[ -n "${SPLIT_INFO_PATH}" ]]; then
    eval_args+=(--split_info_file "${SPLIT_INFO_PATH}")
fi
if [[ -n "${RESULTS_FILE_PATH}" ]]; then
    eval_args+=(--results_file "${RESULTS_FILE_PATH}")
fi
if [[ -n "${RUN_CONFIG_FILE_PATH}" ]]; then
    eval_args+=(--run_config_file "${RUN_CONFIG_FILE_PATH}")
fi
if [[ -n "${EVAL_MAX_TEST_SAMPLES}" ]]; then
    eval_args+=(--max_test_samples "${EVAL_MAX_TEST_SAMPLES}")
fi
if [[ "${SAVE_TEXT_HISTORY}" == "1" ]]; then
    eval_args+=(--save_text_history)
fi
if [[ -n "${TEXT_HISTORY_DIR}" ]]; then
    eval_args+=(--text_history_dir "${TEXT_HISTORY_DIR}")
fi

python -u tg_mad/evaluate.py "${eval_args[@]}"

echo "Evaluation complete."
date
