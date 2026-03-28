#!/bin/bash

# SLURM baseline generation for MMLU formal_logic.
# Generates both the train (19) and eval (126) legacy-format baseline histories.

#SBATCH -p PA100q
#SBATCH --gres=gpu:3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o job-baseline-formal-logic-%j.output
#SBATCH --time=24:00:00
#SBATCH -J baseline-formal-logic

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/export/home3/dazhou/debate-or-vote}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-}"

apply_experiment_profile_defaults() {
    if [[ -z "${EXPERIMENT_PROFILE}" ]]; then
        return
    fi
    while IFS='=' read -r key value; do
        [[ -z "${key}" ]] && continue
        if [[ -z "${!key:-}" ]]; then
            export "${key}=${value}"
        fi
    done < <("${PYTHON_BIN}" - "${EXPERIMENT_PROFILE}" <<'PY'
from tg_mad.experiment_profiles import get_stage_env_defaults
import sys

for key, value in get_stage_env_defaults(sys.argv[1], "baseline").items():
    print(f"{key}={value}")
PY
)
}

apply_experiment_profile_defaults

# Keep defaults aligned with the locked baseline setup as closely as possible.
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
DEBATER_PORT="${DEBATER_PORT:-8000}"
DEBATER_MODEL_NAME="${DEBATER_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
DEBATER_MODEL="${DEBATER_MODEL:-hosted_vllm/${DEBATER_MODEL_NAME}}"
DEBATER_BASE_URL="${DEBATER_BASE_URL:-http://${SERVER_HOST}:${DEBATER_PORT}/v1}"
START_DEBATER_SERVER="${START_DEBATER_SERVER:-1}"
DEBATER_GPU_MEMORY="${DEBATER_GPU_MEMORY:-0.45}"
DEBATER_MIN_FREE_MIB="${DEBATER_MIN_FREE_MIB:-18000}"
DEBATER_MAX_NUM_SEQS="${DEBATER_MAX_NUM_SEQS:-1}"
DEBATER_MAX_MODEL_LEN="${DEBATER_MAX_MODEL_LEN:-16384}"
DEBATER_DTYPE="${DEBATER_DTYPE:-bfloat16}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"

DATA_DIR="${DATA_DIR:-./datasets}"
DATASET="${DATASET:-formal_logic}"
N_AGENTS="${N_AGENTS:-5}"
N_ROUNDS="${N_ROUNDS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
SEED="${SEED:-42}"
ALLOW_FAILED_GENERATIONS="${ALLOW_FAILED_GENERATIONS:-0}"

TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-}"
RUN_TRAIN_POOL="${RUN_TRAIN_POOL:-1}"
RUN_EVAL_POOL="${RUN_EVAL_POOL:-1}"

VLLM_LOG_DIR="${VLLM_LOG_DIR:-out/logs}"
mkdir -p "${VLLM_LOG_DIR}"
VLLM_LOG_PATH="${VLLM_LOG_DIR}/formal_logic_baseline_vllm_${SLURM_JOB_ID:-$$}.log"

SERVER_PID=""

run_cmd() {
    printf '+'
    for arg in "$@"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}

pick_debater_gpu() {
    if [[ "${START_DEBATER_SERVER}" != "1" ]]; then
        return
    fi
    local pick_output
    if ! pick_output="$("${PYTHON_BIN}" - "${DEBATER_MIN_FREE_MIB}" <<'PY'
import sys
import torch

min_free_mib = int(sys.argv[1])

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this baseline job.")

rows = []
device_count = torch.cuda.device_count()
if device_count == 1:
    # In a single-GPU SLURM allocation, trust the assigned device and let the
    # server startup perform the real readiness check.
    print("0 0 0")
    raise SystemExit(0)

for index in range(device_count):
    try:
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        continue
    rows.append((index, free_bytes // (1024 * 1024), total_bytes // (1024 * 1024)))

if not rows:
    raise SystemExit("No probeable visible GPUs available for baseline generation.")

rows.sort(key=lambda row: (row[1], row[2], -row[0]), reverse=True)
best = rows[0]
if best[1] < min_free_mib:
    raise SystemExit(
        f"Best visible GPU {best[0]} only has {best[1]} MiB free, below required {min_free_mib} MiB."
    )

print(f"{best[0]} {best[1]} {best[2]}")
PY
)"; then
        echo "Failed to select a clean debater GPU."
        return 1
    fi

    local picked_gpu picked_free picked_total
    read -r picked_gpu picked_free picked_total <<<"${pick_output}"
    export CUDA_VISIBLE_DEVICES="${picked_gpu}"
    echo "Selected GPU ${picked_gpu} for debater (${picked_free}/${picked_total} MiB free)."
}

ensure_port_available() {
    if [[ "${START_DEBATER_SERVER}" != "1" ]]; then
        return
    fi
    local candidate_port="${DEBATER_PORT}"
    if ! "${PYTHON_BIN}" - "${SERVER_HOST}" "${candidate_port}" <<'PY'
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
family = socket.AF_INET6 if ":" in host else socket.AF_INET
sock = socket.socket(family, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
    then
        DEBATER_PORT="$("${PYTHON_BIN}" - "${SERVER_HOST}" <<'PY'
import socket, sys
host = sys.argv[1]
family = socket.AF_INET6 if ":" in host else socket.AF_INET
sock = socket.socket(family, socket.SOCK_STREAM)
sock.bind((host, 0))
print(sock.getsockname()[1])
sock.close()
PY
)"
        DEBATER_BASE_URL="http://${SERVER_HOST}:${DEBATER_PORT}/v1"
        echo "Debater port ${candidate_port} is busy; switched to ${DEBATER_PORT}."
    fi
}

wait_for_server() {
    local waited=0
    local health_url="http://${SERVER_HOST}:${DEBATER_PORT}/health"
    echo "Waiting for debater server at ${health_url} ..."
    while (( waited < MAX_WAIT_SECONDS )); do
        if "${PYTHON_BIN}" - "${health_url}" <<'PY'
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import sys
url = sys.argv[1]
try:
    with urlopen(url, timeout=2) as resp:
        raise SystemExit(0 if 200 <= resp.status < 300 else 1)
except (HTTPError, URLError):
    raise SystemExit(1)
PY
        then
            echo "Debater server is healthy."
            return
        fi
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "Debater server exited before becoming healthy. Tail of ${VLLM_LOG_PATH}:"
            tail -n 80 "${VLLM_LOG_PATH}" || true
            exit 1
        fi
        sleep 5
        waited=$((waited + 5))
        echo "  waiting... ${waited}/${MAX_WAIT_SECONDS}s"
    done
    echo "Debater server did not become healthy within ${MAX_WAIT_SECONDS}s."
    tail -n 80 "${VLLM_LOG_PATH}" || true
    exit 1
}

start_debater_server() {
    if [[ "${START_DEBATER_SERVER}" != "1" ]]; then
        echo "Reusing external debater server at ${DEBATER_BASE_URL}"
        return
    fi
    local server_cmd=(
        "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
        --host "${SERVER_HOST}"
        --port "${DEBATER_PORT}"
        --model "${DEBATER_MODEL_NAME}"
        --download-dir ./models
        --gpu-memory-utilization "${DEBATER_GPU_MEMORY}"
        --max-num-seqs "${DEBATER_MAX_NUM_SEQS}"
        --dtype "${DEBATER_DTYPE}"
        --tensor-parallel-size 1
        --generation-config vllm
        --max-model-len "${DEBATER_MAX_MODEL_LEN}"
        --enforce-eager
    )
    printf '+'
    for arg in "${server_cmd[@]}"; do
        printf ' %q' "${arg}"
    done
    printf ' > %q 2>&1 &\n' "${VLLM_LOG_PATH}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi
    "${server_cmd[@]}" >"${VLLM_LOG_PATH}" 2>&1 &
    SERVER_PID=$!
    wait_for_server
}

trap cleanup EXIT

pick_debater_gpu
ensure_port_available
start_debater_server

COMMON_ARGS=(
    -m tg_mad.generate_history
    --dataset "${DATASET}"
    --debater_base_url "${DEBATER_BASE_URL}"
    --debater_model "${DEBATER_MODEL}"
    --data_dir "${DATA_DIR}"
    --n_agents "${N_AGENTS}"
    --n_rounds "${N_ROUNDS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --seed "${SEED}"
)

if [[ -n "${EXPERIMENT_PROFILE}" ]]; then
    COMMON_ARGS+=(--experiment_profile "${EXPERIMENT_PROFILE}")
fi

if [[ "${ALLOW_FAILED_GENERATIONS}" == "1" ]]; then
    COMMON_ARGS+=(--allow_failed_generations)
fi

if [[ "${RUN_TRAIN_POOL}" == "1" ]]; then
    TRAIN_ARGS=("${COMMON_ARGS[@]}" --pool train)
    if [[ -n "${TRAIN_OUTPUT_PATH}" ]]; then
        TRAIN_ARGS+=(--output_path "${TRAIN_OUTPUT_PATH}")
    fi
    run_cmd "${PYTHON_BIN}" "${TRAIN_ARGS[@]}"
fi

if [[ "${RUN_EVAL_POOL}" == "1" ]]; then
    EVAL_ARGS=("${COMMON_ARGS[@]}" --pool eval)
    if [[ -n "${EVAL_OUTPUT_PATH}" ]]; then
        EVAL_ARGS+=(--output_path "${EVAL_OUTPUT_PATH}")
    fi
    run_cmd "${PYTHON_BIN}" "${EVAL_ARGS[@]}"
fi
