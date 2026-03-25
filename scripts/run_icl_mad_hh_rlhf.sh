#!/bin/bash

# Generate ICL-MAD baseline for hh_rlhf eval pool (300 samples, QA mode).
# Requires larger max_model_len for the ~32K-token ICL system prompt.

#SBATCH -p PA100q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o job-icl-mad-hh-rlhf-%j.output
#SBATCH --time=24:00:00
#SBATCH -J icl-mad-hh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/export/home3/dazhou/debate-or-vote}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Allow vLLM to use context beyond max_position_embeddings (32768).
# Qwen2.5 supports up to 128K via YaRN RoPE scaling.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
DEBATER_PORT="${DEBATER_PORT:-8000}"
DEBATER_MODEL_NAME="${DEBATER_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
DEBATER_MODEL="${DEBATER_MODEL:-hosted_vllm/${DEBATER_MODEL_NAME}}"
DEBATER_BASE_URL="${DEBATER_BASE_URL:-http://${SERVER_HOST}:${DEBATER_PORT}/v1}"
DEBATER_GPU_MEMORY="${DEBATER_GPU_MEMORY:-0.95}"
DEBATER_MIN_FREE_MIB="${DEBATER_MIN_FREE_MIB:-30000}"
DEBATER_MAX_NUM_SEQS="${DEBATER_MAX_NUM_SEQS:-1}"
# Larger context for ICL prompt (~32K tokens for 60 hh_rlhf examples + question + debate + generation)
# 49152 is enough for ~40K actual tokens and fits on 40GB A100 at 0.95 utilization
DEBATER_MAX_MODEL_LEN="${DEBATER_MAX_MODEL_LEN:-49152}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"

ICL_MODE="${ICL_MODE:-qa}"
ICL_MAX_EXAMPLES="${ICL_MAX_EXAMPLES:-}"

DATA_DIR="${DATA_DIR:-./datasets}"
N_AGENTS="${N_AGENTS:-5}"
N_ROUNDS="${N_ROUNDS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
SEED="${SEED:-42}"

VLLM_LOG_DIR="${VLLM_LOG_DIR:-out/logs}"
mkdir -p "${VLLM_LOG_DIR}"
VLLM_LOG_PATH="${VLLM_LOG_DIR}/icl_mad_hh_rlhf_vllm_${SLURM_JOB_ID:-$$}.log"

SERVER_PID=""

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}

pick_debater_gpu() {
    local pick_output
    if ! pick_output="$("${PYTHON_BIN}" - "${DEBATER_MIN_FREE_MIB}" <<'PY'
import sys
import torch

min_free_mib = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable.")

rows = []
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    rows.append((index, free_bytes // (1024 * 1024), total_bytes // (1024 * 1024)))

if not rows:
    raise SystemExit("No visible GPUs.")

rows.sort(key=lambda row: (row[1], row[2], -row[0]), reverse=True)
best = rows[0]
if best[1] < min_free_mib:
    raise SystemExit(f"Best GPU {best[0]} only has {best[1]} MiB free, below required {min_free_mib} MiB.")

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
        echo "Port busy; switched to ${DEBATER_PORT}."
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
    local server_cmd=(
        "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
        --host "${SERVER_HOST}"
        --port "${DEBATER_PORT}"
        --model "${DEBATER_MODEL_NAME}"
        --download-dir ./models
        --gpu-memory-utilization "${DEBATER_GPU_MEMORY}"
        --max-num-seqs "${DEBATER_MAX_NUM_SEQS}"
        --dtype bfloat16
        --tensor-parallel-size 1
        --generation-config vllm
        --max-model-len "${DEBATER_MAX_MODEL_LEN}"
        --enforce-eager
    )
    printf '+ '
    printf '%q ' "${server_cmd[@]}"
    printf '> %q 2>&1 &\n' "${VLLM_LOG_PATH}"
    "${server_cmd[@]}" >"${VLLM_LOG_PATH}" 2>&1 &
    SERVER_PID=$!
    wait_for_server
}

trap cleanup EXIT

echo "=== ICL-MAD Baseline Generation (hh_rlhf, ${ICL_MODE}) ==="
echo "Node: $(hostname)"
echo "Repo: ${REPO_ROOT}"
echo "DEBATER_MAX_MODEL_LEN: ${DEBATER_MAX_MODEL_LEN}"

pick_debater_gpu
ensure_port_available
start_debater_server

GEN_ARGS=(
    -u -m tg_mad.generate_history
    --dataset hh_rlhf
    --pool eval
    --debater_base_url "${DEBATER_BASE_URL}"
    --debater_model "${DEBATER_MODEL}"
    --data_dir "${DATA_DIR}"
    --n_agents "${N_AGENTS}"
    --n_rounds "${N_ROUNDS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --seed "${SEED}"
    --icl_mode "${ICL_MODE}"
    --allow_failed_generations
)

if [[ -n "${ICL_MAX_EXAMPLES}" ]]; then
    GEN_ARGS+=(--icl_max_examples "${ICL_MAX_EXAMPLES}")
fi

echo "Running: ${PYTHON_BIN} ${GEN_ARGS[*]}"
"${PYTHON_BIN}" "${GEN_ARGS[@]}"

echo "=== ICL-MAD baseline generation complete ==="
