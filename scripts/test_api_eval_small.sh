#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o job-test-api-eval-%j.output
#SBATCH --time=01:00:00
#SBATCH -J test-api

# Small test: 1 sample, batch_size=1, 1 epoch
# Only debater on GPU, evaluator via kimi-k2.5 API

set -euo pipefail

echo "=== Test: API Evaluator (small) ==="
echo "Node: $(hostname)"
date

# Load API key
source .env
if [[ -z "${KIMI_API_KEY:-}" ]]; then
    echo "ERROR: KIMI_API_KEY not set. Check .env file."
    exit 1
fi
echo "KIMI_API_KEY loaded (length=${#KIMI_API_KEY})"

export OPENAI_API_KEY=EMPTY
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_TOKEN
HF_TOKEN="$(cat ./token 2>/dev/null || echo "")"

# Start debater vLLM (only model on GPU — full memory)
DEBATER_LOG="out/tg_mad/vllm_debater_test.log"
mkdir -p out/tg_mad

echo "Starting debater vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --download-dir ./models \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --generation-config vllm \
    > "${DEBATER_LOG}" 2>&1 &
DEBATER_PID=$!

cleanup() {
    kill "${DEBATER_PID}" 2>/dev/null || true
    wait "${DEBATER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server
echo "Waiting for debater server..."
WAITED=0
while (( WAITED < 300 )); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "Debater server ready!"
        break
    fi
    if ! kill -0 "${DEBATER_PID}" 2>/dev/null; then
        echo "ERROR: Debater server died. Log:"
        tail -30 "${DEBATER_LOG}"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  waiting... ${WAITED}/300s"
done

# Run training: 1 sample, 1 epoch, batch_size=1
MAIN_REPO="/export/home3/dazhou/debate-or-vote"
EXISTING_DATA="${MAIN_REPO}/out/history/gsm8k/gsm8k_500__qwen3-4b_N=3_R=3.jsonl"

echo ""
echo "=== Running TG-MAD training (1 sample, API evaluator) ==="
python -u tg_mad/train.py \
    --evaluator_type api \
    --debater_base_url http://127.0.0.1:8000/v1 \
    --existing_data "${EXISTING_DATA}" \
    --data_dir "${MAIN_REPO}/datasets" \
    --train_size 1 \
    --batch_size 1 \
    --num_epochs 1 \
    --max_new_tokens 512 \
    --evaluator_max_new_tokens 2048 \
    --seed 42 \
    --output_dir out/tg_mad_test_api \
    --allow_failed_generations

echo ""
echo "=== Test complete ==="
echo "Check out/tg_mad_test_api/prompt_history.json for results"
date
