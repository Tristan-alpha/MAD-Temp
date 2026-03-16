#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o job-tgmad-5s-api-%j.output
#SBATCH --time=06:00:00
#SBATCH -J tgmad-5s

# Training: 5 samples, batch_size=1 (5 optimizer steps), n_rounds=1
# Round 0 = MV accuracy, Round 1 = MAD accuracy
# API evaluator (kimi-k2.5), debater on local GPU

set -euo pipefail

OUTPUT_DIR="out/tg_mad_5s_api"
MAIN_REPO="/export/home3/dazhou/debate-or-vote"
EXISTING_DATA="${MAIN_REPO}/out/history/gsm8k/gsm8k_500__qwen3-4b_N=3_R=3.jsonl"

echo "=== TG-MAD Training (5 samples, API eval, n_rounds=1) ==="
echo "Node: $(hostname)"
date

# Load API key
source .env
if [[ -z "${KIMI_API_KEY:-}" ]]; then
    echo "ERROR: KIMI_API_KEY not set"; exit 1
fi
echo "KIMI_API_KEY loaded"

export OPENAI_API_KEY=EMPTY
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_TOKEN
HF_TOKEN="$(cat ./token 2>/dev/null || echo "")"

mkdir -p "${OUTPUT_DIR}"
DEBATER_LOG="${OUTPUT_DIR}/vllm_debater.log"

cleanup() {
    kill "${DEBATER_PID:-}" 2>/dev/null || true
    wait "${DEBATER_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

# Start debater vLLM
echo "Starting debater vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --download-dir ./models \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --dtype bfloat16 \
    --enforce-eager \
    --generation-config vllm \
    > "${DEBATER_LOG}" 2>&1 &
DEBATER_PID=$!

WAITED=0
while (( WAITED < 300 )); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "Debater server ready!"; break
    fi
    if ! kill -0 "${DEBATER_PID}" 2>/dev/null; then
        echo "ERROR: Debater died"; tail -30 "${DEBATER_LOG}"; exit 1
    fi
    sleep 5; WAITED=$((WAITED + 5))
    echo "  waiting... ${WAITED}/300s"
done

# === TRAINING ===
echo ""
echo "========================================="
echo "  PHASE 1: TRAINING (5 samples, 1 epoch)"
echo "========================================="
python -u tg_mad/train.py \
    --evaluator_type api \
    --debater_base_url http://127.0.0.1:8000/v1 \
    --existing_data "${EXISTING_DATA}" \
    --data_dir "${MAIN_REPO}/datasets" \
    --train_size 5 \
    --batch_size 1 \
    --num_epochs 1 \
    --n_rounds 1 \
    --max_new_tokens 512 \
    --evaluator_max_new_tokens 2048 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --allow_failed_generations

echo ""
echo "Training complete. Prompt history saved."

# === EVALUATION ===
echo ""
echo "========================================="
echo "  PHASE 2: EVALUATION (test set)"
echo "========================================="
python -u tg_mad/evaluate.py \
    --debater_base_url http://127.0.0.1:8000/v1 \
    --existing_data "${EXISTING_DATA}" \
    --data_dir "${MAIN_REPO}/datasets" \
    --prompt_history "${OUTPUT_DIR}/prompt_history.json" \
    --output_dir "${OUTPUT_DIR}" \
    --n_rounds 1 \
    --max_new_tokens 512 \
    --seed 42 \
    --allow_failed_generations

echo ""
echo "========================================="
echo "  ALL DONE"
echo "========================================="
date
