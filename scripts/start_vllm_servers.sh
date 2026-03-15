#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o job-vllm-servers-%j.output
#SBATCH --time=24:00:00
#SBATCH -J vllm-tgmad

set -euo pipefail

echo "=== Starting vLLM servers for TG-MAD ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

# Load HF token
export HF_TOKEN=$(cat ./token 2>/dev/null || echo "")
export OPENAI_API_KEY=EMPTY

# Fail fast instead of attempting a network install inside the SLURM job.
if ! python -c "import vllm" >/dev/null 2>&1; then
    echo "ERROR: vllm is not installed in the current environment."
    echo "Install it in your user environment, for example:"
    echo "  INSTALL_VLLM=1 bash scripts/setup_textgrad_env.sh"
    exit 1
fi

# Start debater model (Qwen3-4B-Instruct-2507) on port 8000
echo "Starting debater model (Qwen3-4B) on port 8000..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --download-dir ./models \
    --port 8000 \
    --gpu-memory-utilization 0.35 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 &

DEBATER_PID=$!
echo "Debater server PID: $DEBATER_PID"

# Start evaluator model (Qwen3-8B) on port 8001
echo "Starting evaluator model (Qwen3-8B) on port 8001..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --download-dir ./models \
    --port 8001 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 &

EVALUATOR_PID=$!
echo "Evaluator server PID: $EVALUATOR_PID"

# Health check: wait until both servers respond
echo "Waiting for servers to be ready..."
MAX_WAIT=300  # 5 minutes
WAITED=0
DEBATER_READY=false
EVALUATOR_READY=false

while [ "$WAITED" -lt "$MAX_WAIT" ]; do
    if [ "$DEBATER_READY" = false ]; then
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "  Debater server (port 8000) is ready!"
            DEBATER_READY=true
        fi
    fi
    if [ "$EVALUATOR_READY" = false ]; then
        if curl -s http://localhost:8001/health > /dev/null 2>&1; then
            echo "  Evaluator server (port 8001) is ready!"
            EVALUATOR_READY=true
        fi
    fi
    if [ "$DEBATER_READY" = true ] && [ "$EVALUATOR_READY" = true ]; then
        echo "Both servers are ready!"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Waiting... (${WAITED}s / ${MAX_WAIT}s)"
done

if [ "$DEBATER_READY" = false ] || [ "$EVALUATOR_READY" = false ]; then
    echo "ERROR: Servers did not start within ${MAX_WAIT}s"
    kill $DEBATER_PID $EVALUATOR_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=== vLLM servers running ==="
echo "Debater:   http://localhost:8000/v1  (PID $DEBATER_PID)"
echo "Evaluator: http://localhost:8001/v1  (PID $EVALUATOR_PID)"
echo ""
echo "To run training:   python -u tg_mad/train.py"
echo "To run evaluation: python -u tg_mad/evaluate.py"
echo ""

# Keep the script running so servers stay alive
wait $DEBATER_PID $EVALUATOR_PID
